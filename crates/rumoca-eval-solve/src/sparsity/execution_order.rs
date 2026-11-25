//! Block order of the algebraic projection under execution reads.
//!
//! Structural analysis orders the blocks by value dependencies. Executing a
//! row reads more than its value does: one tensor program computes every lane
//! of `der(x[i]) = r[i]*(x[i] - uu[i])` and loads `uu` whole, so the row of
//! lane 1 also loads `uu[2]`. A block that reads a target of a later block
//! refuses the staged refresh schedule (`RefreshExecutionRefusal`) and marks
//! that block as invalidating an earlier one, and every refresh then projects
//! block by block. Where the later block does not depend on the reader, the
//! reader can simply run after it.

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};

use rumoca_ir_solve::{
    ComputeNode, ImplicitRowTargets, IndexIntervals, ProjectionBlocks, ProjectionStage, ScalarSlot, SolveProblem,
};

use rumoca_ir_solve::StructuralPattern;
use crate::{EvalSolveError, to_scalar_program_block};

/// Reorder the algebraic projection blocks so the targets a block's execution
/// reads are produced before it, wherever such an order exists. Value
/// dependencies are execution reads too, so the issued order stays a valid
/// one; blocks keep their issued order unless a read moves them, and a read
/// that closes a cycle keeps the issued order and its refusal.
///
/// A declaration run moves as one native store. Its source, storage and lane
/// order stay bound together. A cycle through a run remains subject to the
/// refresh owner’s execution-order check.
pub fn order_algebraic_projection_by_execution_reads(
    problem: &mut SolveProblem,
) -> Result<(), EvalSolveError> {
    let plan = &problem.continuous.algebraic_projection_plan.blocks;
    let stages = plan.stages();
    if stages.len() < 2 {
        return Ok(());
    }
    let rhs = &problem.continuous.implicit_rhs;
    let scalarized;
    let source = match rhs.nodes.as_slice() {
        [ComputeNode::ScalarPrograms(programs)] => programs,
        _ => {
            scalarized = to_scalar_program_block(rhs)?;
            &scalarized
        }
    };
    let ranged_programs = stages.iter().filter_map(|stage| match stage {
        ProjectionStage::SingletonDeclarationRun { first_row, storage, .. } if storage.scalar_count > 0 =>
            source.program_index_for_output(*first_row),
        _ => None,
    }).collect::<BTreeSet<_>>();
    let reads = source.programs().iter().enumerate()
        .map(|(index, program)| {
            let span = source.program_span(index);
            if ranged_programs.contains(&index) {
                StructuralPattern::derive_execution_y_dependency_ranges(program, span)
                    .map(ExecutionReads::Whole)
            } else {
                StructuralPattern::derive_output_execution_y_dependency_ranges(program, span)
                    .map(ExecutionReads::Outputs)
            }.map_err(|error| super::from_pattern_error(error, span))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let Some(order) = execution_order(stages, |row| {
        let (program, offset) = source.output_position(row)?;
        match reads.get(program)? {
            ExecutionReads::Whole(reads) => Some(reads.as_ref()),
            ExecutionReads::Outputs(outputs) => outputs.get(offset).map(AsRef::as_ref),
        }
    }) else {
        return Ok(());
    };
    let shared_targets = problem
        .continuous
        .implicit_row_targets
        .shares_projection_owner(plan);
    let reordered = ProjectionBlocks::from_stages(
        order.into_iter().map(|index| stages[index].clone()).collect(),
    )?;
    if shared_targets {
        problem.continuous.implicit_row_targets = ImplicitRowTargets::from_projection(&reordered);
    }
    problem.continuous.algebraic_projection_plan.blocks = reordered;
    Ok(())
}

/// A native store executes all lanes together. Explicit row assignments own
/// one output's value dependencies and the obligations of its source program.
enum ExecutionReads {
    Whole(std::sync::Arc<IndexIntervals>),
    Outputs(Vec<std::sync::Arc<IndexIntervals>>),
}

/// Build an index over the disjoint target intervals of the issued stages.
fn producer_intervals(stages: &[ProjectionStage]) -> BTreeMap<usize, (usize, usize)> {
    let mut producers = BTreeMap::new();
    for (owner, stage) in stages.iter().enumerate() {
        match stage {
            ProjectionStage::Explicit(block) => {
                for &target in &block.y_indices {
                    producers.insert(target, (target, owner));
                }
            }
            ProjectionStage::SingletonDeclarationRun { storage, .. } => {
                if let ScalarSlot::Y { index, .. } = storage.base
                    && storage.scalar_count != 0
                {
                    producers.insert(index, (index + (storage.scalar_count - 1), owner));
                }
            }
        }
    }
    producers
}

fn execution_order<'a>(
    stages: &[ProjectionStage],
    reads: impl Fn(usize) -> Option<&'a IndexIntervals>,
) -> Option<Vec<usize>> {
    let producers = producer_intervals(stages);
    let mut needs = Vec::with_capacity(stages.len());
    for (owner, stage) in stages.iter().enumerate() {
        let mut needed = BTreeSet::new();
        let mut include_row = |row: usize| {
            let Some(columns) = reads(row) else { return };
            for interval in columns.intervals() {
                let start = *interval.start();
                // A read can begin inside a producer's declaration run.
                if let Some((_, &(end, producer))) = producers.range(..start).next_back()
                    && end >= start
                {
                    needed.insert(producer);
                }
                needed.extend(producers.range(interval).map(|(_, &(_, producer))| producer));
            }
        };
        match stage {
            ProjectionStage::Explicit(block) => block.rows.iter().copied().for_each(include_row),
            // All lanes of the declaration are issued by the same source
            // program; the projection source owns its complete read set.
            ProjectionStage::SingletonDeclarationRun { first_row, storage, .. } => {
                if storage.scalar_count != 0 { include_row(*first_row); }
            }
        }
        needed.remove(&owner);
        needs.push(needed);
    }
    if needs
        .iter()
        .enumerate()
        .all(|(block, needed)| needed.iter().all(|&producer| producer < block))
    {
        return None;
    }
    let mut placed = vec![false; stages.len()];
    let mut order = Vec::with_capacity(stages.len());
    for block in 0..stages.len() {
        // Hoist the producers this block still waits for, each as soon as its
        // own reads are produced. What is left waiting is an execution cycle:
        // the block then keeps its issued place, which every dependency on an
        // earlier block already allows.
        while !placed[block] {
            let next = ready_producer(block, &needs, &placed).unwrap_or(block);
            placed[next] = true;
            order.push(next);
        }
    }
    order
        .iter()
        .enumerate()
        .any(|(position, &block)| position != block)
        .then_some(order)
}

/// The first block in issued order, among `block` and the unplaced producers
/// it waits for directly or through them, whose own producers are all placed.
fn ready_producer(block: usize, needs: &[BTreeSet<usize>], placed: &[bool]) -> Option<usize> {
    let mut reached = BTreeSet::from([block]);
    let mut pending = vec![block];
    let mut ready = None::<usize>;
    while let Some(waiting) = pending.pop() {
        let mut unmet = needs[waiting].iter().filter(|&&producer| !placed[producer]);
        if unmet.clone().next().is_none() {
            ready = Some(ready.map_or(waiting, |ready| ready.min(waiting)));
            continue;
        }
        pending.extend(unmet.by_ref().filter(|&&producer| reached.insert(producer)));
    }
    ready
}
