//! The shared-value segments of one exact refresh assignment schedule
//! (SPEC_0043 §6a): one checked construction shared by its runtime executors.

use std::sync::Arc;

use super::{
    ContinuousRefreshConstructionError, ContinuousRefreshOwners, ExactRefreshAssignmentSchedule,
};
use crate::{AssignmentProgram, ComputeBlock, ScalarProgramBlock, SharedValueSegments};

/// A proven assignment schedule: shared programs, exact provenance, and each
/// program's solver targets in output order. Program storage is immutable and
/// shared with prepared interpreter and native consumers.
#[derive(Clone, Debug)]
pub struct SharedAssignmentSchedule {
    block: ScalarProgramBlock,
    target_rows: Arc<[Box<[usize]>]>,
}

impl SharedAssignmentSchedule {
    /// Immutable segment programs with dense output identities and the
    /// provenance of each segment's first source program.
    #[must_use]
    pub fn block(&self) -> &ScalarProgramBlock {
        &self.block
    }

    /// Solver targets for each stored program, in its output order.
    #[must_use]
    pub fn target_rows(&self) -> &[Box<[usize]>] {
        &self.target_rows
    }
}

impl ExactRefreshAssignmentSchedule {
    /// The shared-value segments of this schedule's final programs, in
    /// schedule order. The checker proves them equivalent to the programs at
    /// construction; executors run the segments in order.
    pub fn shared_segments(
        &self,
        source: &ComputeBlock,
        owners: &ContinuousRefreshOwners,
    ) -> Result<SharedAssignmentSchedule, ContinuousRefreshConstructionError> {
        let mut blocks = Vec::with_capacity(self.program_ids().len());
        let mut targets = Vec::with_capacity(self.program_ids().len());
        for id in self.program_ids() {
            let Some(program) = owners.exact_assignment_program(*id) else {
                return Err(ContinuousRefreshConstructionError {
                    reason: "exact assignment schedule refers to a missing program".to_string(),
                });
            };
            let block = program.final_scalar_program(source)?;
            if block.programs().len() != 1 {
                return Err(ContinuousRefreshConstructionError {
                    reason: "exact assignment owner is not one correlated program".to_string(),
                });
            }
            blocks.push(block);
            targets.push(program.target_indices());
        }
        let programs = blocks
            .iter()
            .zip(&targets)
            .map(|(block, targets)| AssignmentProgram {
                ops: &block.programs()[0],
                targets,
            })
            .collect::<Vec<_>>();
        let segments = SharedValueSegments::derive(&programs).map_err(|error| {
            ContinuousRefreshConstructionError {
                reason: format!("shared-value segments cannot be constructed: {error:?}"),
            }
        })?;
        // The checker is the construction proof that the segments compute the
        // programs' values; it runs once here, never while executing them.
        if let Err(error) = segments.check(&programs) {
            return Err(ContinuousRefreshConstructionError {
                reason: format!("shared-value segments fail their proof: {error:?}"),
            });
        }
        let spans = segments
            .segments()
            .iter()
            .map(|segment| blocks[segment.first_program()].program_span(0))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| ContinuousRefreshConstructionError {
                reason: "exact assignment owner has no provenance".to_string(),
            })?;
        let (programs, target_rows): (Vec<_>, Vec<_>) = segments.into_assignment_rows().unzip();
        let block = ScalarProgramBlock::with_program_spans(programs, spans).map_err(|error| {
            ContinuousRefreshConstructionError {
                reason: format!("shared assignment programs fail construction: {error}"),
            }
        })?;
        Ok(SharedAssignmentSchedule {
            block,
            target_rows: target_rows.into(),
        })
    }
}
