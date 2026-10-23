//! Register fields of the operations a shared-value segment renames, and the
//! exact registers any operation reads.

use super::super::*;

/// How an operation uses one register field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Role {
    /// The first register the operation writes.
    Destination,
    /// One scalar source register.
    Scalar,
    /// The first register of a source range.
    RangeStart,
}

/// Whether a segment renames `op`: a pure value of scalar or range sources.
/// An operation with an effect, a nested body, a table, a seed, or an indexed
/// load is not renamed, nor is a copy or an output store (the segment builder
/// handles those itself). [`visit_registers`] visits every register field of
/// exactly these operations.
pub(super) fn renamable(op: &LinearOp) -> bool {
    match op {
        LinearOp::Const { .. }
        | LinearOp::LoadTime { .. }
        | LinearOp::LoadP { .. }
        | LinearOp::Unary { .. }
        | LinearOp::Binary { .. }
        | LinearOp::Compare { .. }
        | LinearOp::Select { .. }
        | LinearOp::DotProduct { .. }
        | LinearOp::MatrixMultiply { .. }
        | LinearOp::TensorBinary { .. }
        | LinearOp::TensorCross { .. }
        | LinearOp::TensorTranspose { .. }
        | LinearOp::TensorConcatenate { .. }
        | LinearOp::TensorFill { .. }
        | LinearOp::TensorIdentity { .. }
        | LinearOp::TensorLoad {
            input: TensorInputKind::P,
            seed_start: None,
            lanes: 1,
            ..
        } => true,
        LinearOp::PureCall { site, .. } => pure_value_call(site),
        _ => false,
    }
}

/// Visit every register field of a [`renamable`] operation in field order,
/// sources before the destination.
pub(super) fn visit_registers(op: &mut LinearOp, visit: &mut dyn FnMut(Role, &mut Reg)) {
    use Role::{Destination as D, Scalar as S};
    match op {
        LinearOp::Const { dst, .. } | LinearOp::LoadTime { dst } | LinearOp::LoadP { dst, .. } => {
            visit(D, dst);
        }
        LinearOp::Unary { dst, arg, .. } => {
            visit(S, arg);
            visit(D, dst);
        }
        LinearOp::Binary { dst, lhs, rhs, .. } | LinearOp::Compare { dst, lhs, rhs, .. } => {
            visit(S, lhs);
            visit(S, rhs);
            visit(D, dst);
        }
        LinearOp::Select {
            dst,
            cond,
            if_true,
            if_false,
        } => {
            visit(S, cond);
            visit(S, if_true);
            visit(S, if_false);
            visit(D, dst);
        }
        _ => visit_range_registers(op, visit),
    }
}

/// The tensor and call operations, whose sources are register ranges.
fn visit_range_registers(op: &mut LinearOp, visit: &mut dyn FnMut(Role, &mut Reg)) {
    use Role::{Destination as D, RangeStart as R};
    match op {
        LinearOp::DotProduct {
            dst,
            lhs_start,
            rhs_start,
            ..
        } => {
            visit(R, lhs_start);
            visit(R, rhs_start);
            visit(D, dst);
        }
        LinearOp::MatrixMultiply {
            dst_start,
            lhs_start,
            rhs_start,
            ..
        }
        | LinearOp::TensorBinary {
            dst_start,
            lhs_start,
            rhs_start,
            ..
        }
        | LinearOp::TensorCross {
            dst_start,
            lhs_start,
            rhs_start,
            ..
        } => {
            visit(R, lhs_start);
            visit(R, rhs_start);
            visit(D, dst_start);
        }
        LinearOp::TensorTranspose {
            dst_start,
            src_start: start,
            ..
        }
        | LinearOp::TensorFill {
            dst_start,
            value_start: start,
            ..
        } => {
            visit(R, start);
            visit(D, dst_start);
        }
        LinearOp::TensorConcatenate {
            dst_start, sources, ..
        } => {
            for source in sources.iter_mut() {
                visit(R, &mut source.start);
            }
            visit(D, dst_start);
        }
        LinearOp::TensorIdentity { dst_start, .. } | LinearOp::TensorLoad { dst_start, .. } => {
            visit(D, dst_start);
        }
        LinearOp::PureCall {
            dst_start,
            input_starts,
            ..
        } => {
            for start in input_starts.iter_mut() {
                visit(R, start);
            }
            visit(D, dst_start);
        }
        _ => {}
    }
}

/// A call whose outputs are values only: a call carrying assertion
/// predicates has an effect and is never renamed.
fn pure_value_call(site: &SolvePureCallSite) -> bool {
    site.outputs()
        .iter()
        .all(|output| output.kind() == crate::SolvePureCallOutputKind::Result)
}

/// Every register `op` reads, ascending and unique. The source owner's structural
/// checks run once; whole-program validation separately proves dominance.
/// `None` means this operation is invalid outside its enclosing context.
pub(super) fn read_registers(op: &LinearOp) -> Option<Vec<Reg>> {
    let reads = RefCell::new(Vec::new());
    let mut validation = ScalarProgramValidationCache::default();
    let cx = OpSources {
        op,
        op_index: 0,
        registers: SourceRegisters::Collect(&reads),
        fold_context: None,
        conditional_capture_count: None,
        max_lanes: validation.max_tensor_lanes(),
    };
    check_op_sources(cx, &mut validation).ok()?;
    let mut reads = reads.into_inner();
    reads.sort_unstable();
    reads.dedup();
    Some(reads)
}
