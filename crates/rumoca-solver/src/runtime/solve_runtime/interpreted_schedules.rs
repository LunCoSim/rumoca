//! Interpreted execution of exact refresh assignment schedules through their
//! shared-value segments (SPEC_0043 §6a): the construction the JIT and the
//! generated C also execute, each segment committing its targets before the
//! next runs.

use std::cell::RefCell;
use std::rc::Rc;

use rumoca_eval_solve::{EvalSolveError, PreparedEvaluationBlock};
use rumoca_ir_solve as solve;
use rustc_hash::FxHashMap;

use super::SolveRuntime;
use crate::RuntimeSolveError;

/// One schedule prepared for the interpreter: its segments as programs and
/// the solver slot of every output, in segment order.
pub(crate) struct InterpretedSchedule {
    block: PreparedEvaluationBlock,
    shared: solve::SharedAssignmentSchedule,
}

/// The prepared schedules of one runtime, by refresh sequence. Construction
/// proves each issued schedule once and retains it for interpreter and native
/// preparation; a sequence without an issued schedule has no entry.
#[derive(Default)]
pub(crate) struct InterpretedSchedules {
    prepared: FxHashMap<solve::RefreshSequenceId, Rc<InterpretedSchedule>>,
    outputs: RefCell<Vec<f64>>,
}

impl Clone for InterpretedSchedules {
    /// A clone shares the prepared schedules and owns its output scratch.
    fn clone(&self) -> Self {
        Self {
            prepared: self.prepared.clone(),
            outputs: RefCell::default(),
        }
    }
}

impl InterpretedSchedules {
    pub(super) fn shared(
        &self,
        sequence: solve::RefreshSequenceId,
    ) -> Option<&solve::SharedAssignmentSchedule> {
        self.prepared
            .get(&sequence)
            .map(|schedule| &schedule.shared)
    }

    /// Prove and prepare every issued exact assignment schedule of `model`.
    pub(crate) fn construct(model: &solve::SolveModel) -> Result<Self, EvalSolveError> {
        let continuous = &model.problem.continuous;
        let owners = &continuous.refresh_owners;
        let mut prepared = FxHashMap::default();
        for schedule in owners.exact_assignment_schedules() {
            let interpreted = prepare_schedule(schedule, &continuous.implicit_rhs, owners)?;
            prepared.insert(schedule.sequence_id(), Rc::new(interpreted));
        }
        Ok(Self {
            prepared,
            outputs: RefCell::default(),
        })
    }
}

impl SolveRuntime {
    /// How many issued exact assignment schedules the interpreter proved and
    /// prepared when this runtime was built.
    #[must_use]
    pub fn interpreted_schedule_count(&self) -> usize {
        self.interpreted_assignment_schedules.prepared.len()
    }

    /// Run the issued schedule of `sequence` through the interpreter; `false`
    /// when the sequence has no issued schedule.
    pub(super) fn try_interpreted_assignment_refresh(
        &self,
        sequence: solve::RefreshSequenceId,
        t: f64,
        solver_y: &mut [f64],
        params: &[f64],
    ) -> Result<bool, RuntimeSolveError> {
        let Some(schedule) = self
            .interpreted_assignment_schedules
            .prepared
            .get(&sequence)
        else {
            return Ok(false);
        };
        let mut outputs = self.interpreted_assignment_schedules.outputs.borrow_mut();
        for (row, targets) in schedule.shared.target_rows().iter().enumerate() {
            schedule.block.eval_row_outputs_unchecked_with_context(
                row,
                solver_y,
                params,
                t,
                self.row_eval_context(),
                &mut outputs,
            )?;
            for (&target, &value) in targets.iter().zip(outputs.iter()) {
                solver_y[target] = value;
            }
        }
        Ok(true)
    }
}

fn prepare_schedule(
    schedule: &solve::ExactRefreshAssignmentSchedule,
    source: &solve::ComputeBlock,
    owners: &solve::ContinuousRefreshOwners,
) -> Result<InterpretedSchedule, EvalSolveError> {
    let invalid = |message: String| EvalSolveError::InvalidRow {
        message,
        span: None,
    };
    let shared = match schedule.shared_segments(source, owners) {
        Ok(shared) => shared,
        Err(error) => return Err(invalid(error.to_string())),
    };
    let block = PreparedEvaluationBlock::new(shared.block().clone())?;
    Ok(InterpretedSchedule { block, shared })
}
