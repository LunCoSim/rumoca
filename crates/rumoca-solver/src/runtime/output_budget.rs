//! Scoped admission for a simulation's retained scalar output storage.
use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use super::solve_ops::RuntimeSolveError;

/// Scalar values, including the separate recorded time vector.
#[derive(Clone, Copy, Debug)]
pub struct SolverOutputBudget {
    max_values: usize,
}
impl SolverOutputBudget {
    pub fn new(max_values: usize) -> Result<Self, RuntimeSolveError> {
        if max_values == 0 {
            return Err(RuntimeSolveError::solve_ir(
                "solver output budget must be positive",
            ));
        }
        Ok(Self { max_values })
    }

    pub fn validate_dimensions(
        self,
        outputs: usize,
        samples: usize,
    ) -> Result<(), RuntimeSolveError> {
        let values = outputs
            .checked_add(1)
            .and_then(|columns| columns.checked_mul(samples))
            .ok_or_else(|| RuntimeSolveError::solve_ir("solver output dimensions overflow"))?;
        if values > self.max_values {
            return Err(RuntimeSolveError::solve_ir(format!(
                "solver output requires {values} scalar values; configured output budget is {}",
                self.max_values
            )));
        }
        Ok(())
    }
    pub fn sample_capacity(self, outputs: usize) -> Result<usize, RuntimeSolveError> {
        let columns = outputs
            .checked_add(1)
            .ok_or_else(|| RuntimeSolveError::solve_ir("solver output dimensions overflow"))?;
        Ok(self.max_values / columns)
    }
}

struct OutputBudgetState {
    budget: SolverOutputBudget,
}
thread_local! {
    static OUTPUT_BUDGET: RefCell<Option<OutputBudgetState>> = const { RefCell::new(None) };
}

/// One admitted solve's immutable budget. Nested calls restore their caller;
/// this guard cannot move to a different thread or Web Worker.
/// Each recorder validates its actual dimensions before reserving or appending.
pub struct SolverOutputBudgetGuard {
    previous: Option<OutputBudgetState>,
    _thread: PhantomData<Rc<()>>,
}
impl SolverOutputBudgetGuard {
    pub fn install(budget: SolverOutputBudget) -> Self {
        let previous = OUTPUT_BUDGET.with(|slot| slot.replace(Some(OutputBudgetState { budget })));
        Self {
            previous,
            _thread: PhantomData,
        }
    }
}
impl Drop for SolverOutputBudgetGuard {
    fn drop(&mut self) {
        OUTPUT_BUDGET.with(|slot| {
            slot.replace(self.previous.take());
        });
    }
}

pub fn validate_solver_output_dimensions(
    outputs: usize,
    samples: usize,
) -> Result<(), RuntimeSolveError> {
    OUTPUT_BUDGET.with(|slot| match slot.borrow().as_ref() {
        Some(state) => state.budget.validate_dimensions(outputs, samples),
        None => Ok(()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn output_admission_counts_time_and_rejects_overflow() {
        let budget = SolverOutputBudget::new(6).unwrap();
        assert!(budget.validate_dimensions(2, 2).is_ok());
        assert!(budget.validate_dimensions(2, 3).is_err());
        assert!(budget.validate_dimensions(usize::MAX, 1).is_err());
        assert!(SolverOutputBudget::new(0).is_err());
    }
    #[test]
    fn nested_output_scopes_restore_budget_and_do_not_cross_threads() {
        let outer = SolverOutputBudgetGuard::install(SolverOutputBudget::new(3).unwrap());
        assert!(validate_solver_output_dimensions(2, 2).is_err());
        std::thread::spawn(|| assert!(validate_solver_output_dimensions(2, 2).is_ok()))
            .join()
            .unwrap();
        {
            let _inner = SolverOutputBudgetGuard::install(SolverOutputBudget::new(6).unwrap());
            assert!(validate_solver_output_dimensions(2, 2).is_ok());
        }
        assert!(validate_solver_output_dimensions(2, 2).is_err());
        drop(outer);
        assert!(validate_solver_output_dimensions(2, 2).is_ok());
    }
}
