//! Fixed root selections retain program identity and native failure boundaries.

use super::*;
use std::cell::Cell;

struct Expression {
    block: PreparedScalarProgramBlock,
    calls: Rc<Cell<usize>>,
    fail: Rc<Cell<bool>>,
}
impl CompiledSolveExpression for Expression {
    fn call(
        &self,
        y: &[f64],
        p: &[f64],
        t: f64,
        _tables: &[rumoca_core::ExternalTableData],
        out: &mut [f64],
    ) -> Result<(), String> {
        self.calls.set(self.calls.get() + 1);
        if self.fail.get() {
            return Err("admitted root batch failed".into());
        }
        self.block
            .eval_with_context(y, p, t, RowEvalContext::default(), out)
            .map_err(|error| error.to_string())
    }
}
#[derive(Default)]
struct Backend {
    calls: Rc<Cell<usize>>,
    fail: Rc<Cell<bool>>,
}
impl SolveExecutionBackend for Backend {
    fn compile_expression(
        &self,
        block: &solve::ScalarProgramBlock,
    ) -> Result<Rc<dyn CompiledSolveExpression>, String> {
        Ok(Rc::new(Expression {
            block: PreparedScalarProgramBlock::new(block.clone()).unwrap(),
            calls: self.calls.clone(),
            fail: self.fail.clone(),
        }))
    }
    fn compile_jacobian_expression(
        &self,
        _: &solve::ScalarProgramBlock,
    ) -> Result<Rc<dyn CompiledSolveJacobianExpression>, String> {
        Err("root fixture has no Jacobian backend".into())
    }
    fn compile_assignment_schedule(
        &self,
        _: &solve::SharedAssignmentSchedule,
    ) -> Result<Rc<dyn CompiledSolveAssignmentSchedule>, String> {
        Err("root fixture has no assignment backend".into())
    }
    fn compile_event_transaction(
        &self,
        _: &solve::EventTransactionProgram,
    ) -> Result<Rc<dyn CompiledSolveEventTransaction>, String> {
        Err("root fixture has no event backend".into())
    }
}
fn fixture() -> (SolveRuntime, Rc<Backend>) {
    let mut model = solve::SolveModel::default();
    model.problem.events.root_conditions =
        spanned_block(vec![time_plus_one_root_row(); 64], "root_batches.mo");
    model.problem.events.root_relation_memory_targets = vec![None; 64];
    model.problem.events.root_zero_domains = vec![solve::RootZeroDomain::Previous; 64];
    model.problem.continuous.refresh_owners =
        solve_eval::refresh_plan::build_continuous_refresh_owners(&mut model.problem).unwrap();
    let backend = Rc::new(Backend::default());
    let runtime = SolveRuntime::new_with_execution_backend(&model, Some(backend.clone())).unwrap();
    assert_eq!(
        runtime
            .root_condition_plan
            .as_ref()
            .unwrap()
            .search_rows
            .len(),
        64
    );
    (runtime, backend)
}
#[test]
fn fixed_root_selection_uses_one_native_batch() {
    let (runtime, backend) = fixture();
    for t in [-0.0, 0.25, 2.5] {
        backend.calls.set(0);
        let values = runtime
            .eval_root_conditions_from_solver_y(t, &[], &[])
            .unwrap();
        assert!(
            values
                .iter()
                .all(|value| value.to_bits() == (t + 1.0_f64).to_bits())
        );
        assert_eq!(
            backend.calls.get(),
            1,
            "one fixed root selection must cross the native boundary once"
        );
    }
}
#[test]
fn admitted_root_execution_failure_is_terminal() {
    let (runtime, backend) = fixture();
    backend.fail.set(true);
    backend.calls.set(0);
    let error = runtime
        .eval_root_conditions_from_solver_y(0.0, &[], &[])
        .expect_err("native execution failure cannot be retried as interpretation");
    assert!(error.to_string().contains("admitted root batch failed"));
    assert_eq!(backend.calls.get(), 1);
}
