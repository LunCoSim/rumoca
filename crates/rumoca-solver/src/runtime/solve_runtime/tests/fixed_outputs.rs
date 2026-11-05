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
            return Err("admitted fixed selection failed".into());
        }
        self.block
            .eval_with_context(y, p, t, RowEvalContext::default(), out)
            .map_err(|error| error.to_string())
    }
}
#[derive(Default)]
struct Backend {
    decline: Rc<Cell<bool>>,
    calls: Rc<Cell<usize>>,
    fail: Rc<Cell<bool>>,
}
impl SolveExecutionBackend for Backend {
    fn compile_expression(
        &self,
        block: &solve::ScalarProgramBlock,
    ) -> Result<Rc<dyn CompiledSolveExpression>, String> {
        if self.decline.get() {
            return Err("native compilation declined".into());
        }
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
        runtime.root_condition_plan.as_ref().unwrap().entries.len(),
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
    assert!(
        error
            .to_string()
            .contains("admitted fixed selection failed")
    );
    assert_eq!(backend.calls.get(), 1);
}

fn visible_model(grouped: bool) -> solve::SolveModel {
    let rows = (0..64)
        .map(|index| {
            let mut row = vec![
                solve::LinearOp::LoadTime { dst: 0 },
                solve::LinearOp::Const {
                    dst: 1,
                    value: index as f64 + 1.0,
                },
                solve::LinearOp::Binary {
                    dst: 2,
                    op: solve::BinaryOp::Add,
                    lhs: 0,
                    rhs: 1,
                },
                solve::LinearOp::StoreOutput { src: 2 },
            ];
            if grouped {
                row.extend([
                    solve::LinearOp::Const {
                        dst: 3,
                        value: -0.0,
                    },
                    solve::LinearOp::StoreOutput { src: 3 },
                ]);
            }
            row
        })
        .collect();
    let mut model = solve::SolveModel {
        visible_names: (0..if grouped { 128 } else { 64 })
            .map(|index| format!("v{index}"))
            .collect(),
        visible_value_rows: spanned_block(rows, "visible_batches.mo"),
        ..Default::default()
    };
    model.problem.continuous.refresh_owners =
        solve_eval::refresh_plan::build_continuous_refresh_owners(&mut model.problem).unwrap();
    model
}

fn visible_fixture() -> (SolveRuntime, Rc<Backend>) {
    let backend = Rc::new(Backend::default());
    let runtime =
        SolveRuntime::new_with_execution_backend(&visible_model(false), Some(backend.clone()))
            .unwrap();
    (runtime, backend)
}

#[test]
fn fixed_visible_selection_uses_one_native_batch() {
    let (runtime, backend) = visible_fixture();
    for t in [-0.0, 0.25, 2.5] {
        backend.calls.set(0);
        let values = runtime.visible_values(&[], &[], t).unwrap();
        for (index, value) in values.iter().enumerate() {
            assert_eq!(value.to_bits(), (t + (index as f64 + 1.0)).to_bits());
        }
        assert_eq!(
            backend.calls.get(),
            1,
            "one fixed visible selection must cross the native boundary once"
        );
    }
}

#[test]
fn admitted_visible_execution_failure_is_terminal() {
    let (runtime, backend) = visible_fixture();
    backend.fail.set(true);
    backend.calls.set(0);
    let error = runtime
        .visible_values(&[], &[], 0.0)
        .expect_err("native execution failure cannot be retried as interpretation");
    assert!(
        error
            .to_string()
            .contains("admitted fixed selection failed")
    );
    assert_eq!(backend.calls.get(), 1);
}

#[test]
fn grouped_visible_outputs_share_native_batch_and_preserve_bits() {
    let backend = Rc::new(Backend::default());
    let runtime =
        SolveRuntime::new_with_execution_backend(&visible_model(true), Some(backend.clone()))
            .unwrap();
    backend.calls.set(0);
    let values = runtime.visible_values(&[], &[], 0.25).unwrap();
    assert_eq!(values.len(), 128);
    for (index, pair) in values.chunks_exact(2).enumerate() {
        assert_eq!(pair[0].to_bits(), (0.25 + (index as f64 + 1.0)).to_bits());
        assert_eq!(pair[1].to_bits(), (-0.0_f64).to_bits());
    }
    assert_eq!(backend.calls.get(), 1);
}

#[test]
fn alternate_visible_batch_executes_only_current_programs() {
    let mut model = visible_model(false);
    let mut rows = model.visible_value_rows.programs().to_vec();
    rows[0][0] = solve::LinearOp::LoadY { dst: 0, index: 17 };
    model.visible_value_rows = spanned_block(rows, "visible_batches.mo");
    let backend = Rc::new(Backend::default());
    let primary = SolveRuntime::new_with_execution_backend(&model, Some(backend.clone())).unwrap();
    let alternate = primary.new_alternate(&visible_model(false)).unwrap();
    backend.calls.set(0);
    let values = alternate.visible_values(&[], &[], 0.25).unwrap();
    assert_eq!(values[0].to_bits(), 1.25_f64.to_bits());
    assert_eq!(backend.calls.get(), 1);
}

#[test]
fn declined_visible_compilation_uses_the_same_checked_selection() {
    let backend = Rc::new(Backend::default());
    backend.decline.set(true);
    let runtime =
        SolveRuntime::new_with_execution_backend(&visible_model(false), Some(backend.clone()))
            .unwrap();
    let values = runtime.visible_values(&[], &[], 0.25).unwrap();
    assert_eq!(values.len(), 64);
    for (index, value) in values.iter().enumerate() {
        assert_eq!(value.to_bits(), (0.25 + (index as f64 + 1.0)).to_bits());
    }
    assert_eq!(backend.calls.get(), 0);
}
