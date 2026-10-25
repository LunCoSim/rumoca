use super::*;
use crate::runtime::solve_runtime::tests::{
    derive_test_structural_artifacts, set_test_implicit_jvp, spanned_block,
};
use std::cell::Cell;

fn rows() -> Vec<Vec<solve::LinearOp>> {
    use solve::BinaryOp::{Add, Sub};
    use solve::LinearOp::{Binary, LoadTime, LoadY, StoreOutput};
    vec![
        vec![
            LoadY { dst: 0, index: 0 },
            LoadY { dst: 1, index: 1 },
            Binary {
                dst: 2,
                op: Add,
                lhs: 0,
                rhs: 1,
            },
            LoadTime { dst: 3 },
            Binary {
                dst: 4,
                op: Sub,
                lhs: 2,
                rhs: 3,
            },
            StoreOutput { src: 4 },
        ],
        vec![
            LoadY { dst: 0, index: 0 },
            LoadY { dst: 1, index: 1 },
            Binary {
                dst: 2,
                op: Add,
                lhs: 0,
                rhs: 0,
            },
            Binary {
                dst: 3,
                op: Sub,
                lhs: 1,
                rhs: 2,
            },
            StoreOutput { src: 3 },
        ],
    ]
}

fn jvp() -> Vec<Vec<solve::LinearOp>> {
    use solve::BinaryOp::{Add, Sub};
    use solve::LinearOp::{Binary, LoadSeed, StoreOutput};
    vec![
        vec![
            LoadSeed { dst: 0, index: 0 },
            LoadSeed { dst: 1, index: 1 },
            Binary {
                dst: 2,
                op: Add,
                lhs: 0,
                rhs: 1,
            },
            StoreOutput { src: 2 },
        ],
        vec![
            LoadSeed { dst: 0, index: 0 },
            LoadSeed { dst: 1, index: 1 },
            Binary {
                dst: 2,
                op: Add,
                lhs: 0,
                rhs: 0,
            },
            Binary {
                dst: 3,
                op: Sub,
                lhs: 1,
                rhs: 2,
            },
            StoreOutput { src: 3 },
        ],
    ]
}

fn fixture() -> SolveRuntime {
    let tearing = solve::BlockTearing {
        tear_y_indices: vec![0],
        residual_rows: vec![0],
        causal_steps: vec![solve::CausalStep {
            row: 1,
            y_index: 1,
            coefficient: solve::CausalCoefficient::Unit,
        }],
    };
    let mut model = solve::SolveModel {
        problem: solve::SolveProblem {
            solve_layout: solve::SolveLayout {
                solver_maps: solve::SolverNameIndexMaps {
                    names: vec!["tear".into(), "causal".into()],
                    ..Default::default()
                },
                algebraic_scalar_count: 2,
                ..Default::default()
            },
            continuous: solve::ContinuousSolveSystem {
                implicit_rhs: solve::ComputeBlock::from_scalar_program_block(spanned_block(
                    rows(),
                    "torn_workspace.mo",
                )),
                implicit_row_targets: vec![
                    Some(solve::scalar_slot_y(0)),
                    Some(solve::scalar_slot_y(1)),
                ],
                algebraic_projection_plan: solve::AlgebraicProjectionPlan {
                    blocks: vec![solve::AlgebraicProjectionBlock {
                        rows: vec![0, 1],
                        y_indices: vec![0, 1],
                        tearing: Some(tearing),
                        alternate_charts: Vec::new(),
                    }],
                },
                ..Default::default()
            },
            ..Default::default()
        },
        initial_y: vec![1.0, 2.0],
        ..Default::default()
    };
    set_test_implicit_jvp(&mut model, jvp(), "torn_workspace_jvp.mo");
    derive_test_structural_artifacts(&mut model);
    SolveRuntime::new_fixture(&model).expect("torn fixture constructs")
}

fn projection(runtime: &SolveRuntime) -> RefreshProjectionModel<'_> {
    RefreshProjectionModel {
        runtime,
        seed_linearizations: None,
        plan: &runtime.algebraic_refresh.simultaneous_plan,
        block_indices: &runtime.algebraic_refresh.simultaneous_block_indices,
        plan_validated: true,
        jacobian_v: ProjectionJacobian::SolverY {
            block: &runtime.implicit_projection_jacobian_v,
            scalar: &runtime.implicit_projection_scalar_jacobian_v,
        },
    }
}

fn tearing(runtime: &SolveRuntime) -> &solve::BlockTearing {
    runtime
        .model
        .problem
        .continuous
        .algebraic_projection_plan
        .blocks[0]
        .tearing
        .as_ref()
        .expect("fixture has an issued tearing")
}

fn sweep(
    runtime: &SolveRuntime,
    y: &mut [f64],
    t: f64,
    residual: &mut Vec<f64>,
) -> Result<bool, RuntimeSolveError> {
    projection(runtime).torn_block_sweep(tearing(runtime), y, &[], t, residual)
}

fn buffers(workspace: &TornSweepWorkspace) -> Vec<(usize, usize)> {
    let values = vec![
        (workspace.raw.as_ptr() as usize, workspace.raw.capacity()),
        (
            workspace.dense_residuals.as_ptr() as usize,
            workspace.dense_residuals.capacity(),
        ),
    ];
    #[cfg(debug_assertions)]
    let values = {
        let mut values = values;
        values.extend([
            (
                workspace.reference_y.as_ptr() as usize,
                workspace.reference_y.capacity(),
            ),
            (
                workspace.reference_residuals.as_ptr() as usize,
                workspace.reference_residuals.capacity(),
            ),
        ]);
        values
    };
    values
}

#[test]
fn torn_workspace_reuses_storage_across_inputs_and_declines() {
    let runtime = fixture();
    let mut residual = Vec::new();
    let mut y = [1.0, 100.0];
    assert!(sweep(&runtime, &mut y, 3.5, &mut residual).unwrap());
    assert_eq!(y, [1.0, 2.0]);
    assert_eq!(residual, [-0.5]);
    let warm = buffers(&runtime.torn_sweep_cache.workspace.borrow());
    for input in [2.0, -3.0, 20.0] {
        y = [input, 100.0];
        assert!(sweep(&runtime, &mut y, input * 3.0 + 0.5, &mut residual).unwrap());
        assert_eq!(y, [input, input + input]);
        assert_eq!(residual, [-0.5]);
    }
    y = [f64::MAX, 100.0];
    assert!(!sweep(&runtime, &mut y, 0.0, &mut residual).unwrap());
    y = [4.0, 100.0];
    assert!(sweep(&runtime, &mut y, 11.0, &mut residual).unwrap());
    assert_eq!(y, [4.0, 8.0]);
    assert_eq!(residual, [1.0]);
    assert_eq!(buffers(&runtime.torn_sweep_cache.workspace.borrow()), warm);

    let other = runtime.clone();
    assert!(other.torn_sweep_cache.entries.borrow().is_empty());
    assert_eq!(other.torn_sweep_cache.workspace.borrow().raw.capacity(), 0);
    let guard = runtime.torn_sweep_cache.workspace.borrow_mut();
    let error = sweep(&runtime, &mut y, 0.0, &mut residual).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("torn sweep workspace is already active")
    );
    assert!(sweep(&other, &mut y, 11.0, &mut residual).unwrap());
    drop(guard);
}

struct NoopSchedule;

impl CompiledSolveAssignmentSchedule for NoopSchedule {
    fn call(
        &self,
        _: &mut [f64],
        _: &[f64],
        _: f64,
        _: &[rumoca_core::ExternalTableData],
    ) -> Result<(), String> {
        Ok(())
    }
}

struct ResidualBlock(Cell<bool>);

impl CompiledSolveExpression for ResidualBlock {
    fn call(
        &self,
        y: &[f64],
        _: &[f64],
        _: f64,
        _: &[rumoca_core::ExternalTableData],
        out: &mut [f64],
    ) -> Result<(), String> {
        assert!(
            out.iter().all(|value| value.to_bits() == 0),
            "every dense slot is reset"
        );
        out[0] = y[0];
        out[1] = -0.0;
        if self.0.replace(false) {
            out[0] = f64::NAN;
            return Err("injected backend decline".into());
        }
        Ok(())
    }
}

#[test]
fn torn_workspace_resets_compiled_outputs_after_a_decline() {
    let runtime = fixture();
    let projection = projection(&runtime);
    let block = Rc::new(ResidualBlock(Cell::new(false)));
    let compiled = CompiledTornSweep {
        schedule: Rc::new(NoopSchedule),
        residual_block: block.clone(),
        residual_outputs: vec![Some(1), Some(0)].into_boxed_slice(),
        residual_len: 2,
    };
    let mut workspace = TornSweepWorkspace::default();
    let mut y = [1.0, 2.0];
    let plan = solve::BlockTearing::default();
    assert_eq!(
        projection
            .eval_compiled_torn_sweep(&compiled, &plan, &mut y, &[], 0.0, &mut workspace)
            .unwrap(),
        Some(TornSweepStatus::Completed)
    );
    let warm = buffers(&workspace);
    for (input, fail) in [(10.0, true), (-3.0, false), (20.0, false)] {
        y[0] = input;
        block.0.set(fail);
        workspace.raw.clear();
        let status = projection
            .eval_compiled_torn_sweep(&compiled, &plan, &mut y, &[], 0.0, &mut workspace)
            .unwrap();
        if fail {
            assert_eq!(status, None);
            assert!(workspace.raw.is_empty());
        } else {
            assert_eq!(status, Some(TornSweepStatus::Completed));
            assert_eq!(workspace.raw[0].unwrap().to_bits(), (-0.0_f64).to_bits());
            assert_eq!(workspace.raw[1], Some(input));
        }
    }
    assert_eq!(buffers(&workspace), warm);
}
