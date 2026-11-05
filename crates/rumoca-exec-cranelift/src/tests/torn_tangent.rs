//! Native tangent batches retain the reference sweep's exact scalar results.

use super::*;
use rumoca_eval_solve::{
    EvalSolveError, TangentPoint, TornTangentEvaluator, TornTangentPreparation,
};
use rumoca_ir_solve::{BinaryOp, BlockTearing, CausalCoefficient, CausalStep, TornTangentPlan};
use std::cell::Cell;

fn source() -> (ScalarProgramBlock, BlockTearing) {
    let rows = vec![
        vec![
            LinearOp::Const { dst: 0, value: 2.0 },
            LinearOp::LoadSeed { dst: 1, index: 0 },
            LinearOp::LoadSeed { dst: 2, index: 1 },
            LinearOp::Binary {
                dst: 3,
                op: BinaryOp::Mul,
                lhs: 0,
                rhs: 1,
            },
            LinearOp::Binary {
                dst: 4,
                op: BinaryOp::Sub,
                lhs: 2,
                rhs: 3,
            },
            LinearOp::StoreOutput { src: 4 },
        ],
        vec![
            LinearOp::LoadY { dst: 0, index: 0 },
            LinearOp::LoadY { dst: 1, index: 1 },
            LinearOp::LoadSeed { dst: 2, index: 0 },
            LinearOp::LoadSeed { dst: 3, index: 1 },
            LinearOp::LoadSeed { dst: 4, index: 2 },
            LinearOp::Binary {
                dst: 5,
                op: BinaryOp::Mul,
                lhs: 2,
                rhs: 1,
            },
            LinearOp::Binary {
                dst: 6,
                op: BinaryOp::Mul,
                lhs: 0,
                rhs: 3,
            },
            LinearOp::Binary {
                dst: 7,
                op: BinaryOp::Add,
                lhs: 5,
                rhs: 6,
            },
            LinearOp::Binary {
                dst: 8,
                op: BinaryOp::Sub,
                lhs: 7,
                rhs: 4,
            },
            LinearOp::StoreOutput { src: 8 },
        ],
        vec![
            LinearOp::LoadSeed { dst: 0, index: 0 },
            LinearOp::LoadSeed { dst: 1, index: 2 },
            LinearOp::Binary {
                dst: 2,
                op: BinaryOp::Add,
                lhs: 0,
                rhs: 1,
            },
            LinearOp::StoreOutput { src: 2 },
        ],
    ];
    let source = ScalarProgramBlock::with_program_spans(rows, vec![fixture_span(); 3]).unwrap();
    let tearing = BlockTearing {
        tear_y_indices: vec![0, 2],
        residual_rows: vec![1, 2],
        causal_steps: vec![CausalStep {
            row: 0,
            y_index: 1,
            coefficient: CausalCoefficient::Unit,
        }],
    };
    (source, tearing)
}

fn native(plan: &TornTangentPlan, source: &ScalarProgramBlock) -> CompiledJacobianV {
    if plan.directional() {
        return compile_jacobian_scalar_program_block(source).unwrap();
    }
    let block = ScalarProgramBlock::with_tangent_lane_programs(
        plan.programs(),
        vec![fixture_span(); plan.programs().len()],
    )
    .unwrap();
    compile_jacobian_scalar_program_block(&block).unwrap()
}

fn bits(jacobian: &rumoca_eval_solve::TornTangentJacobian) -> Vec<u64> {
    jacobian
        .residual
        .iter()
        .chain(&jacobian.recovered)
        .map(|value| value.to_bits())
        .collect()
}

fn call_native_tangent(
    compiled: &CompiledJacobianV,
    point: TangentPoint<'_>,
    program: usize,
    seed: &[f64],
    out: &mut Vec<f64>,
) -> Result<bool, EvalSolveError> {
    compiled
        .call_program_outputs(
            program,
            rumoca_eval_solve::JacobianEvalInputs {
                y: point.y,
                p: point.p,
                t: point.t,
                seed,
            },
            &[],
            out,
        )
        .map(|()| true)
        .map_err(|error| EvalSolveError::InvalidRow {
            message: error.to_string(),
            span: Some(fixture_span()),
        })
}

fn check_native_tangent_failure_recovery(
    evaluator: &TornTangentEvaluator,
    compiled: &CompiledJacobianV,
    point: TangentPoint<'_>,
    coefficient: usize,
    stride: usize,
) -> rumoca_eval_solve::TornTangentJacobian {
    let calls = Cell::new(0);
    let failure = evaluator.eval_through(point, &mut |program, seed, out| {
        calls.set(calls.get() + 1);
        if calls.get() == 2 {
            return Err(EvalSolveError::InvalidRow {
                message: "injected tangent execution error".into(),
                span: Some(fixture_span()),
            });
        }
        call_native_tangent(compiled, point, program, seed, out)
    });
    assert!(
        failure
            .unwrap_err()
            .to_string()
            .contains("injected tangent execution error")
    );
    assert!(
        evaluator
            .eval_through(point, &mut |program, seed, out| {
                call_native_tangent(compiled, point, program, seed, out)?;
                if program == 0 {
                    out[coefficient] = f64::NAN;
                }
                Ok(true)
            })
            .unwrap()
            .is_none()
    );
    evaluator
        .eval_through(point, &mut |program, seed, out| {
            assert_eq!(seed.len(), point.y.len() * stride);
            assert!(
                seed[3 * stride..]
                    .iter()
                    .all(|value| value.to_bits() == 0.0_f64.to_bits())
            );
            call_native_tangent(compiled, point, program, seed, out)
        })
        .unwrap()
        .unwrap()
}

#[test]
fn native_torn_workspace_recovers_after_errors_and_nonfinite_coefficients() {
    let (source, tearing) = source();
    for plan in [
        TornTangentPlan::derive(&tearing, &source).unwrap(),
        TornTangentPlan::derive_directional(&tearing, &source).unwrap(),
    ] {
        let compiled = native(&plan, &source);
        let coefficient = if plan.directional() {
            0
        } else {
            plan.lanes() - 1
        };
        let stride = if plan.directional() { 1 } else { plan.lanes() };
        let evaluator = TornTangentEvaluator::new(plan.clone(), &source).unwrap();
        for width in [3, 3035, 5, 4096, 3] {
            let mut y = vec![-3.0; width];
            y[..3].copy_from_slice(&[width as f64 * 1.25, -0.0, 3.125]);
            let point = TangentPoint {
                y: &y,
                p: &[],
                t: 0.0,
                context: Default::default(),
            };
            let expected = TornTangentEvaluator::new(plan.clone(), &source)
                .unwrap()
                .eval(point)
                .unwrap()
                .unwrap();
            let actual = check_native_tangent_failure_recovery(
                &evaluator,
                &compiled,
                point,
                coefficient,
                stride,
            );
            assert_eq!(bits(&actual), bits(&expected));
        }
    }
}

#[test]
fn native_torn_workspace_allows_reentry_and_rejects_short_coordinates() {
    let (source, tearing) = source();
    for plan in [
        TornTangentPlan::derive(&tearing, &source).unwrap(),
        TornTangentPlan::derive_directional(&tearing, &source).unwrap(),
    ] {
        let compiled = native(&plan, &source);
        let evaluator = TornTangentEvaluator::new(plan, &source).unwrap();
        let y = [1.25, 2.5, 3.125];
        let point = TangentPoint {
            y: &y,
            p: &[],
            t: 0.0,
            context: Default::default(),
        };
        let expected = bits(&evaluator.eval(point).unwrap().unwrap());
        let nested = Cell::new(false);
        let actual = evaluator
            .eval_through(point, &mut |program, seed, out| {
                if !nested.replace(true) {
                    assert_eq!(bits(&evaluator.eval(point).unwrap().unwrap()), expected);
                }
                call_native_tangent(&compiled, point, program, seed, out)
            })
            .unwrap()
            .unwrap();
        assert_eq!(bits(&actual), expected);
        let invalid = TangentPoint { y: &[], ..point };
        let error = evaluator.eval_through(invalid, &mut |_, _, _| {
            panic!("invalid coordinates must fail before calling a program")
        });
        assert!(matches!(
            error,
            Err(EvalSolveError::MissingInput {
                vector: "y",
                len: 0,
                ..
            })
        ));
        assert_eq!(bits(&evaluator.eval(point).unwrap().unwrap()), expected);
    }
}

#[test]
fn native_torn_preparation_shares_clean_workspace_between_lane_shapes() {
    let (source, tearing) = source();
    let mut preparation = TornTangentPreparation::new(&source);
    let lane_plan = TornTangentPlan::derive(&tearing, &source).unwrap();
    let direction_plan = TornTangentPlan::derive_directional(&tearing, &source).unwrap();
    let native_lanes = native(&lane_plan, &source);
    let native_directions = native(&direction_plan, &source);
    let lanes = preparation.prepare(lane_plan.clone()).unwrap();
    let directions = preparation.prepare(direction_plan).unwrap();
    let mut y = vec![-3.0; 64];
    y[..3].copy_from_slice(&[1.25, 2.5, 3.125]);
    let point = TangentPoint {
        y: &y,
        p: &[],
        t: 0.0,
        context: Default::default(),
    };
    let expected = bits(&lanes.eval(point).unwrap().unwrap());
    for (evaluator, compiled, stride) in [
        (&lanes, &native_lanes, lane_plan.lanes()),
        (&directions, &native_directions, 1),
        (&lanes, &native_lanes, lane_plan.lanes()),
        (&directions, &native_directions, 1),
    ] {
        let actual = evaluator
            .eval_through(point, &mut |program, seed, out| {
                assert_eq!(seed.len(), point.y.len() * stride);
                assert!(
                    seed[3 * stride..]
                        .iter()
                        .all(|value| value.to_bits() == 0.0_f64.to_bits())
                );
                call_native_tangent(compiled, point, program, seed, out)
            })
            .unwrap()
            .unwrap();
        assert_eq!(bits(&actual), expected);
    }
    let nested = Cell::new(false);
    let actual = lanes
        .eval_through(point, &mut |program, seed, out| {
            if !nested.replace(true) {
                assert_eq!(bits(&directions.eval(point).unwrap().unwrap()), expected);
            }
            call_native_tangent(&native_lanes, point, program, seed, out)
        })
        .unwrap()
        .unwrap();
    assert_eq!(bits(&actual), expected);
}

#[test]
#[ignore = "diagnostic: compare independent optimized evaluator executables"]
fn native_torn_tangent_workspace_diagnostic() {
    let (source, tearing) = source();
    for directional in [false, true] {
        let plan = if directional {
            TornTangentPlan::derive_directional(&tearing, &source).unwrap()
        } else {
            TornTangentPlan::derive(&tearing, &source).unwrap()
        };
        let compiled = native(&plan, &source);
        let evaluator = TornTangentEvaluator::new(plan, &source).unwrap();
        let mut y = vec![0.0; 3035];
        y[..3].copy_from_slice(&[1.25, 2.5, 3.125]);
        let point = TangentPoint {
            y: &y,
            p: &[],
            t: 0.0,
            context: Default::default(),
        };
        let expected = bits(&evaluator.eval(point).unwrap().unwrap());
        let mut call = |program, seed: &[f64], out: &mut Vec<f64>| {
            compiled
                .call_program_outputs(
                    program,
                    rumoca_eval_solve::JacobianEvalInputs {
                        y: &y,
                        p: &[],
                        t: 0.0,
                        seed,
                    },
                    &[],
                    out,
                )
                .map(|()| true)
                .map_err(|error| EvalSolveError::InvalidRow {
                    message: error.to_string(),
                    span: Some(fixture_span()),
                })
        };
        assert_eq!(
            bits(&evaluator.eval_through(point, &mut call).unwrap().unwrap()),
            expected
        );
        let began = std::time::Instant::now();
        let mut last = None;
        for _ in 0..8192 {
            last = std::hint::black_box(evaluator.eval_through(point, &mut call).unwrap());
        }
        let elapsed = began.elapsed().as_nanos();
        assert_eq!(bits(&last.unwrap()), expected);
        println!(
            "torn_workspace directional={directional} reads=8192 coordinates=3035 elapsed_ns={elapsed}"
        );
    }
}

#[test]
fn native_torn_tangent_batches_preserve_bits_and_reduce_invocations() {
    let (source, tearing) = source();
    let direction_plan = TornTangentPlan::derive_directional(&tearing, &source).unwrap();
    let lane_plan = TornTangentPlan::derive(&tearing, &source).unwrap();
    assert!(!lane_plan.directional());
    let directions = TornTangentEvaluator::new(direction_plan.clone(), &source).unwrap();
    let lanes = TornTangentEvaluator::new(lane_plan.clone(), &source).unwrap();
    let native_directions = native(&direction_plan, &source);
    let native_lanes = native(&lane_plan, &source);
    for y in [
        [1.25, 2.5, 3.125],
        [-3.2, -6.4, 20.48],
        [0.001, 0.002, 0.000002],
    ] {
        let point = TangentPoint {
            y: &y,
            p: &[],
            t: 0.0,
            context: Default::default(),
        };
        let expected = directions.eval(point).unwrap().unwrap();
        assert_eq!(bits(&lanes.eval(point).unwrap().unwrap()), bits(&expected));
        for (evaluator, native, expected_calls) in [
            (&directions, &native_directions, 7),
            (&lanes, &native_lanes, 3),
        ] {
            let calls = Cell::new(0);
            let mut call = |program, seed: &[f64], out: &mut Vec<f64>| {
                calls.set(calls.get() + 1);
                native
                    .call_program_outputs(
                        program,
                        rumoca_eval_solve::JacobianEvalInputs {
                            y: &y,
                            p: &[],
                            t: 0.0,
                            seed,
                        },
                        &[],
                        out,
                    )
                    .map(|()| true)
                    .map_err(|error| EvalSolveError::InvalidRow {
                        message: error.to_string(),
                        span: Some(fixture_span()),
                    })
            };
            let actual = evaluator.eval_through(point, &mut call).unwrap().unwrap();
            assert_eq!(bits(&actual), bits(&expected));
            assert_eq!(calls.get(), expected_calls);
        }
    }
}

#[test]
fn native_torn_tangent_execution_errors_do_not_become_reference_execution() {
    let (source, tearing) = source();
    for plan in [
        TornTangentPlan::derive(&tearing, &source).unwrap(),
        TornTangentPlan::derive_directional(&tearing, &source).unwrap(),
    ] {
        let native = native(&plan, &source);
        let evaluator = TornTangentEvaluator::new(plan, &source).unwrap();
        let y = [1.25, 2.5, 3.125];
        let point = TangentPoint {
            y: &y,
            p: &[],
            t: 0.0,
            context: Default::default(),
        };
        let result = evaluator.eval_through(point, &mut |program, _, out| {
            native
                .call_program_outputs(
                    program,
                    rumoca_eval_solve::JacobianEvalInputs {
                        y: &y,
                        p: &[],
                        t: 0.0,
                        seed: &[],
                    },
                    &[],
                    out,
                )
                .map(|()| true)
                .map_err(|error| EvalSolveError::InvalidRow {
                    message: error.to_string(),
                    span: Some(fixture_span()),
                })
        });
        assert!(
            result.is_err(),
            "invalid native seed extent must remain a terminal error"
        );
    }
}
