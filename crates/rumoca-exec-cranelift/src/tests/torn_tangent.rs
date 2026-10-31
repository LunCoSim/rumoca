//! Native tangent batches retain the reference sweep's exact scalar results.

use super::*;
use rumoca_eval_solve::{EvalSolveError, TangentPoint, TornTangentEvaluator};
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
