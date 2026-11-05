//! Diagnostic for the complete fixed root-selection execution boundary.
use super::*;
#[test]
#[ignore = "manual native root-selection throughput measurement"]
fn fixed_root_selection_diagnostic() {
    const WIDTH: usize = 384;
    const ITERATIONS: usize = 512;
    let programs = (0..WIDTH)
        .map(|index| {
            vec![
                solve::LinearOp::LoadTime { dst: 0 },
                solve::LinearOp::Const {
                    dst: 1,
                    value: index as f64 * 0.125 + 1.0,
                },
                solve::LinearOp::Binary {
                    dst: 2,
                    op: solve::BinaryOp::Add,
                    lhs: 0,
                    rhs: 1,
                },
                solve::LinearOp::StoreOutput { src: 2 },
            ]
        })
        .collect();
    let mut model = solve::SolveModel::default();
    model.problem.events.root_conditions = solve::ScalarProgramBlock::with_source_span(
        programs,
        span().require_provenance("native root diagnostic").unwrap(),
    )
    .unwrap();
    model.problem.events.root_relation_memory_targets = vec![None; WIDTH];
    model.problem.events.root_zero_domains = vec![solve::RootZeroDomain::Previous; WIDTH];
    model.problem.continuous.refresh_owners =
        rumoca_eval_solve::refresh_plan::build_continuous_refresh_owners(&mut model.problem)
            .unwrap();
    let backend = crate::native_execution::backend(&model.pure_calls);
    let runtime = rumoca_solver::runtime::solve_runtime::SolveRuntime::new_with_execution_backend(
        &model,
        Some(backend),
    )
    .unwrap();
    for index in 0..16 {
        runtime
            .eval_root_conditions_from_solver_y(index as f64 * 0.01, &[], &[])
            .unwrap();
    }
    let start = std::time::Instant::now();
    for index in 0..ITERATIONS {
        let t = index as f64 * 0.01;
        let values = runtime
            .eval_root_conditions_from_solver_y(std::hint::black_box(t), &[], &[])
            .unwrap();
        assert_eq!(values.len(), WIDTH);
        for (index, value) in values.iter().enumerate() {
            assert_eq!(
                value.to_bits(),
                (t + (index as f64 * 0.125 + 1.0)).to_bits()
            );
        }
        std::hint::black_box(values);
    }
    println!(
        "root_selection: width={WIDTH} iterations={ITERATIONS} elapsed_ms={:.6}",
        start.elapsed().as_secs_f64() * 1000.0
    );
}
