//! Tangent evaluators the runtime prepares once per projection plan.

use super::*;

pub(super) struct PreparedTornTangent {
    pub(super) evaluator: rumoca_eval_solve::TornTangentEvaluator,
    pub(super) compiled: Option<Rc<dyn CompiledSolveJacobianExpression>>,
}

/// The tangent evaluator of each projection block's tearing over the
/// solver-Y JVP rows `jvp`, aligned with `plan.blocks`.
pub(super) fn torn_tangent_evaluators(
    plan: &solve::AlgebraicProjectionPlan,
    jvp: &solve::ScalarProgramBlock,
    backend: Option<&Rc<dyn SolveExecutionBackend>>,
    directional: Option<&Rc<dyn CompiledSolveJacobianExpression>>,
) -> Result<Rc<[Option<PreparedTornTangent>]>, EvalSolveError> {
    // One-direction plans all read the same JVP rows; prepare them once.
    let mut shared = None;
    plan.blocks
        .iter()
        .map(|block| {
            let Some(tearing) = block.tearing.as_ref() else {
                return Ok(None);
            };
            let Ok(mut plan) = solve::TornTangentPlan::derive(tearing, jvp) else {
                return Ok(None);
            };
            let mut compiled = None;
            if directional.is_some() {
                if !plan.directional() {
                    compiled = compiled_tangent_lanes(&plan, jvp, backend)?;
                }
                if compiled.is_none() {
                    let Ok(directions) = solve::TornTangentPlan::derive_directional(tearing, jvp)
                    else {
                        return Ok(None);
                    };
                    plan = directions;
                    compiled = directional.cloned();
                }
            }
            let evaluator = rumoca_eval_solve::TornTangentEvaluator::sharing_directions(
                plan,
                jvp,
                &mut shared,
            )?;
            Ok(Some(PreparedTornTangent {
                evaluator,
                compiled,
            }))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Rc::from)
}

/// Compile the already checked lane programs through the backend's existing
/// Jacobian entry point, retaining the original program provenance.
fn compiled_tangent_lanes(
    plan: &solve::TornTangentPlan,
    jvp: &solve::ScalarProgramBlock,
    backend: Option<&Rc<dyn SolveExecutionBackend>>,
) -> Result<Option<Rc<dyn CompiledSolveJacobianExpression>>, EvalSolveError> {
    let mut spans = vec![None; plan.programs().len()];
    let sources = plan
        .steps()
        .iter()
        .map(|step| (step.source.program, step.row))
        .chain(
            plan.residuals()
                .iter()
                .map(|row| (row.source.program, row.row)),
        );
    for (program, row) in sources {
        spans[program] = jvp.span_for_output(row);
    }
    let spans = spans
        .into_iter()
        .map(|span| {
            span.ok_or_else(|| EvalSolveError::InvalidRow {
                message: "torn tangent program has no canonical source span".into(),
                span: None,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let block = solve::ScalarProgramBlock::with_tangent_lane_programs(plan.programs(), spans)?;
    let compiled = compiled_jacobian(backend, "runtime_torn_tangent_lanes", &block);
    tracing::debug!(
        target: "rumoca_solver::native_execution",
        lanes = plan.lanes(), programs = plan.programs().len(),
        tensor_lanes = block.max_tensor_lanes(), native = compiled.is_some(),
        "prepared torn tangent lanes"
    );
    Ok(compiled)
}

/// The colored tangent evaluator of each projection block's issued Jacobian
/// application, aligned with `plan.blocks` and their structures.
pub(super) fn colored_tangent_evaluators(
    plan: &solve::AlgebraicProjectionPlan,
    structures: &solve::ContinuousStructuralArtifacts,
) -> Rc<[Option<rumoca_eval_solve::ColoredTangentEvaluator>]> {
    plan.blocks
        .iter()
        .zip(structures.algebraic_projection())
        .map(|(block, structure)| {
            if !rumoca_eval_solve::projection_policy::jacobian_sources().colored_lanes {
                return None;
            }
            let application = structure.jacobian_application().filter(|application| {
                application.rows() == block.rows && application.y_indices() == block.y_indices
            })?;
            let plan = solve::ColoredTangentPlan::derive(application).ok()?;
            Some(rumoca_eval_solve::ColoredTangentEvaluator::new(plan))
        })
        .collect()
}

#[cfg(test)]
mod tests;
