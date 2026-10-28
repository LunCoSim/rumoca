//! Replay DAE-C21 ordered algorithm owners through structural reconstruction.
use super::{RebuiltOwnerIdentities, TargetVariable};
use rumoca_ir_dae as dae;

pub(super) fn rebuild<'target>(
    source: dae::DaeView<'_>,
    target: &mut dae::DaeConstruction<'target>,
    expressions: &[dae::ExprId<'target>],
    identities: &RebuiltOwnerIdentities<'_, 'target>,
) -> Result<(), dae::DaeConstructionError> {
    target.model_events(|events| {
        for (_, transaction) in source.model_event_transactions() {
            let targets = transaction
                .targets()
                .map(|source| rebuilt_target(source, identities));
            let steps = transaction
                .steps()
                .map(|step| rebuilt_step(step, expressions, identities));
            events.transaction(targets, steps, transaction.provenance())?;
        }
        Ok(())
    })
}

fn rebuilt_step<'target>(
    step: dae::ModelEventStepView<'_>,
    expressions: &[dae::ExprId<'target>],
    identities: &RebuiltOwnerIdentities<'_, 'target>,
) -> dae::ModelEventStep<'target> {
    let definitions = step.definitions().map(|definition| {
        dae::ModelEventDefinition::new(
            rebuilt_target(definition.target(), identities),
            expressions[definition.value().index() as usize],
            definition.provenance(),
        )
    });
    dae::ModelEventStep::new(
        identities.conditions[step.trigger().index() as usize],
        identities.conditions[step.guard().index() as usize],
        step.clock()
            .map(|clock| identities.clocks[clock.index() as usize].clock_id()),
        definitions,
        step.provenance(),
    )
}

fn rebuilt_target<'target>(
    source: dae::ModelEventTarget<'_>,
    identities: &RebuiltOwnerIdentities<'_, 'target>,
) -> dae::ModelEventTarget<'target> {
    match identities.variables[source.variable() as usize].identity {
        TargetVariable::DiscreteReal(id) => dae::ModelEventTarget::DiscreteReal(id),
        TargetVariable::DiscreteValue(id) => dae::ModelEventTarget::DiscreteValue(id),
        _ => unreachable!("checked event transaction target retains its discrete role"),
    }
}
