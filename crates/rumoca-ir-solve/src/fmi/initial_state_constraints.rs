//! Checked instance constraints for continuous state coordinates left free by
//! the original initialization owner. Programs and source equations stay intact.

use std::collections::{BTreeMap, BTreeSet};

use super::{FmiComponent, FmiComponentError, FmiStorageColumn, FmiVariable};
use crate::{InitializationSolveSystem, ScalarSlot, SolveEventClass, SolveVariableStorageRole};

/// A finite accepted value for one exact checked state scalar identity.
/// This is an initialization constraint, not an initialization start guess.
#[derive(Clone, Debug, PartialEq)]
pub struct InitialStateConstraint {
    name: String,
    value: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum InitialStateConstraintError {
    #[error("an initial state constraint requires a non-empty scalar name")]
    EmptyName,
    #[error("initial state constraint `{name}` must be finite")]
    NonFiniteValue { name: String },
}

impl InitialStateConstraint {
    pub fn new(name: impl Into<String>, value: f64) -> Result<Self, InitialStateConstraintError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(InitialStateConstraintError::EmptyName);
        }
        if !value.is_finite() {
            return Err(InitialStateConstraintError::NonFiniteValue { name });
        }
        Ok(Self { name, value })
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub const fn value(&self) -> f64 {
        self.value
    }
}

/// Structural facts whose state cannot be transferred by continuous free-state
/// constraints. They are derived from the checked component, not source names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InitialStateConstraintFeatures {
    pub event_class: Option<SolveEventClass>,
    pub discrete_storage: bool,
    pub pre_history: bool,
    pub reduced_state_charts: bool,
    pub parameter_initialization_unknowns: bool,
}

impl InitialStateConstraintFeatures {
    #[must_use]
    pub fn is_continuous_state_only(&self) -> bool {
        self.event_class.is_none()
            && !self.discrete_storage
            && !self.pre_history
            && !self.reduced_state_charts
            && !self.parameter_initialization_unknowns
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FmiInitialStateConstraintError {
    #[error(
        "nonredundant free initial state constraints {required_changes:?} (name, requested, prepared) cannot transfer these component features: {features:?}"
    )]
    UnsupportedFeatures {
        features: InitialStateConstraintFeatures,
        required_changes: Vec<(String, f64, f64)>,
    },
    #[error("`{name}` is not a free initial state scalar of this checked component")]
    NotFreeState { name: String },
    #[error("initial state constraint `{name}` is repeated")]
    DuplicateState { name: String },
    #[error(
        "initial state constraint `{name}` value {value} is outside [{minimum:?}, {maximum:?}]"
    )]
    OutsideDomain {
        name: String,
        value: f64,
        minimum: Option<f64>,
        maximum: Option<f64>,
    },
    #[error("initial state constraint does not match checked storage: {0}")]
    Storage(String),
    #[error(transparent)]
    InvalidComponent(#[from] FmiComponentError),
}

impl FmiComponent {
    /// Exact scalar source identities of the checked state slots that have no
    /// given, projection, or update owner at initialization, in storage order.
    #[must_use]
    pub fn free_initial_state_names(&self) -> Vec<String> {
        let free = self.free_initial_state_slots();
        self.storage_variables()
            .filter(|variable| variable.role() == Some(SolveVariableStorageRole::State))
            .flat_map(|variable| {
                let base = variable
                    .storage()
                    .expect("storage variable has storage")
                    .base();
                variable
                    .scalar_names()
                    .iter()
                    .enumerate()
                    .filter(|(scalar, _)| free.contains(&(base + scalar)))
                    .map(|(_, name)| name.clone())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[must_use]
    pub fn initial_state_constraint_features(&self) -> InitialStateConstraintFeatures {
        let problem = self.problem();
        let layout = &problem.solve_layout;
        InitialStateConstraintFeatures {
            event_class: crate::solve_event_class(problem),
            discrete_storage: !layout.discrete_real_scalar_names.is_empty()
                || !layout.discrete_valued_scalar_names.is_empty(),
            pre_history: !layout.pre_param_bindings.is_empty(),
            reduced_state_charts: !problem.continuous.reduced_chart_set.is_empty(),
            parameter_initialization_unknowns: problem
                .initialization
                .projection_unknowns()
                .iter()
                .any(|slot| matches!(slot, ScalarSlot::P { .. })),
        }
    }

    /// Derive an instance component from an immutable prepared component.
    /// Only previously free state coordinates may acquire given ownership.
    /// Every original residual, update and numerical projection is retained;
    /// initialization must still certify them against the admitted values.
    ///
    /// Names, domains and duplicates are checked before normalization. A value
    /// bitwise equal to its checked `initial_y` coordinate requires no new owner.
    /// If every request is an identity, the returned clone shares the original
    /// component program and preserves its initialization and event ownership.
    /// Only nonredundant requests pass the continuous-state capability guard and
    /// acquire given ownership. Callers must certify every requested observation
    /// after initialization events, including normalized identities. This does
    /// not transfer hybrid, delay, discrete or parameter-solve state; the source
    /// component stays intact.
    pub fn with_free_initial_state_constraints(
        &self,
        constraints: &[InitialStateConstraint],
    ) -> Result<Self, FmiInitialStateConstraintError> {
        // Identity and domain admission precede normalization: an invalid name,
        // repeated request or out-of-domain value is never a harmless no-op.
        let mut accepted = self.resolve_initial_state_constraints(constraints)?;
        let mut required_changes = Vec::new();
        for (&index, &(value, input_index, scalar)) in &accepted {
            let prepared = self.model.initial_y.get(index).copied().ok_or_else(|| {
                FmiInitialStateConstraintError::Storage(format!(
                    "state index {index} has no prepared coordinate"
                ))
            })?;
            if value.to_bits() != prepared.to_bits() {
                required_changes.push((
                    self.construction_inputs[input_index].scalar_names[scalar].clone(),
                    value,
                    prepared,
                ));
            }
        }
        accepted.retain(|index, (value, _, _)| {
            value.to_bits() != self.model.initial_y[*index].to_bits()
        });
        if accepted.is_empty() {
            return Ok(self.clone());
        }
        let features = self.initial_state_constraint_features();
        if !features.is_continuous_state_only() {
            return Err(FmiInitialStateConstraintError::UnsupportedFeatures {
                features,
                required_changes,
            });
        }
        let mut model = self.model.as_ref().clone();
        let mut inputs = self.construction_inputs.clone();
        let mut initialization = std::mem::take(&mut model.problem.initialization).into_input();
        for (&index, &(value, input_index, scalar)) in &accepted {
            model.initial_y[index] = value;
            inputs[input_index].start[scalar] = value;
            initialization.given_state_indices.push(index);
        }
        initialization.given_state_indices.sort_unstable();
        model.problem.initialization = InitializationSolveSystem::construct(initialization)
            .map_err(|error| FmiInitialStateConstraintError::Storage(error.to_string()))?;
        Self::construct(model, inputs).map_err(Into::into)
    }

    fn free_initial_state_slots(&self) -> BTreeSet<usize> {
        self.problem()
            .initialization
            .free_state_indices(self.model.state_scalar_count())
            .into_iter()
            .collect()
    }

    fn resolve_initial_state_constraints(
        &self,
        constraints: &[InitialStateConstraint],
    ) -> Result<BTreeMap<usize, (f64, usize, usize)>, FmiInitialStateConstraintError> {
        let free = self.free_initial_state_slots();
        let mut accepted = BTreeMap::new();
        for constraint in constraints {
            let (variable, scalar) = self
                .storage_variables()
                .find_map(|variable| {
                    variable
                        .scalar_names()
                        .iter()
                        .position(|name| name == constraint.name())
                        .map(|scalar| (variable, scalar))
                })
                .ok_or_else(|| not_free(constraint))?;
            let storage = variable.storage().expect("storage variable has storage");
            let index = storage.base() + scalar;
            if storage.column() != FmiStorageColumn::Y
                || variable.role() != Some(SolveVariableStorageRole::State)
                || !free.contains(&index)
            {
                return Err(not_free(constraint));
            }
            validate_domain(variable, scalar, constraint)?;
            let input_index = self
                .construction_inputs
                .iter()
                .position(|input| {
                    input
                        .scalar_names
                        .get(scalar)
                        .is_some_and(|name| name == constraint.name())
                })
                .ok_or_else(|| FmiInitialStateConstraintError::Storage(constraint.name().into()))?;
            if accepted
                .insert(index, (constraint.value(), input_index, scalar))
                .is_some()
            {
                return Err(FmiInitialStateConstraintError::DuplicateState {
                    name: constraint.name().into(),
                });
            }
        }
        Ok(accepted)
    }
}

fn not_free(constraint: &InitialStateConstraint) -> FmiInitialStateConstraintError {
    FmiInitialStateConstraintError::NotFreeState {
        name: constraint.name().into(),
    }
}

fn validate_domain(
    variable: &FmiVariable,
    scalar: usize,
    constraint: &InitialStateConstraint,
) -> Result<(), FmiInitialStateConstraintError> {
    let bound = |values: Option<&[f64]>| -> Result<Option<f64>, FmiInitialStateConstraintError> {
        values
            .map(|values| {
                values.get(scalar).copied().ok_or_else(|| {
                    FmiInitialStateConstraintError::Storage(constraint.name().into())
                })
            })
            .transpose()
    };
    let minimum = bound(variable.minimum())?;
    let maximum = bound(variable.maximum())?;
    if minimum.is_some_and(|minimum| constraint.value() < minimum)
        || maximum.is_some_and(|maximum| constraint.value() > maximum)
    {
        return Err(FmiInitialStateConstraintError::OutsideDomain {
            name: constraint.name().into(),
            value: constraint.value(),
            minimum,
            maximum,
        });
    }
    Ok(())
}
