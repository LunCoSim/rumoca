//! Borrowed admission for numerical hosts without continuous event location.

use super::FmiRuntimeView;

/// Why a component needs event handling during continuous integration.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct FmiParameterEventError(&'static str);

impl FmiRuntimeView<'_> {
    /// Prove that every event partition is determined by parameters and constants.
    ///
    /// This is the same ME-EVENT-003 proof used by C codegen and linked event
    /// indicator construction. Predicates and their original messages remain
    /// executable at initialization and after parameter changes. Success does
    /// not authorize a host to skip those lifecycle operations; it proves only
    /// that continuous integration needs no event location or discrete updates.
    pub fn validate_parameter_only_events(&self) -> Result<(), FmiParameterEventError> {
        super::static_assertions::validate(self.model)
            .map(|_| ())
            .map_err(FmiParameterEventError)
    }
}
