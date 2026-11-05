//! Complete public observations and input overlays at the common ME owner.

use indexmap::IndexMap;
use rumoca_core::VarName;

use super::{MeSessionError, MeSimulationSession};

impl MeSimulationSession<'_, '_> {
    /// Batched `fmi3GetFloat64` over every output at the current coordinate.
    pub fn output_values(&self) -> Result<Vec<f64>, MeSessionError> {
        self.require_live()?;
        Ok(self.host.observe_current()?)
    }

    /// Visit one complete checked observation in declared output order, then
    /// host-applied inputs absent from those outputs in input admission order.
    /// Component evaluation, cardinality and overlay positions are checked before visiting;
    /// Labels share the canonical interned variable-name payload; cloning a
    /// label retains that payload without copying its text.
    pub fn visit_visible_values(
        &self,
        mut visit: impl FnMut(&VarName, f64),
    ) -> Result<(), MeSessionError> {
        let mut values = self.output_values()?;
        let names = &self.host.output_labels;
        if names.len() != values.len() {
            return Err(MeSessionError::Contract {
                reason: format!(
                    "the component returned {} output values for {} declared outputs",
                    values.len(),
                    names.len()
                ),
            });
        }
        for (name, input) in &self.host.inputs {
            if let Some(index) = input.output_index {
                let slot = values
                    .get_mut(index)
                    .ok_or_else(|| MeSessionError::Contract {
                        reason: format!(
                            "recorded input '{name}' has no declared output at {index}"
                        ),
                    })?;
                *slot = input.value;
            }
        }
        for (name, value) in names.iter().zip(values) {
            visit(name, value);
        }
        for input in self.host.inputs.values() {
            if input.output_index.is_none() {
                visit(&input.label, input.value);
            }
        }
        Ok(())
    }

    /// Current visible values keyed by name, including host-applied inputs.
    pub fn visible_values(&self) -> Result<IndexMap<String, f64>, MeSessionError> {
        let count = self.host.output_labels.len();
        let mut visible = IndexMap::with_capacity(count);
        self.visit_visible_values(|name, value| {
            visible.insert(name.to_string(), value);
        })?;
        Ok(visible)
    }
}
