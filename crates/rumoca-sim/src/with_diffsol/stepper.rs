//! Real-time stepper API for interactive simulation with external control inputs.
//!
//! The [`SimStepper`] allows stepping a simulation forward incrementally,
//! injecting input values between steps, and reading outputs — suitable for
//! controller-in-the-loop and real-time use cases.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use rumoca_eval_dae::runtime::VarEnv;
use rumoca_ir_dae as dae;
use rumoca_phase_solve::eliminate::EliminationResult;

use super::{Dae, SimError};
use crate::TimeoutBudget;
use crate::runtime::layout::SimulationContext;

/// Options for creating a [`SimStepper`].
#[derive(Debug, Clone)]
pub struct StepperOptions {
    pub rtol: f64,
    pub atol: f64,
    pub scalarize: bool,
    pub max_wall_seconds_per_step: Option<f64>,
    /// Input values to seed the initial-condition solve with. Keys are
    /// flattened scalar names (see [`SimStepper::input_names`]).
    ///
    /// Without this, unbound `input` variables resolve to the default
    /// value 0 during IC solve; a subsequent `set_input()` then
    /// introduces a step discontinuity at t=0 that the stepper cannot
    /// recover from. Pre-populating via this field lets the IC
    /// solver (and both the startup projection and diffsol's own
    /// consistent-IC probe) evaluate the residual at the caller's
    /// intended operating point, so the first `step()` starts from a
    /// consistent state.
    pub initial_inputs: HashMap<String, f64>,
}

impl Default for StepperOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-6,
            atol: 1e-6,
            scalarize: true,
            max_wall_seconds_per_step: None,
            initial_inputs: HashMap::new(),
        }
    }
}

/// Snapshot of the stepper's current state.
#[derive(Debug, Clone)]
pub struct StepperState {
    pub time: f64,
    pub values: HashMap<String, f64>,
}

/// Trait for type-erasing the diffsol solver internals.
pub(crate) trait StepperInner {
    fn step(&mut self, dt: f64, dae: &Dae, budget: &TimeoutBudget) -> Result<(), SimError>;
    fn time(&self) -> f64;
    fn solver_state_y(&self) -> Vec<f64>;
    /// Clear BDF history buffers and reset step size.
    /// Must be called when inputs change discontinuously so that
    /// the polynomial extrapolation does not diverge.
    fn reset_solver_history(&mut self);
    /// Overwrite the solver's current state vector. Used to seed the
    /// BDF restart with an algebraically-consistent y after an input
    /// discontinuity — without this, the solver restarts from a state
    /// consistent with the *previous* input values and has to find
    /// the new algebraic manifold on the fly inside its first
    /// substep, which is fragile.
    fn set_solver_state_y(&mut self, y: &[f64]);
}

/// A real-time simulation stepper that supports external input injection.
///
/// Created from a compiled DAE model, the stepper allows:
/// - Setting input values by name between steps
/// - Stepping forward by a time increment
/// - Reading state/output values by name
pub struct SimStepper {
    pub(crate) inner: Box<dyn StepperInner>,
    pub(crate) dae: Dae,
    pub(crate) sim_context: SimulationContext,
    #[allow(dead_code)]
    pub(crate) param_values: Vec<f64>,
    pub(crate) input_overrides: Rc<RefCell<HashMap<String, f64>>>,
    #[allow(dead_code)]
    pub(crate) n_x: usize,
    #[allow(dead_code)]
    pub(crate) n_total: usize,
    pub(crate) solver_names: Vec<String>,
    pub(crate) max_wall_seconds_per_step: Option<f64>,
    /// Absolute tolerance used as the convergence target for the
    /// algebraic re-projection after an input change. Matches the
    /// `atol` passed to the stepper at build time so the projection
    /// operates at the same precision as the integrator.
    pub(crate) atol: f64,
    /// Substitutions from algebraic elimination — used to reconstruct
    /// eliminated variables (e.g. outputs) in `get()` and `state()`.
    pub(crate) elim: EliminationResult,
    /// Set when `set_input` changes a value; cleared after solver history reset.
    pub(crate) inputs_dirty: bool,
}

impl SimStepper {
    /// Create a new stepper from a DAE model.
    ///
    /// This runs the full preparation pipeline (structural analysis, initial
    /// condition solving, kernel compilation) and creates a BDF solver ready
    /// for interactive stepping.
    pub fn new(dae: &dae::Dae, opts: StepperOptions) -> Result<Self, SimError> {
        super::build_stepper(dae, opts)
    }

    /// Set an input value by name. Takes effect on the next `step()` call.
    ///
    /// The name should match the flattened scalar name of an input variable
    /// (e.g., `"u"` for a scalar input, `"u[1]"` for an array element).
    pub fn set_input(&mut self, name: &str, value: f64) -> Result<(), SimError> {
        let valid_names = self.sim_context.input_scalar_names();
        if !valid_names.iter().any(|n| n == name) {
            return Err(SimError::SolverError(format!(
                "unknown input '{}', available inputs: {:?}",
                name, valid_names
            )));
        }
        let mut overrides = self.input_overrides.borrow_mut();
        let old = overrides.get(name).copied();
        overrides.insert(name.to_string(), value);
        if old != Some(value) {
            self.inputs_dirty = true;
        }
        Ok(())
    }

    /// Set multiple inputs at once.
    pub fn set_inputs(&mut self, inputs: &[(&str, f64)]) -> Result<(), SimError> {
        for &(name, value) in inputs {
            self.set_input(name, value)?;
        }
        Ok(())
    }

    /// Step the simulation forward by `dt` seconds.
    pub fn step(&mut self, dt: f64) -> Result<(), SimError> {
        if self.inputs_dirty {
            // Re-project algebraics onto the manifold implied by the new
            // input values BEFORE clearing BDF history. Without this, the
            // BDF restart would start from a state consistent with the
            // previous inputs and have to chase the algebraic jump inside
            // its first substep — which is fragile and can stall after
            // repeated input changes. Mirrors the t=0 startup projection.
            let y_before = self.inner.solver_state_y();
            let t_now = self.inner.time();
            let budget = TimeoutBudget::new(self.max_wall_seconds_per_step);
            if let Some(projected) = crate::with_diffsol::problem::project_algebraics_with_fixed_states_at_time(
                &self.dae,
                &y_before,
                self.n_x,
                t_now,
                self.atol,
                &budget,
                Some(self.input_overrides.clone()),
            )? {
                if projected.len() == y_before.len()
                    && projected
                        .iter()
                        .zip(y_before.iter())
                        .any(|(lhs, rhs)| (lhs - rhs).abs() > 1.0e-12)
                {
                    self.inner.set_solver_state_y(&projected);
                }
            }
            self.inner.reset_solver_history();
            self.inputs_dirty = false;
        }
        let budget = TimeoutBudget::new(self.max_wall_seconds_per_step);
        self.inner.step(dt, &self.dae, &budget)
    }

    /// Get the current simulation time.
    pub fn time(&self) -> f64 {
        self.inner.time()
    }

    /// Build a variable environment from the current solver state and inputs,
    /// including reconstructed eliminated variables.
    fn build_env(&self) -> VarEnv<f64> {
        let y = self.inner.solver_state_y();
        let mut env = VarEnv::default();
        env.vars.insert("time".to_string(), self.inner.time());
        for (idx, name) in self.solver_names.iter().enumerate() {
            if let Some(&val) = y.get(idx) {
                env.vars.insert(name.clone(), val);
            }
        }
        for (name, &val) in self.input_overrides.borrow().iter() {
            env.vars.insert(name.clone(), val);
        }
        for ((name, _var), &val) in self.dae.parameters.iter().zip(self.param_values.iter()) {
            env.vars.entry(name.as_str().to_string()).or_insert(val);
        }
        crate::reconstruct::apply_eliminated_substitutions_to_env(&self.elim, &mut env);
        env
    }

    /// Read a single variable value by name.
    ///
    /// Works for states, algebraics, outputs, inputs, and eliminated variables.
    pub fn get(&self, name: &str) -> Option<f64> {
        let y = self.inner.solver_state_y();
        if let Some(idx) = self.sim_context.solver_idx_for_target(name) {
            return y.get(idx).copied();
        }
        if let Some(&val) = self.input_overrides.borrow().get(name) {
            return Some(val);
        }
        if !self.elim.substitutions.is_empty() {
            let env = self.build_env();
            let val = env.vars.get(name).copied();
            if val.is_some() {
                return val;
            }
        }
        None
    }

    /// Get a snapshot of all current variable values.
    pub fn state(&self) -> StepperState {
        let env = self.build_env();
        let values = env.vars.into_iter().collect();
        StepperState {
            time: self.inner.time(),
            values,
        }
    }

    /// List available input names.
    pub fn input_names(&self) -> &[String] {
        self.sim_context.input_scalar_names()
    }

    /// List all solver variable names (states, algebraics, outputs).
    pub fn variable_names(&self) -> &[String] {
        &self.solver_names
    }
}
