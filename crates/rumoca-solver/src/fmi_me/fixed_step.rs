//! A fixed-step classical RK4 drive of one component through the Model
//! Exchange calls a fixed-step importer makes: the four stage derivatives,
//! then the accepted point, the completed integrator step, and the event
//! iteration any requested event needs. A generated component driven by the
//! same calls is compared with it step for step, reduced-chart switches
//! included (SPEC_0040 STRUCT-T07 constraint-fold chart rows).

use super::{
    MeAdvanceRequest, MeContinuousPoint, MeDerivativeHandle, MeError, MeEventCause, MeEventEntry,
    MeInstanceConfig, MeIntegrationError, MeIntegratorBackend, MeModelArtifact, MeStepCandidate,
    MeTime, SolveMeKernel, accepted_interval_contains, integrator::canonical_coordinate,
};

/// One reduced-chart switch the drive requested at an accepted step.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedStepChartSwitch {
    /// The accepted step, counted from one.
    pub step: usize,
    pub time: f64,
    pub from: usize,
    pub to: usize,
    /// The active chart's and the target's conditioning at the request.
    pub sigma_active: f64,
    pub sigma_target: f64,
}

/// The states after every accepted step (and any event at it) and the chart
/// switches requested along the way.
#[derive(Clone, Debug, Default)]
pub struct FixedStepRun {
    pub times: Vec<f64>,
    pub states: Vec<Vec<f64>>,
    pub switches: Vec<FixedStepChartSwitch>,
}

/// Drive `artifact` from time zero to `stop` in steps of `step`, with the
/// component's default refresh tolerance as the FMI tolerance. The drive is an
/// importer, so the component refreshes from its committed seed
/// (`RefreshExecutor::ImporterDriven`, SPEC_0044 ME-PROJ-005), as the
/// generated component it is compared with does.
pub fn fixed_step_rk4(
    artifact: &MeModelArtifact,
    step: f64,
    stop: f64,
) -> Result<FixedStepRun, MeError> {
    let tolerance = rumoca_eval_solve::projection_policy::ALGEBRAIC_REFRESH_TOLERANCE;
    let config = MeInstanceConfig::new("fixed-step", tolerance, 0.0, stop)?.importer_driven();
    let mut kernel = SolveMeKernel::instantiate(artifact.source(), &config)?;
    kernel.enter_initialization_mode()?;
    kernel.exit_initialization_mode()?;
    settle_events(&mut kernel)?;
    kernel.enter_continuous_time_mode()?;
    let mut states = vec![0.0; artifact.continuous_state_count()];
    kernel.get_continuous_states(&mut states)?;
    let steps = (stop / step).round() as usize;
    let mut run = FixedStepRun::default();
    for index in 1..=steps {
        let time = if index == steps {
            stop
        } else {
            index as f64 * step
        };
        advance_rk4(&mut kernel, &mut states, (index - 1) as f64 * step, step)?;
        kernel.set_time(MeTime::at(time))?;
        kernel.set_continuous_states(&states)?;
        let completed = kernel.completed_integrator_step(true)?;
        if completed.enter_event_mode {
            if let Some((from, to, sigma_active, sigma_target)) = kernel.requested_chart_switch() {
                run.switches.push(FixedStepChartSwitch {
                    step: index,
                    time,
                    from,
                    to,
                    sigma_active,
                    sigma_target,
                });
            }
            kernel.enter_event_mode(MeEventEntry {
                cause: MeEventCause::BasisChange,
                event_time: time,
                horizon: stop,
            })?;
            settle_events(&mut kernel)?;
            kernel.enter_continuous_time_mode()?;
            kernel.get_continuous_states(&mut states)?;
        }
        run.times.push(time);
        run.states.push(states.clone());
    }
    Ok(run)
}

fn settle_events(kernel: &mut SolveMeKernel) -> Result<(), MeError> {
    while kernel.update_discrete_states()?.discrete_states_need_update {}
    Ok(())
}

/// One classical RK4 step of `states` from `start`, evaluated as a fixed-step
/// importer does: set the time and states, then read the derivatives.
fn advance_rk4(
    kernel: &mut SolveMeKernel,
    states: &mut [f64],
    start: f64,
    step: f64,
) -> Result<(), MeError> {
    let derivative = |time: f64, point: &[f64]| -> Result<Vec<f64>, MeError> {
        kernel.set_time(MeTime::at(time))?;
        kernel.set_continuous_states(point)?;
        let mut out = vec![0.0; point.len()];
        kernel.continuous_state_derivatives_into(&mut out)?;
        Ok(out)
    };
    let (accepted, _) = rk4_trial(start, start + step, step, states, derivative)?;
    states.copy_from_slice(&accepted);
    Ok(())
}

struct Rk4History {
    start: f64,
    end: f64,
    initial: Vec<f64>,
    final_state: Vec<f64>,
    stages: [Vec<f64>; 4],
}

/// Classical RK4 on an exact fixed-step lattice with a cubic continuous extension.
pub struct FixedRk4Integrator {
    origin: f64,
    index: u64,
    dt: f64,
    derivatives: Option<MeDerivativeHandle>,
    history: Option<Rk4History>,
}

fn integration_error(reason: impl Into<String>) -> MeIntegrationError {
    MeIntegrationError::Contract {
        reason: reason.into(),
    }
}

impl FixedRk4Integrator {
    /// Start the lattice at the component's initial coordinate.
    pub fn new(origin: f64, dt: f64) -> Result<Self, MeIntegrationError> {
        if !origin.is_finite() || !dt.is_finite() || dt <= 0.0 {
            return Err(integration_error(
                "fixed RK4 requires a finite origin and positive finite dt",
            ));
        }
        Ok(Self {
            origin: canonical_coordinate(origin),
            index: 0,
            dt,
            derivatives: None,
            history: None,
        })
    }

    fn validate_coordinate(&self, point: &MeContinuousPoint) -> Result<(), MeIntegrationError> {
        let expected = self.origin + self.index as f64 * self.dt;
        if point.time().to_bits() != expected.to_bits() {
            return Err(integration_error(format!(
                "fixed-rk4 requires grid coordinate {expected}, received {}",
                point.time()
            )));
        }
        Ok(())
    }
}

impl MeIntegratorBackend for FixedRk4Integrator {
    fn initialize(
        &mut self,
        point: &MeContinuousPoint,
        derivatives: MeDerivativeHandle,
    ) -> Result<(), MeIntegrationError> {
        self.validate_coordinate(point)?;
        self.derivatives = Some(derivatives);
        self.history = None;
        Ok(())
    }

    fn advance(
        &mut self,
        request: &MeAdvanceRequest,
    ) -> Result<MeStepCandidate, MeIntegrationError> {
        let point = request.current();
        self.validate_coordinate(point)?;
        let next = self
            .index
            .checked_add(1)
            .ok_or_else(|| integration_error("fixed-rk4 step index overflowed"))?;
        let end = self.origin + next as f64 * self.dt;
        let handle = self
            .derivatives
            .as_ref()
            .ok_or_else(|| integration_error("fixed-rk4 has no admitted derivative capability"))?;
        let eval = |time, state: &[f64]| {
            handle
                .derivatives(time, state)
                .map_err(|_| MeIntegrationError::DerivativeRefused)
        };
        let t = point.time();
        let h = self.dt;
        let y = point.states();
        let (values, stages) = rk4_trial(t, end, h, y, eval)?;
        self.history = Some(Rk4History {
            start: t,
            end,
            initial: y.to_vec(),
            final_state: values.clone(),
            stages,
        });
        self.index = next;
        // Classical RK4's cubic continuous extension has local order three.
        Ok(MeStepCandidate::new(end, values, 3))
    }

    fn sample(&self, time: f64, states: &mut [f64]) -> Result<(), MeIntegrationError> {
        let history = self
            .history
            .as_ref()
            .ok_or_else(|| integration_error("fixed-rk4 has no accepted interval to sample"))?;
        if states.len() != history.initial.len()
            || !accepted_interval_contains(history.start, history.end, time)
        {
            return Err(integration_error(
                "fixed-rk4 sample is outside its accepted interval or state width",
            ));
        }
        if time.to_bits() == history.end.to_bits() {
            states.copy_from_slice(&history.final_state);
            return Ok(());
        }
        if time.to_bits() == history.start.to_bits() {
            states.copy_from_slice(&history.initial);
            return Ok(());
        }
        let theta = (time - history.start) / self.dt;
        let square = theta * theta;
        let cube = square * theta;
        let weights = [
            theta - 1.5 * square + (2.0 / 3.0) * cube,
            square - (2.0 / 3.0) * cube,
            square - (2.0 / 3.0) * cube,
            -0.5 * square + (2.0 / 3.0) * cube,
        ];
        for (i, state) in states.iter_mut().enumerate() {
            let delta: f64 = history
                .stages
                .iter()
                .zip(weights)
                .map(|(stage, weight)| stage[i] * weight)
                .sum();
            *state = history.initial[i] + self.dt * delta;
        }
        Ok(())
    }

    fn truncate_reset(&mut self, point: &MeContinuousPoint) -> Result<(), MeIntegrationError> {
        self.validate_coordinate(point)?;
        self.history = None;
        Ok(())
    }
}

/// Return the accepted state and its stage derivatives from one classical RK4 trial.
fn rk4_trial<E>(
    start: f64,
    end: f64,
    step: f64,
    state: &[f64],
    mut derivative: impl FnMut(f64, &[f64]) -> Result<Vec<f64>, E>,
) -> Result<(Vec<f64>, [Vec<f64>; 4]), E> {
    let half = step * 0.5;
    let k1 = derivative(start, state)?;
    let k2 = derivative(start + half, &stage_state(state, &k1, half))?;
    let k3 = derivative(start + half, &stage_state(state, &k2, half))?;
    let k4 = derivative(end, &stage_state(state, &k3, step))?;
    let accepted = state
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value + (step / 6.0) * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index])
        })
        .collect();
    Ok((accepted, [k1, k2, k3, k4]))
}

fn stage_state(state: &[f64], derivative: &[f64], scale: f64) -> Vec<f64> {
    state
        .iter()
        .zip(derivative)
        .map(|(value, derivative)| value + scale * derivative)
        .collect()
}
