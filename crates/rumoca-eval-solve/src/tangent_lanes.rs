//! Evaluation of checked multi-lane tangent programs.

use std::cell::RefCell;

use rumoca_ir_solve::{ColoredTangentPlan, TangentLaneProgram, TornTangentPlan};

use crate::{
    EvalSolveError, OutputCursor, PreparedRowEval, RowEvalContext, RowEvalScratch,
    RowInputRequirements, SimulationRuntimeState, eval_row_prepared_maybe_fast,
    row_input_requirements, validate_input_requirements, validate_output_len,
};

/// A [`TangentLaneProgram`] prepared for repeated evaluation.
///
/// Seeds are element-major (`seed[i * lanes + l]` is lane `l` of seed index
/// `i`) and outputs lane-major (`out[l * m + o]` is lane `l` of output `o`).
pub struct PreparedTangentLaneProgram {
    program: TangentLaneProgram,
    /// The inputs the program reads, derived once; an error is reported at
    /// every evaluation.
    requirements: Result<RowInputRequirements, EvalSolveError>,
    scratch: RefCell<RowEvalScratch>,
}

impl PreparedTangentLaneProgram {
    #[must_use]
    pub fn new(program: TangentLaneProgram) -> Self {
        Self {
            requirements: row_input_requirements(program.ops()),
            program,
            scratch: RefCell::new(RowEvalScratch::default()),
        }
    }

    #[must_use]
    pub const fn program(&self) -> &TangentLaneProgram {
        &self.program
    }

    /// Evaluate every lane of every output into `out`.
    pub fn eval(
        &self,
        y: &[f64],
        p: &[f64],
        t: f64,
        context: RowEvalContext<'_>,
        out: &mut [f64],
    ) -> Result<(), EvalSolveError> {
        let local_runtime_state;
        let context = match context.runtime_state {
            Some(_) => context,
            None => {
                local_runtime_state = SimulationRuntimeState::new();
                context.with_runtime_state(&local_runtime_state)
            }
        };
        let ops = self.program.ops();
        validate_output_len(out, self.program.lanes() * self.program.lane_outputs())?;
        validate_input_requirements(self.requirements.clone()?, y, p, context.seed)?;
        let mut scratch = self.scratch.borrow_mut();
        let mut sink = OutputCursor::new(out);
        eval_row_prepared_maybe_fast(
            PreparedRowEval::new(ops, self.program.register_count(), y, p, t, context),
            true,
            &mut scratch,
            &mut sink,
        )
    }
}

/// The point and context one tangent evaluation reads.
#[derive(Clone, Copy)]
pub struct TangentPoint<'a> {
    pub y: &'a [f64],
    pub p: &'a [f64],
    pub t: f64,
    pub context: RowEvalContext<'a>,
}

/// The reduced tear Jacobian and the recovered-coordinate sensitivities of
/// one torn block, both row-major with one column per tear.
#[derive(Clone, Debug, PartialEq)]
pub struct TornTangentJacobian {
    pub residual: Vec<f64>,
    pub recovered: Vec<f64>,
}

/// Answers one prepared tangent program: `(program, seed, out)`. A decline
/// precedes execution; an admitted execution error must propagate.
pub type TangentProgramCall<'a> =
    dyn FnMut(usize, &[f64], &mut Vec<f64>) -> Result<bool, EvalSolveError> + 'a;

/// A [`TornTangentPlan`] prepared for repeated evaluation.
pub struct TornTangentEvaluator {
    plan: TornTangentPlan,
    programs: Vec<PreparedTangentLaneProgram>,
    /// The JVP rows a one-direction plan evaluates.
    directions: Option<std::rc::Rc<crate::PreparedEvaluationBlock>>,
    workspace: std::rc::Rc<RefCell<TornTangentWorkspace>>,
}

/// Prepare torn blocks over one immutable JVP source, sharing its directional
/// program preparation and one checked-out evaluation workspace.
pub struct TornTangentPreparation<'a> {
    jvp: &'a rumoca_ir_solve::ScalarProgramBlock,
    directions: Option<std::rc::Rc<crate::PreparedEvaluationBlock>>,
    workspace: std::rc::Rc<RefCell<TornTangentWorkspace>>,
}

#[derive(Default)]
struct TornTangentWorkspace {
    seed: Vec<f64>,
    seed_writes: Vec<std::ops::Range<usize>>,
    out: Vec<f64>,
    coefficients: Vec<f64>,
}

impl TornTangentWorkspace {
    fn prepare(
        &mut self,
        plan: &TornTangentPlan,
        coordinates: usize,
        lanes: usize,
    ) -> Result<usize, EvalSolveError> {
        for range in self.seed_writes.drain(..) {
            self.seed[range].fill(0.0);
        }
        let len = coordinates
            .checked_mul(lanes)
            .ok_or_else(|| EvalSolveError::ShapeContract {
                message: "torn tangent seed extent overflows".into(),
                span: None,
            })?;
        if len > self.seed.len() {
            self.seed
                .try_reserve_exact(len - self.seed.len())
                .map_err(|_| EvalSolveError::ShapeContract {
                    message: "torn tangent seed allocation failed".into(),
                    span: None,
                })?;
            self.seed.resize(len, 0.0);
        }
        self.out.clear();
        self.coefficients.clear();
        // Register every possible seed write before evaluation, so even an
        // interrupted sweep is cleared before another plan uses these buffers.
        let targets = plan
            .tear_targets()
            .iter()
            .copied()
            .chain(plan.steps().iter().map(|step| step.target));
        for target in targets {
            if target >= coordinates {
                return Err(EvalSolveError::MissingInput {
                    vector: "y",
                    index: target,
                    len: coordinates,
                    span: None,
                });
            }
            self.seed_writes.push(target * lanes..(target + 1) * lanes);
        }
        Ok(len)
    }
}

impl<'a> TornTangentPreparation<'a> {
    #[must_use]
    pub fn new(jvp: &'a rumoca_ir_solve::ScalarProgramBlock) -> Self {
        Self {
            jvp,
            directions: None,
            workspace: std::rc::Rc::new(RefCell::new(TornTangentWorkspace::default())),
        }
    }

    /// Prepare `plan` over this group's JVP rows. Evaluators retain the shared
    /// preparation and workspace independently of this construction scope.
    pub fn prepare(
        &mut self,
        plan: TornTangentPlan,
    ) -> Result<TornTangentEvaluator, EvalSolveError> {
        let directions = if plan.directional() {
            if self.directions.is_none() {
                self.directions = Some(std::rc::Rc::new(crate::PreparedEvaluationBlock::new(
                    self.jvp.clone(),
                )?));
            }
            self.directions.clone()
        } else {
            None
        };
        let programs = plan
            .programs()
            .iter()
            .cloned()
            .map(PreparedTangentLaneProgram::new)
            .collect();
        Ok(TornTangentEvaluator {
            plan,
            programs,
            directions,
            workspace: self.workspace.clone(),
        })
    }
}

impl TornTangentEvaluator {
    /// Prepare one standalone plan over the JVP rows it was derived from.
    pub fn new(
        plan: TornTangentPlan,
        jvp: &rumoca_ir_solve::ScalarProgramBlock,
    ) -> Result<Self, EvalSolveError> {
        TornTangentPreparation::new(jvp).prepare(plan)
    }

    #[must_use]
    pub const fn plan(&self) -> &TornTangentPlan {
        &self.plan
    }

    /// Evaluate the reduced tear Jacobian at `point`, whose causal
    /// coordinates hold the sweep of its tears. `None` when a causal
    /// coefficient is not finite ([`causal_coefficient_is_finite`]).
    pub fn eval(
        &self,
        point: TangentPoint<'_>,
    ) -> Result<Option<TornTangentJacobian>, EvalSolveError> {
        self.eval_through(point, &mut |_, _, _| Ok(false))
    }

    /// [`Self::eval`] with `call` answering the plan's prepared tangent
    /// programs: `call(program, seed, out)` writes the program's outputs under
    /// `seed` and returns `true`, or returns `false` to leave the program to
    /// the prepared evaluator. It must give the prepared evaluator's values.
    pub fn eval_through(
        &self,
        point: TangentPoint<'_>,
        call: &mut TangentProgramCall<'_>,
    ) -> Result<Option<TornTangentJacobian>, EvalSolveError> {
        // A callback can reenter this evaluator. Keep each invocation's buffers
        // outside the cell until its evaluation has returned.
        let mut workspace = self.workspace.take();
        let result = match &self.directions {
            Some(directions) => self.eval_directions(directions, point, call, &mut workspace),
            None => self.eval_multilane(point, call, &mut workspace),
        };
        self.workspace.replace(workspace);
        result
    }

    fn eval_multilane(
        &self,
        point: TangentPoint<'_>,
        call: &mut TangentProgramCall<'_>,
        workspace: &mut TornTangentWorkspace,
    ) -> Result<Option<TornTangentJacobian>, EvalSolveError> {
        let lanes = self.plan.lanes();
        let tears = lanes - 1;
        let steps = self.plan.steps();
        let seed_len = workspace.prepare(&self.plan, point.y.len(), lanes)?;
        let TornTangentWorkspace { seed, out, .. } = workspace;
        let seed = &mut seed[..seed_len];
        for (column, &target) in self.plan.tear_targets().iter().enumerate() {
            seed[target * lanes + column] = 1.0;
        }
        let mut recovered = Vec::with_capacity(steps.len() * tears);
        let mut outputs = 0;
        for (index, step) in steps.iter().enumerate() {
            if step.group > 0 {
                let group = &steps[index..index + step.group];
                seed_group(seed, group, (lanes, tears), 1.0);
                outputs = self.eval_lanes(point, seed, step.source.program, out, call)?;
                seed_group(seed, group, (lanes, tears), 0.0);
            }
            let tangent = |lane: usize| out[lane * outputs + step.source.output];
            let coefficient = tangent(tears);
            if !causal_coefficient_is_finite(coefficient) {
                return Ok(None);
            }
            for lane in 0..tears {
                let value = -tangent(lane) / coefficient;
                seed[step.target * lanes + lane] = value;
                recovered.push(value);
            }
        }
        let mut residual = vec![0.0; tears * tears];
        for (row, entry) in self.plan.residuals().iter().enumerate() {
            if entry.group > 0 {
                outputs = self.eval_lanes(point, seed, entry.source.program, out, call)?;
            }
            for lane in 0..tears {
                residual[row * tears + lane] = out[lane * outputs + entry.source.output];
            }
        }
        Ok(Some(TornTangentJacobian {
            residual,
            recovered,
        }))
    }

    /// The one-direction form: every step's coefficient with its target seeded
    /// alone, then each tear column's tangents through the steps in sweep
    /// order and the reduced rows. Each value equals its lane in the
    /// multi-lane form.
    fn eval_directions(
        &self,
        directions: &crate::PreparedEvaluationBlock,
        point: TangentPoint<'_>,
        call: &mut TangentProgramCall<'_>,
        workspace: &mut TornTangentWorkspace,
    ) -> Result<Option<TornTangentJacobian>, EvalSolveError> {
        let tears = self.plan.lanes() - 1;
        let steps = self.plan.steps();
        let seed_len = workspace.prepare(&self.plan, point.y.len(), 1)?;
        let TornTangentWorkspace {
            seed,
            out,
            coefficients,
            ..
        } = workspace;
        let seed = &mut seed[..seed_len];
        let mut direction = |seed: &[f64], program: usize, out: &mut Vec<f64>| {
            if call(program, seed, out)? {
                return Ok(());
            }
            directions.eval_row_outputs_unchecked_with_context(
                program,
                point.y,
                point.p,
                point.t,
                RowEvalContext {
                    seed: Some(seed),
                    ..point.context
                },
                out,
            )
        };
        for (index, step) in steps.iter().enumerate() {
            if step.group == 0 {
                continue;
            }
            let group = &steps[index..index + step.group];
            seed_group(seed, group, (1, 0), 1.0);
            direction(seed, step.source.program, out)?;
            seed_group(seed, group, (1, 0), 0.0);
            coefficients.extend(group.iter().map(|member| out[member.source.output]));
            if coefficients[index..]
                .iter()
                .any(|coefficient| !causal_coefficient_is_finite(*coefficient))
            {
                return Ok(None);
            }
        }
        let mut recovered = vec![0.0; steps.len() * tears];
        let mut residual = vec![0.0; tears * tears];
        for (column, &tear) in self.plan.tear_targets().iter().enumerate() {
            seed[tear] = 1.0;
            for (index, step) in steps.iter().enumerate() {
                let fresh = step.group > 0;
                fresh
                    .then(|| direction(seed, step.source.program, out))
                    .transpose()?;
                let value = -out[step.source.output] / coefficients[index];
                seed[step.target] = value;
                recovered[index * tears + column] = value;
            }
            for (row, entry) in self.plan.residuals().iter().enumerate() {
                let fresh = entry.group > 0;
                fresh
                    .then(|| direction(seed, entry.source.program, out))
                    .transpose()?;
                residual[row * tears + column] = out[entry.source.output];
            }
            seed[tear] = 0.0;
            for step in steps {
                seed[step.target] = 0.0;
            }
        }
        Ok(Some(TornTangentJacobian {
            residual,
            recovered,
        }))
    }

    /// Evaluate lane program `program` under `seed` into `out`; the number of
    /// outputs per lane.
    fn eval_lanes(
        &self,
        point: TangentPoint<'_>,
        seed: &[f64],
        program: usize,
        out: &mut Vec<f64>,
        call: &mut TangentProgramCall<'_>,
    ) -> Result<usize, EvalSolveError> {
        let prepared = &self.programs[program];
        let outputs = prepared.program().lane_outputs();
        out.resize(self.plan.lanes() * outputs, 0.0);
        if call(program, seed, out)? {
            return Ok(outputs);
        }
        prepared.eval(
            point.y,
            point.p,
            point.t,
            RowEvalContext {
                seed: Some(seed),
                ..point.context
            },
            out,
        )?;
        Ok(outputs)
    }
}

/// Whether a causal step's evaluated coefficient admits the division that
/// recovers its target. Construction proves every retained step's coefficient
/// a nonzero constant (SPEC_0043 §4), and the forward tangent of a row
/// whose other terms do not read the target equals that constant, so a zero
/// here is a construction defect a debug build asserts against. Run-time
/// values can still make it non-finite (a zero tangent times an infinite
/// partial is NaN), which declines the torn solve.
#[must_use]
pub fn causal_coefficient_is_finite(coefficient: f64) -> bool {
    debug_assert_ne!(
        coefficient, 0.0,
        "a causal step's coefficient is proven nonzero at construction"
    );
    coefficient.is_finite()
}

/// Set lane `lane` of every member target of `group` in a seed with `stride`
/// lanes per coordinate to `value`.
fn seed_group(
    seed: &mut [f64],
    group: &[rumoca_ir_solve::TornTangentStep],
    (stride, lane): (usize, usize),
    value: f64,
) {
    for member in group {
        seed[member.target * stride + lane] = value;
    }
}

/// A [`ColoredTangentPlan`] prepared for repeated evaluation.
pub struct ColoredTangentEvaluator {
    plan: ColoredTangentPlan,
    programs: Vec<PreparedTangentLaneProgram>,
}

impl ColoredTangentEvaluator {
    #[must_use]
    pub fn new(plan: ColoredTangentPlan) -> Self {
        let programs = plan
            .programs()
            .iter()
            .cloned()
            .map(PreparedTangentLaneProgram::new)
            .collect();
        Self { plan, programs }
    }

    #[must_use]
    pub const fn plan(&self) -> &ColoredTangentPlan {
        &self.plan
    }

    /// Evaluate every placement into the column-major buffer `out` (length
    /// [`ColoredTangentPlan::output_len`]), leaving other entries untouched.
    /// `seed_len` is the length of one direction of the application's seed.
    pub fn eval(
        &self,
        (y, p, t): (&[f64], &[f64], f64),
        context: RowEvalContext<'_>,
        seed_len: usize,
        out: &mut [f64],
    ) -> Result<(), EvalSolveError> {
        validate_output_len(out, self.plan.output_len())?;
        let mut values = Vec::new();
        for call in self.plan.calls() {
            let prepared = &self.programs[call.program];
            let lanes = call.colors.len();
            let mut seed = vec![0.0; seed_len * lanes];
            let seeded = call.colors.iter().enumerate().flat_map(|(lane, &color)| {
                self.plan.color_seeds()[color]
                    .iter()
                    .map(move |index| index * lanes + lane)
            });
            for position in seeded {
                seed[position] = 1.0;
            }
            let outputs = prepared.program().lane_outputs();
            values.resize(lanes * outputs, 0.0);
            prepared.eval(
                y,
                p,
                t,
                RowEvalContext {
                    seed: Some(&seed),
                    ..context
                },
                &mut values,
            )?;
            for &(lane, offset, destination) in call.placements.iter() {
                out[destination] = values[lane * outputs + offset];
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
