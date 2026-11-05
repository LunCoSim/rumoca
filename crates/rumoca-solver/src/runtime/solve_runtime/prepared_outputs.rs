//! Checked fixed output selections share one preparation and execution owner.

use super::*;

/// One selected output in deterministic program and request order.
#[derive(Clone)]
pub(super) struct SelectedProgramOutput {
    pub(super) program: usize,
    request_order: usize,
    pub(super) output: usize,
    pub(super) offset: usize,
}

pub(super) fn prepare_selected_program_outputs(
    selected: &mut Vec<SelectedProgramOutput>,
    block: &PreparedScalarProgramBlock,
    output_indices: &[usize],
) -> Result<(), RuntimeSolveError> {
    selected.clear();
    reserve_runtime_vec_capacity(selected, output_indices.len(), "selected program outputs")?;
    for (request_order, &output) in output_indices.iter().enumerate() {
        let (program, offset) = block.row_output_position(output).ok_or_else(|| {
            RuntimeSolveError::solve_ir(format!(
                "selected expression output {output} has no owning program"
            ))
        })?;
        selected.push(SelectedProgramOutput {
            program,
            request_order,
            output,
            offset,
        });
    }
    selected.sort_unstable_by_key(|row| (row.program, row.request_order));
    Ok(())
}

/// The source projection and optional native product contain no numerical history.
#[derive(Clone)]
pub(super) struct PreparedOutputSelection {
    block: PreparedScalarProgramBlock,
    compiled: Option<Rc<dyn CompiledSolveExpression>>,
    outputs: Box<[(usize, usize)]>,
    required_output_len: usize,
}

impl PreparedOutputSelection {
    pub(super) fn is_empty(&self) -> bool {
        self.outputs.is_empty()
    }

    pub(super) fn new(
        source: &PreparedScalarProgramBlock,
        selected: &[usize],
        backend: Option<&Rc<dyn SolveExecutionBackend>>,
        primary: Option<&Self>,
    ) -> Result<Self, EvalSolveError> {
        let OutputProjection {
            block,
            outputs,
            required_output_len,
        } = OutputProjection::new(source, selected)?;
        // A changed selection must never execute superseded programs through
        // an overlay. Only exact immutable program reuse is admitted here.
        let reuse = match BlockReuse::of(primary.map(|batch| &batch.block), &block) {
            BlockReuse::Same => BlockReuse::Same,
            _ => BlockReuse::Fresh,
        };
        let compiled = reuse.expression(
            primary.and_then(|batch| batch.compiled.as_ref()),
            &block,
            &mut |block| {
                if block.is_empty() {
                    None
                } else {
                    compiled_expression(backend, "fixed output selection", block)
                }
            },
        );
        let block = reuse.prepared(primary.map(|batch| &batch.block), block)?;
        Ok(Self {
            block,
            compiled,
            outputs,
            required_output_len,
        })
    }

    pub(super) fn evaluate(
        &self,
        runtime: &SolveRuntime,
        point: RowEvalPoint<'_>,
        out: &mut [f64],
    ) -> Result<(), RuntimeSolveError> {
        if out.len() < self.required_output_len {
            return Err(RuntimeSolveError::solve_ir(format!(
                "fixed output selection requires {} output entries, got {}",
                self.required_output_len,
                out.len()
            )));
        }
        if self.outputs.is_empty() {
            return Ok(());
        }
        let mut values = runtime.compiled_output_scratch.borrow_mut();
        values.resize(self.block.len(), 0.0);
        if let Some(compiled) = &self.compiled {
            compiled
                .call(
                    point.y,
                    point.p,
                    point.t,
                    runtime.model.external_tables.as_slice(),
                    &mut values,
                )
                .map_err(RuntimeSolveError::solve_ir)?;
        } else {
            self.block.eval_with_context(
                point.y,
                point.p,
                point.t,
                runtime.row_eval_context(),
                &mut values,
            )?;
        }
        for &(output, offset) in &self.outputs {
            out[output] = values[offset];
        }
        Ok(())
    }
}

/// Checked source/output ownership produced once for one fixed selection.
struct OutputProjection {
    block: solve::ScalarProgramBlock,
    outputs: Box<[(usize, usize)]>,
    required_output_len: usize,
}

impl OutputProjection {
    fn new(
        source: &PreparedScalarProgramBlock,
        selected: &[usize],
    ) -> Result<Self, EvalSolveError> {
        let invalid = |message: String| EvalSolveError::InvalidRow {
            message,
            span: source.block().first_source_span(),
        };
        let mut selection = Vec::<SelectedProgramOutput>::new();
        prepare_selected_program_outputs(&mut selection, source, selected)
            .map_err(|error| invalid(error.to_string()))?;
        let mut programs = Vec::new();
        let mut spans = Vec::new();
        let mut outputs = Vec::with_capacity(selection.len());
        let mut count = 0usize;
        let mut required_output_len = 0usize;
        for group in selection.chunk_by(|a, b| a.program == b.program) {
            let program = group[0].program;
            programs.push(
                source
                    .block()
                    .programs()
                    .get(program)
                    .ok_or_else(|| invalid(format!("selected source program {program} is absent")))?
                    .clone(),
            );
            spans.push(source.block().program_span(program).ok_or_else(|| {
                invalid(format!(
                    "selected source program {program} has no provenance"
                ))
            })?);
            for row in group {
                outputs.push((
                    row.output,
                    count
                        .checked_add(row.offset)
                        .ok_or_else(|| invalid("selected output offset overflow".into()))?,
                ));
                required_output_len = required_output_len.max(
                    row.output
                        .checked_add(1)
                        .ok_or_else(|| invalid("selected output index overflow".into()))?,
                );
            }
            count = count
                .checked_add(source.row_output_count(program).ok_or_else(|| {
                    invalid(format!(
                        "selected source program {program} has no output extent"
                    ))
                })?)
                .ok_or_else(|| invalid("selected output extent overflow".into()))?;
        }
        let block =
            solve::ScalarProgramBlock::with_output_indices(programs, spans, (0..count).collect())
                .map_err(|error| invalid(error.to_string()))?;
        Ok(Self {
            block,
            outputs: outputs.into_boxed_slice(),
            required_output_len,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> PreparedScalarProgramBlock {
        let span = rumoca_core::Span::from_offsets(
            rumoca_core::SourceId::from_source_name("root_selection.mo"),
            1,
            2,
        );
        let block = solve::ScalarProgramBlock::with_output_indices(
            vec![
                vec![
                    solve::LinearOp::LoadY { dst: 0, index: 0 },
                    solve::LinearOp::Const {
                        dst: 1,
                        value: 1.2345678901234567,
                    },
                    solve::LinearOp::StoreOutput { src: 0 },
                    solve::LinearOp::StoreOutput { src: 1 },
                ],
                vec![
                    solve::LinearOp::LoadY { dst: 0, index: 17 },
                    solve::LinearOp::StoreOutput { src: 0 },
                ],
            ],
            vec![span; 2],
            vec![3, 1, 8],
        )
        .unwrap();
        PreparedScalarProgramBlock::new(block).unwrap()
    }

    #[test]
    fn grouped_sparse_selection_preserves_bits_and_skips_other_programs() {
        let batch = PreparedOutputSelection::new(&source(), &[1, 3, 1], None, None).unwrap();
        let runtime = SolveRuntime::new_fixture(&solve::SolveModel::default()).unwrap();
        let mut out = [99.0; 9];
        batch
            .evaluate(
                &runtime,
                RowEvalPoint {
                    y: &[-0.0],
                    p: &[],
                    t: 0.0,
                },
                &mut out,
            )
            .unwrap();
        assert_eq!(out[1].to_bits(), 1.2345678901234567_f64.to_bits());
        assert_eq!(out[3].to_bits(), (-0.0_f64).to_bits());
        assert!(
            out.iter()
                .enumerate()
                .all(|(index, value)| index == 1 || index == 3 || *value == 99.0)
        );
    }

    #[test]
    fn missing_owner_and_short_destination_reject_without_execution() {
        assert!(PreparedOutputSelection::new(&source(), &[7], None, None).is_err());
        let batch = PreparedOutputSelection::new(&source(), &[1, 3], None, None).unwrap();
        let runtime = SolveRuntime::new_fixture(&solve::SolveModel::default()).unwrap();
        let mut out = [99.0; 3];
        let error = batch
            .evaluate(
                &runtime,
                RowEvalPoint {
                    y: &[],
                    p: &[],
                    t: 0.0,
                },
                &mut out,
            )
            .unwrap_err();
        assert!(error.to_string().contains("requires 4 output entries"));
        assert_eq!(out, [99.0; 3]);
    }
}
