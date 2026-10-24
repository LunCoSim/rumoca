use crate::{LinearOp, ScalarProgramBlock};

pub(crate) struct ProgramOutputBinding {
    pub program: usize,
    pub offset: usize,
    pub output_count: usize,
    pub logical_index: usize,
}

impl ScalarProgramBlock {
    /// Borrow the complete output pairing proved by this block's constructor.
    pub(crate) fn output_bindings(&self) -> impl Iterator<Item = ProgramOutputBinding> + '_ {
        output_bindings(self.programs(), self.output_indices())
    }
}

/// Pair constructor-validated program outputs with their logical identities.
pub(crate) fn output_bindings<'a>(
    programs: &'a [Vec<LinearOp>],
    output_indices: &'a [usize],
) -> impl Iterator<Item = ProgramOutputBinding> + 'a {
    programs
        .iter()
        .enumerate()
        .flat_map(|(program, ops)| {
            let count = ScalarProgramBlock::program_output_count(ops);
            (0..count).map(move |offset| (program, offset, count))
        })
        .zip(output_indices)
        .map(
            |((program, offset, output_count), &logical_index)| ProgramOutputBinding {
                program,
                offset,
                output_count,
                logical_index,
            },
        )
}
