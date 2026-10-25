//! Native storage for one result at an owner-issued complete input coordinate.

use super::*;
use cranelift_codegen::isa::TargetFrontendConfig;
use cranelift_module::{DataDescription, DataId};

/// One zero-initialized writable data object owned by the native module.
/// Code refers to it symbolically, so immutable compiled artifacts can be
/// rebound to fresh result cells. The acyclic call graph excludes reentry into
/// the same owner; distinct owners and primal/directional forms have distinct
/// storage. The compiled table preserves the non-Sync mutation boundary.
pub(super) struct InputResults {
    cells: DataId,
    input_cells: u32,
    output_cells: u32,
}

impl InputResults {
    pub(super) fn new(
        module: &mut JITModule,
        coordinate: solve::SolvePureCallInputCoordinate,
    ) -> Result<Self, CompileError> {
        let input_cells = u32::try_from(coordinate.input_cells()).map_err(|_| {
            CompileError::Backend("pure call input extent exceeds native storage".into())
        })?;
        let output_cells = u32::try_from(coordinate.output_cells()).map_err(|_| {
            CompileError::Backend("pure call output extent exceeds native storage".into())
        })?;
        let count = checked_cells(input_cells, output_cells, "pure result coordinate")?;
        let count = checked_cells(count, 1, "pure result flag")?;
        // All offsets used below must be representable before any code is issued.
        let bytes = cell_offset(count, "pure result storage")? as usize;
        let cells = module
            .declare_anonymous_data(true, false)
            .map_err(to_backend_err)?;
        let mut data = DataDescription::new();
        data.define_zeroinit(bytes);
        data.set_align(CELL_BYTES as u64);
        module.define_data(cells, &data).map_err(to_backend_err)?;
        Ok(Self {
            cells,
            input_cells,
            output_cells,
        })
    }

    pub(super) fn enter(
        &self,
        builder: &mut FunctionBuilder<'_>,
        module: &JITModule,
        config: TargetFrontendConfig,
        input: Value,
        output: Value,
    ) -> Result<Value, CompileError> {
        let data = declare_far_data_in_func(module, self.cells, builder.func)?;
        let cache = builder.ins().global_value(config.pointer_type(), data);
        let cached_input = cell_pointer(builder, cache, 1, "pure result inputs")?;
        let ready = builder.ins().load(types::I64, MemFlags::new(), cache, 0);
        let hit = builder.create_block();
        let compare = builder.create_block();
        let miss = builder.create_block();
        builder.ins().brif(ready, compare, &[], miss, &[]);
        builder.switch_to_block(compare);
        builder.seal_block(compare);
        let equal = equal_inputs(builder, config, input, cached_input, self.input_cells);
        builder.ins().brif(equal, hit, &[], miss, &[]);
        builder.switch_to_block(hit);
        builder.seal_block(hit);
        let cached_output = self.output_pointer(builder, cache)?;
        copy_cells(builder, config, cached_output, output, self.output_cells);
        status::succeed(builder);
        builder.switch_to_block(miss);
        builder.seal_block(miss);
        let zero = builder.ins().iconst(types::I64, 0);
        builder.ins().store(MemFlags::new(), zero, cache, 0);
        Ok(cache)
    }

    pub(super) fn publish(
        &self,
        builder: &mut FunctionBuilder<'_>,
        config: TargetFrontendConfig,
        cache: Value,
        input: Value,
        output: Value,
    ) -> Result<(), CompileError> {
        let cached_input = cell_pointer(builder, cache, 1, "pure result inputs")?;
        let cached_output = self.output_pointer(builder, cache)?;
        copy_cells(builder, config, input, cached_input, self.input_cells);
        copy_cells(builder, config, output, cached_output, self.output_cells);
        // Publish last, only after the complete typed body returned successfully.
        let one = builder.ins().iconst(types::I64, 1);
        builder.ins().store(MemFlags::new(), one, cache, 0);
        Ok(())
    }

    fn output_pointer(
        &self,
        builder: &mut FunctionBuilder<'_>,
        cache: Value,
    ) -> Result<Value, CompileError> {
        cell_pointer(builder, cache, 1 + self.input_cells, "pure result outputs")
    }
}

fn equal_inputs(
    builder: &mut FunctionBuilder<'_>,
    config: TargetFrontendConfig,
    input: Value,
    cached: Value,
    cells: u32,
) -> Value {
    if cells > 4 {
        let size = builder
            .ins()
            .iconst(config.pointer_type(), i64::from(cells) * CELL_BYTES as i64);
        let comparison = builder.call_memcmp(config, input, cached, size);
        return builder.ins().icmp_imm(IntCC::Equal, comparison, 0);
    }
    let mut equal = builder.ins().iconst(types::I8, 1);
    for index in 0..cells {
        let offset = (index as usize * CELL_BYTES) as i32;
        let left = builder
            .ins()
            .load(types::I64, MemFlags::new(), input, offset);
        let right = builder
            .ins()
            .load(types::I64, MemFlags::new(), cached, offset);
        let same = builder.ins().icmp(IntCC::Equal, left, right);
        equal = builder.ins().band(equal, same);
    }
    equal
}

fn copy_cells(
    builder: &mut FunctionBuilder<'_>,
    config: TargetFrontendConfig,
    source: Value,
    destination: Value,
    cells: u32,
) {
    builder.emit_small_memory_copy(
        config,
        destination,
        source,
        u64::from(cells) * CELL_BYTES as u64,
        CELL_BYTES as u8,
        CELL_BYTES as u8,
        true,
        MemFlags::new(),
    );
}
