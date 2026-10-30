//! Bounded native dispatch for independently selectable scalar programs.
//!
//! A group shares compilation and relocation overhead. Each switch arm retains
//! its own lowering state and returns after evaluating only that program.

use super::*;
use cranelift_frontend::Switch;

pub(super) type ResidualFn = unsafe extern "C" fn(*const f64, *const f64, f64, *mut f64, u32) -> u8;
type JacobianFn =
    unsafe extern "C" fn(*const f64, *const f64, f64, *const f64, *mut f64, u32) -> u8;

#[derive(Clone, Copy)]
pub(super) struct JacobianEntry {
    function: JacobianFn,
    index: u32,
}

impl JacobianEntry {
    pub(super) fn finalize(
        module: &JITModule,
        id: FuncId,
        index: u32,
    ) -> Result<Self, CompileError> {
        let pointer = finalized_pointer(module, id)?;
        // SAFETY: compile_selected_group with JacobianV declares JacobianFn.
        let function = unsafe { std::mem::transmute::<*const u8, JacobianFn>(pointer) };
        Ok(Self { function, index })
    }

    pub(super) unsafe fn call(
        self,
        y: &[f64],
        p: &[f64],
        t: f64,
        v: &[f64],
        out: &mut [f64],
    ) -> u8 {
        // SAFETY: the caller validates this selected program's input and output
        // extents. The entry and selector are issued together by the compiler.
        unsafe {
            (self.function)(
                y.as_ptr(),
                p.as_ptr(),
                t,
                v.as_ptr(),
                out.as_mut_ptr(),
                self.index,
            )
        }
    }
}

fn finalized_pointer(module: &JITModule, id: FuncId) -> Result<*const u8, CompileError> {
    let pointer = module.get_finalized_function(id);
    if pointer.is_null() {
        return Err(CompileError::Backend(
            "Cranelift returned a null dispatch function pointer".into(),
        ));
    }
    Ok(pointer)
}

pub(super) fn finalize_residual(
    module: &JITModule,
    id: FuncId,
) -> Result<ResidualFn, CompileError> {
    let pointer = finalized_pointer(module, id)?;
    // SAFETY: compile_selected_group with Residual declares ResidualFn.
    Ok(unsafe { std::mem::transmute::<*const u8, ResidualFn>(pointer) })
}

pub(super) fn group_end(rows: &[Vec<LinearOp>], start: usize) -> usize {
    const MAX_PROGRAMS: usize = 32;
    const MAX_OPERATIONS: usize = 4_000;
    let mut end = start + 1;
    let mut cost = residual_program_cost(&rows[start]);
    while end < rows.len() && end - start < MAX_PROGRAMS {
        let next = cost.saturating_add(residual_program_cost(&rows[end]));
        if next > MAX_OPERATIONS {
            break;
        }
        cost = next;
        end += 1;
    }
    end
}

impl CraneliftEmitter {
    pub(super) fn compile_selected_group(
        &mut self,
        rows: &[Vec<LinearOp>],
        kind: RowKind,
        name: &str,
    ) -> Result<FuncId, CompileError> {
        for row in rows {
            self.ensure_fold_programs(row, kind)?;
            self.ensure_conditional_programs(row, kind)?;
        }
        let pointer_type = self.module.target_config().pointer_type();
        let mut signature = self.module.make_signature();
        signature.returns.push(AbiParam::new(types::I8));
        signature.params.extend([
            AbiParam::new(pointer_type),
            AbiParam::new(pointer_type),
            AbiParam::new(types::F64),
        ]);
        if kind.has_seed() {
            signature.params.push(AbiParam::new(pointer_type));
        }
        signature
            .params
            .extend([AbiParam::new(pointer_type), AbiParam::new(types::I32)]);
        let id = self
            .module
            .declare_function(name, Linkage::Local, &signature)
            .map_err(to_backend_err)?;
        let mut context = self.module.make_context();
        context.func.signature = signature;
        let mut builder_context = FunctionBuilderContext::new();
        {
            let mut builder = FunctionBuilder::new(&mut context.func, &mut builder_context);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);
            let params = builder.block_params(entry).to_vec();
            let invalid = builder.create_block();
            let blocks: Vec<_> = rows.iter().map(|_| builder.create_block()).collect();
            let mut switch = Switch::new();
            for (index, block) in blocks.iter().copied().enumerate() {
                switch.set_entry(index as u128, block);
            }
            switch.emit(
                &mut builder,
                *params.last().expect("dispatch selector"),
                invalid,
            );
            builder.switch_to_block(invalid);
            builder.seal_block(invalid);
            let failure = builder.ins().iconst(types::I8, 4);
            builder.ins().return_(&[failure]);
            for (row, block) in rows.iter().zip(blocks) {
                builder.switch_to_block(block);
                builder.seal_block(block);
                self.lower_independent_row(&mut builder, row, kind, &params)?;
                status::succeed(&mut builder);
            }
            builder.finalize();
        }
        self.module
            .define_function(id, &mut context)
            .map_err(to_backend_err)?;
        self.module.clear_context(&mut context);
        Ok(id)
    }

    pub(super) fn lower_independent_row(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        row: &[LinearOp],
        kind: RowKind,
        params: &[cranelift_codegen::ir::Value],
    ) -> Result<(), CompileError> {
        let pointer_type = self.module.target_config().pointer_type();
        let (v_ptr, out_ptr) = if kind.has_seed() {
            (Some(params[3]), params[4])
        } else {
            (None, params[3])
        };
        let mut regs = HashMap::new();
        let mut loaded_y = HashMap::new();
        let mut loaded_p = HashMap::new();
        let backing_regs_ptr = create_row_register_tape(fb, pointer_type, row)?;
        RowLowerCtx {
            fb,
            module: &mut self.module,
            math: &mut self.math,
            regs: &mut regs,
            y_ptr: params[0],
            p_ptr: params[1],
            t_value: params[2],
            v_ptr,
            backing_regs_ptr,
            flags: MemFlags::new(),
            loaded_y: backing_regs_ptr.is_none().then_some(&mut loaded_y),
            loaded_p: backing_regs_ptr.is_none().then_some(&mut loaded_p),
            fold_carried: None,
            fold_indices: None,
            fold_index_constants: None,
            fold_captures: None,
            fold_captures_ptr: None,
            conditional_captures: None,
            conditional_captures_ptr: None,
            fold_functions: &self.fold_functions,
            conditional_functions: &self.conditional_functions,
            pure_call_functions: &self.pure_call_functions,
            pure_call_results: HashMap::new(),
            nested_fold_results: HashMap::new(),
            conditional_results: HashMap::new(),
            fold_carried_versions: Vec::new(),
            known_constants: HashMap::new(),
        }
        .lower_row_outputs(row, out_ptr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_dispatch_selector_fails_without_writing_output() {
        let rows = vec![vec![
            LinearOp::Const { dst: 0, value: 2.0 },
            LinearOp::StoreOutput { src: 0 },
        ]];
        let mut emitter = CraneliftEmitter::new(None).unwrap();
        let id = emitter
            .compile_selected_group(&rows, RowKind::Residual, "invalid_selector")
            .unwrap();
        finalize_jit_module(&mut emitter.module).unwrap();
        let function = finalize_residual(&emitter.module, id).unwrap();
        let mut output = [123.0];
        // SAFETY: the function uses no inputs, output has one slot, and an
        // invalid selector returns before entering any row or accessing data.
        let result = unsafe { function([].as_ptr(), [].as_ptr(), 0.0, output.as_mut_ptr(), 1) };
        assert!(matches!(
            status::check(result),
            Err(CompileError::Backend(_))
        ));
        assert_eq!(output, [123.0]);
    }
}
