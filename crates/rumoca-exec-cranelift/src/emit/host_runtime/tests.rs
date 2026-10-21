use super::host_jit_builder;
use crate::emit::owned_jit_module::OwnedJitModule;
use cranelift_codegen::ir::{AbiParam, InstBuilder, StackSlotData, StackSlotKind, types};
use cranelift_codegen::settings::ProbestackStrategy;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::JITModule;
use cranelift_module::{Linkage, Module};

const FRAME_BYTES: u32 = 768 * 1024;
const LAST_CELL: i32 = FRAME_BYTES as i32 - 8;

extern "C" fn read_frame_edges(frame: *const f64) -> f64 {
    // SAFETY: the generated caller supplies its live, aligned stack allocation.
    unsafe { *frame + *frame.byte_offset(LAST_CELL as isize) }
}

fn large_frame_module() -> (OwnedJitModule, unsafe extern "C" fn(f64) -> f64) {
    let mut builder = host_jit_builder().expect("native JIT configuration");
    builder.symbol("read_frame_edges", read_frame_edges as *const u8);
    let mut module = OwnedJitModule::new(JITModule::new(builder));
    assert!(module.isa().flags().enable_probestack());
    assert_eq!(
        module.isa().flags().probestack_strategy(),
        ProbestackStrategy::Inline
    );
    let pointer = module.target_config().pointer_type();
    let mut helper_signature = module.make_signature();
    helper_signature.params.push(AbiParam::new(pointer));
    helper_signature.returns.push(AbiParam::new(types::F64));
    let helper = module
        .declare_function("read_frame_edges", Linkage::Import, &helper_signature)
        .unwrap();
    let mut signature = module.make_signature();
    signature.params.push(AbiParam::new(types::F64));
    signature.returns.push(AbiParam::new(types::F64));
    let entry = module
        .declare_function("large_frame", Linkage::Export, &signature)
        .unwrap();
    let mut context = module.make_context();
    context.func.signature = signature;
    let mut frontend = FunctionBuilderContext::new();
    let mut function = FunctionBuilder::new(&mut context.func, &mut frontend);
    let block = function.create_block();
    function.append_block_params_for_function_params(block);
    function.switch_to_block(block);
    function.seal_block(block);
    let input = function.block_params(block)[0];
    let frame = function.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        FRAME_BYTES,
        3,
    ));
    let one = function.ins().f64const(1.0);
    let last = function.ins().fadd(input, one);
    function.ins().stack_store(input, frame, 0);
    function.ins().stack_store(last, frame, LAST_CELL);
    let address = function.ins().stack_addr(pointer, frame, 0);
    let helper = module.declare_func_in_func(helper, function.func);
    let call = function.ins().call(helper, &[address]);
    let called = function.inst_results(call)[0];
    let first = function.ins().stack_load(types::F64, frame, 0);
    let last = function.ins().stack_load(types::F64, frame, LAST_CELL);
    let edges = function.ins().fadd(first, last);
    let result = function.ins().fadd(called, edges);
    function.ins().return_(&[result]);
    function.finalize();
    module.define_function(entry, &mut context).unwrap();
    assert!(context.compiled_code().unwrap().frame_size >= FRAME_BYTES);
    module.finalize_definitions().unwrap();
    // SAFETY: the emitted signature is F64 -> F64; the returned owner retains code.
    let function = unsafe {
        std::mem::transmute::<*const u8, unsafe extern "C" fn(f64) -> f64>(
            module.get_finalized_function(entry),
        )
    };
    (module, function)
}

#[test]
fn large_jit_frame_on_fresh_worker_stack() {
    let (_module, function) = large_frame_module();
    // Windows reserves this stack but initially commits only its top pages.
    // Compile before spawning so compiler work cannot pre-touch the worker stack.
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            for input in [-2.0, 0.0, 3.25] {
                // SAFETY: the module remains alive until this worker has joined.
                let actual = unsafe { function(input) };
                assert_eq!(actual.to_bits(), (4.0 * input + 2.0).to_bits());
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
