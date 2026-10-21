use super::*;
use rumoca_core::{SourceId, Span};
use rumoca_ir_solve::{BinaryOp, LinearOp, ScalarProgramBlock, UnaryOp};
use std::{borrow::Cow, collections::BTreeMap};

#[derive(Default)]
struct MemoryStore(BTreeMap<Vec<u8>, Vec<u8>>);

impl CacheKvStore for MemoryStore {
    fn get(&self, key: &[u8]) -> Option<Cow<'_, [u8]>> {
        self.0.get(key).map(|value| Cow::Borrowed(value.as_slice()))
    }
    fn insert(&mut self, key: &[u8], value: Vec<u8>) {
        assert!(self.0.len() < 64);
        assert!(value.len() < 1024 * 1024);
        self.0.insert(key.to_vec(), value);
    }
}

fn compile(constant: f64) -> crate::CompiledExpressionRows {
    use LinearOp as L;
    let source = ScalarProgramBlock::with_program_spans(
        vec![vec![
            L::LoadY { dst: 0, index: 0 },
            L::Unary {
                dst: 1,
                op: UnaryOp::Sin,
                arg: 0,
            },
            L::LoadP { dst: 2, index: 0 },
            L::Binary {
                dst: 3,
                op: BinaryOp::Mul,
                lhs: 1,
                rhs: 2,
            },
            L::Const {
                dst: 4,
                value: constant,
            },
            L::Binary {
                dst: 5,
                op: BinaryOp::Add,
                lhs: 3,
                rhs: 4,
            },
            L::StoreOutput { src: 5 },
        ]],
        vec![Span::from_offsets(
            SourceId::from_source_name("cache.mo"),
            0,
            1,
        )],
    )
    .unwrap();
    crate::compile_expression_scalar_program_block(&source).unwrap()
}

fn values(compiled: &crate::CompiledExpressionRows) -> Vec<u64> {
    [-100., -1., -0., 0., 0.25, 1., 100.]
        .into_iter()
        .flat_map(|x| {
            [0., 0.5, -2.5].map(|p| {
                let mut out = [0.];
                compiled.call(&[x], &[p], 0., &mut out).unwrap();
                out[0].to_bits()
            })
        })
        .collect()
}

#[test]
fn cached_native_code_matches_fresh_modules_and_changed_inputs() {
    let store = Rc::new(RefCell::new(MemoryStore::default()));
    let plain = values(&compile(3.));
    let (cold, cold_usage) = with_native_code_cache(store.clone(), || values(&compile(3.)));
    assert_eq!(cold, plain);
    assert!(cold_usage.compilations > 0);
    let (warm, warm_usage) = with_native_code_cache(store.clone(), || values(&compile(3.)));
    assert_eq!(warm, plain);
    assert_eq!(warm_usage.attempts, cold_usage.attempts);
    assert_eq!(warm_usage.hits, warm_usage.attempts);
    assert_eq!(warm_usage.compilations, 0);
    let (changed, changed_usage) = with_native_code_cache(store, || values(&compile(4.)));
    assert_ne!(changed, plain);
    assert!(changed_usage.compilations > 0);
    assert!(ACTIVE.with(|active| active.borrow().is_none()));
}

#[test]
fn cache_scope_restores_parent_on_return_unwind_and_new_threads() {
    let outer = Rc::new(RefCell::new(MemoryStore::default()));
    let inner = Rc::new(RefCell::new(MemoryStore::default()));
    let (_, usage) = with_native_code_cache(outer.clone(), || {
        drop(compile(1.));
        let (_, child) = with_native_code_cache(inner.clone(), || drop(compile(2.)));
        assert!(child.compilations > 0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_native_code_cache(inner.clone(), || panic!("scope unwind"));
        }));
        assert!(result.is_err());
        std::thread::spawn(|| {
            assert!(ACTIVE.with(|active| active.borrow().is_none()));
        })
        .join()
        .unwrap();
        drop(compile(1.));
    });
    assert!(usage.compilations > 0 && usage.hits > 0);
    assert_eq!(usage.attempts, usage.compilations + usage.hits);
    assert!(ACTIVE.with(|active| active.borrow().is_none()));
}

#[test]
fn held_cache_store_rejects_compilation_without_panicking() {
    let store = Rc::new(RefCell::new(MemoryStore::default()));
    let source = ScalarProgramBlock::with_program_spans(
        vec![vec![
            LinearOp::Const { dst: 0, value: 1. },
            LinearOp::StoreOutput { src: 0 },
        ]],
        vec![Span::from_offsets(
            SourceId::from_source_name("held.mo"),
            0,
            1,
        )],
    )
    .unwrap();
    let _lease = store.borrow_mut();
    let (result, _) = with_native_code_cache(store.clone(), || {
        crate::compile_expression_scalar_program_block(&source)
    });
    assert!(
        result
            .err()
            .unwrap()
            .to_string()
            .contains("native compilation cache is already in use")
    );
    assert!(ACTIVE.with(|active| active.borrow().is_none()));
}

extern "C" fn first_import(x: f64) -> f64 {
    x * 1.25
}
extern "C" fn second_import(x: f64) -> f64 {
    x * -2.5
}

fn relocated_import(shift: bool) -> Vec<u64> {
    use cranelift_codegen::ir::{AbiParam, InstBuilder, types};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
    use cranelift_jit::JITBuilder;
    use cranelift_module::Linkage;
    let mut builder = JITBuilder::new(cranelift_module::default_libcall_names()).unwrap();
    builder.symbol(
        "host",
        if shift {
            second_import as *const u8
        } else {
            first_import as *const u8
        },
    );
    let mut module = JITModule::new(builder);
    let mut signature = module.make_signature();
    signature.params.push(AbiParam::new(types::F64));
    signature.returns.push(AbiParam::new(types::F64));
    if shift {
        module
            .declare_function("unused", Linkage::Import, &signature)
            .unwrap();
    }
    let host = module
        .declare_function("host", Linkage::Import, &signature)
        .unwrap();
    let id = module
        .declare_function("entry", Linkage::Export, &signature)
        .unwrap();
    let mut context = module.make_context();
    context.func.signature = signature;
    let mut frontend = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut context.func, &mut frontend);
    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    builder.seal_block(entry);
    let input = builder.block_params(entry)[0];
    let host = module.declare_func_in_func(host, builder.func);
    let call = builder.ins().call(host, &[input]);
    let value = builder.inst_results(call)[0];
    builder.ins().return_(&[value]);
    builder.finalize();
    define_function(&mut module, id, &mut context).unwrap();
    module.finalize_definitions().unwrap();
    // SAFETY: the function has the issued F64 -> F64 ABI and its arena is live.
    let function: unsafe extern "C" fn(f64) -> f64 =
        unsafe { std::mem::transmute(module.get_finalized_function(id)) };
    let output = [-100., -1., -0., 0., 0.25, 1., 100.]
        .map(|x| {
            let expected = if shift {
                second_import(x)
            } else {
                first_import(x)
            };
            let actual = unsafe { function(x) };
            assert_eq!(actual.to_bits(), expected.to_bits());
            actual.to_bits()
        })
        .to_vec();
    // SAFETY: no function pointer is called after this point.
    unsafe { module.free_memory() };
    output
}

#[test]
fn cache_hits_rebind_function_identity_and_host_addresses() {
    let store = Rc::new(RefCell::new(MemoryStore::default()));
    let (first, cold) = with_native_code_cache(store.clone(), || relocated_import(false));
    let (second, warm) = with_native_code_cache(store, || relocated_import(true));
    assert_eq!(cold.compilations, 1);
    assert_eq!(warm.hits, 1);
    assert_ne!(first, second);
}

#[test]
fn malformed_native_artifacts_report_rejection_and_rebuild_exact_results() {
    let store = Rc::new(RefCell::new(MemoryStore::default()));
    let (expected, _) = with_native_code_cache(store.clone(), || values(&compile(3.)));
    for artifact in store.borrow_mut().0.values_mut() {
        *artifact = vec![0];
    }
    let (actual, usage) = with_native_code_cache(store, || values(&compile(3.)));
    assert_eq!(actual, expected);
    assert!(usage.rejected_entries > 0);
    assert_eq!(usage.rejected_entries, usage.compilations);
    assert_eq!(usage.hits, 0);
}
