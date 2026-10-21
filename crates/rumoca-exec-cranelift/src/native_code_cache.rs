//! Scoped reuse of Cranelift's relocatable compilation artifacts.
//!
//! The host owns storage, byte/entry limits and persistence. This adapter owns
//! compilation and rebinds every artifact's relocations in its new JIT module.
//! Cache keys are issued by Cranelift from its function stencil, compiler
//! version, target ISA and flags. No executable arena or simulation state is
//! retained by this scope. Stores must contain only trusted compiler artifacts.

use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

use cranelift_codegen::{CodegenError, Context, incremental_cache::CacheKvStore};
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Module, ModuleError, ModuleReloc};

#[cfg(test)]
mod tests;

type Store = Rc<RefCell<dyn CacheKvStore>>;

/// Completed native compiler work in one explicit preparation scope.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeCodeCacheUsage {
    pub attempts: u64,
    pub hits: u64,
    pub compilations: u64,
    /// Stored artifacts rejected by Cranelift; included in `compilations`.
    pub rejected_entries: u64,
}

#[derive(Clone)]
struct Scope {
    store: Store,
    usage: Rc<Cell<NativeCodeCacheUsage>>,
}

thread_local! {
    static ACTIVE: RefCell<Option<Scope>> = const { RefCell::new(None) };
}

/// Compile synchronously using one host-owned, bounded artifact store.
///
/// The scope follows this thread's nested compilation calls and restores its
/// caller's store on return or unwinding. It does not propagate to async tasks
/// or other threads. Numeric workspaces, executable arenas and FMI instances
/// remain owned by their ordinary compiled/runtime objects.
pub fn with_native_code_cache<T: CacheKvStore + 'static, R>(
    store: Rc<RefCell<T>>,
    compile: impl FnOnce() -> R,
) -> (R, NativeCodeCacheUsage) {
    let store: Store = store;
    let usage = Rc::new(Cell::new(NativeCodeCacheUsage::default()));
    let scope = Scope {
        store,
        usage: usage.clone(),
    };
    let _restore = Restore(ACTIVE.with(|active| active.replace(Some(scope))));
    let result = compile();
    (result, usage.get())
}

struct Restore(Option<Scope>);

impl Drop for Restore {
    fn drop(&mut self) {
        ACTIVE.with(|active| active.replace(self.0.take()));
    }
}

struct Lookup<'a> {
    store: &'a mut dyn CacheKvStore,
    matched: Cell<bool>,
}

impl CacheKvStore for Lookup<'_> {
    fn get(&self, key: &[u8]) -> Option<std::borrow::Cow<'_, [u8]>> {
        let artifact = self.store.get(key);
        self.matched.set(artifact.is_some());
        artifact
    }

    fn insert(&mut self, key: &[u8], artifact: Vec<u8>) {
        self.store.insert(key, artifact);
    }
}

pub(crate) fn define_function(
    module: &mut JITModule,
    id: FuncId,
    context: &mut Context,
) -> Result<(), Box<ModuleError>> {
    let Some(scope) = ACTIVE.with(|active| active.borrow().clone()) else {
        return module.define_function(id, context).map_err(Box::new);
    };
    let mut store = scope.store.try_borrow_mut().map_err(|_| {
        ModuleError::Compilation(CodegenError::Unsupported(
            "native compilation cache is already in use".into(),
        ))
    })?;
    let mut usage = scope.usage.get();
    usage.attempts += 1;
    scope.usage.set(usage);
    let mut lookup = Lookup {
        store: &mut *store,
        matched: Cell::new(false),
    };
    let hit = context
        .compile_with_cache(module.isa(), &mut lookup, &mut Default::default())
        .map_err(ModuleError::from)?
        .1;
    usage.hits += u64::from(hit);
    usage.compilations += u64::from(!hit);
    usage.rejected_entries += u64::from(lookup.matched.get() && !hit);
    scope.usage.set(usage);
    let code = context.compiled_code().ok_or_else(|| {
        ModuleError::Compilation(CodegenError::Unsupported(
            "native compilation produced no code".into(),
        ))
    })?;
    let relocations: Vec<_> = code
        .buffer
        .relocs()
        .iter()
        .map(|relocation| ModuleReloc::from_mach_reloc(relocation, &context.func, id))
        .collect();
    module
        .define_function_bytes(
            id,
            code.buffer.alignment as u64,
            code.code_buffer(),
            &relocations,
        )
        .map_err(Box::new)
}
