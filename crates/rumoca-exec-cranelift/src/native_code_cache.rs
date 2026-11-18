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
    sync::Mutex,
};

use cranelift_codegen::{Context, incremental_cache::CacheKvStore, isa::TargetIsa};
#[cfg(test)]
use cranelift_module::{Module, ModuleReloc};

#[cfg(test)]
pub(crate) mod tests;

type Store = Rc<RefCell<dyn CacheKvStore>>;

static JIT_PERF_MAP_REGISTRATION: Mutex<()> = Mutex::new(());

// Cranelift appends each perf-map row through multiple writes. Serialize its
// function registration only while that diagnostic map is enabled.
pub(crate) fn with_perf_map_registration<T>(enabled: bool, register: impl FnOnce() -> T) -> T {
    if enabled {
        let _guard = JIT_PERF_MAP_REGISTRATION
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        register()
    } else {
        register()
    }
}

pub(crate) fn perf_map_recording_enabled() -> bool {
    cfg!(unix) && std::env::var_os("PERF_BUILDID_DIR").is_some()
}

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

/// Whether preparation on this thread has admitted an artifact store.
pub(crate) fn cache_active() -> bool {
    ACTIVE.with(|active| active.borrow().is_some())
}

/// Compile with the same target as ordinary queued compilation.
pub(crate) fn compile_function(
    context: &mut Context,
    target: &dyn TargetIsa,
) -> Result<(), String> {
    let scope = ACTIVE.with(|active| active.borrow().clone());
    let Some(scope) = scope else {
        return context
            .compile(target, &mut Default::default())
            .map(|_| ())
            .map_err(|error| format!("{:?}", error.inner));
    };
    let mut store = scope
        .store
        .try_borrow_mut()
        .map_err(|_| "native compilation cache is already in use".to_owned())?;
    let mut usage = scope.usage.get();
    usage.attempts += 1;
    scope.usage.set(usage);
    let mut lookup = Lookup {
        store: &mut *store,
        matched: Cell::new(false),
    };
    let (_, hit) = context
        .compile_with_cache(target, &mut lookup, &mut Default::default())
        .map_err(|error| format!("{error:?}"))?;
    usage.hits += u64::from(hit);
    usage.compilations += u64::from(!hit);
    usage.rejected_entries += u64::from(lookup.matched.get() && !hit);
    scope.usage.set(usage);
    Ok(())
}

#[cfg(test)]
fn define_function(
    module: &mut cranelift_jit::JITModule,
    id: cranelift_module::FuncId,
    context: &mut Context,
) -> Result<(), String> {
    compile_function(context, module.isa())?;
    let code = context.compiled_code().ok_or("compiled code is missing")?;
    let relocations: Vec<_> = code
        .buffer
        .relocs()
        .iter()
        .map(|relocation| ModuleReloc::from_mach_reloc(relocation, &context.func, id))
        .collect();
    module
        .define_function_bytes(
            id,
            code.buffer.alignment.into(),
            code.code_buffer(),
            &relocations,
        )
        .map_err(|error| error.to_string())
}
