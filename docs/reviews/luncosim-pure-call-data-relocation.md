# Native pure-call result storage relocation

Status: native owner checks, scoped Clippy and fixed Tier 1 canary pass.
Application admission and production timing are separate qualification gates.

## Owner and invariant

`rumoca-exec-cranelift::emit::typed_program::InputResults` declares one writable,
zero-initialized module data object per primal/directional native helper. Its
checked byte extent and alignment retain the typed cell layout. The existing
owned JIT module holds both code and data until no callable address remains.
`CompiledPureCallTable` explicitly remains non-Sync through its private `Cell`
marker; runtime result cells and evaluator state never enter artifact storage.

`declare_far_data_in_func` consumes the module's standard symbolic data
reference and disables colocation assumptions. The maintained Cranelift cache
and module relocation path bind that reference to the current arena. Native
helpers retain their two-pointer ABI. Complete input-bit comparison, seed
inclusion, miss invalidation, publish-after-success and caller-observed
assertion predicates are unchanged. The retired heap-cell buffers and retained
per-table result-buffer vector are removed, with no second execution path.

This preserves the closed-input/native-owner storage contract recorded in
SPEC_0043 §6a and the target adapter ownership of SPEC_0029/SPEC_0041. It changes
native representation, not IR semantics or numerical policy. No hook, cache-key
normalization, source-language rule, dependency, compiler flag or tolerance is
introduced.

## Reproduction and focused verification

The extended independent-table test holds both compiled tables alive under
one artifact store. Unchanged native code misses both helper entries in its
second construction: zero hits where two are required. The fixed code reuses
all entries without compilation. Interleaved primal and directional calls
retain exact finite/signed-zero bits, seed invalidation and two independent
executions for each new input coordinate. Dropping the first table leaves the
second valid. A changed unary body still misses and computes its own result.

All 114 owner tests pass, including the separately enabled cross-ISA case.
The latter compiles symbolic data-address materialization for x86-64 and
AArch64 and requires full-width `Abs8` relocations. This proves code generation,
not execution on AArch64 or macOS. Existing nested-call, tensor/assertion,
failed-execution, artifact-rejection and lifetime cases also pass. A shared
bounded test store supplies the existing native artifact fixture.

Commands use Rust 1.97.1, the regular target and sccache, with four build, test
and Rayon workers:

```sh
cargo +1.97.1 test --locked -p rumoca-exec-cranelift \
  --features cranelift-codegen/arm64 --lib -j4 -- --include-ignored --nocapture
cargo +1.97.1 clippy --locked -p rumoca-exec-cranelift \
  --features cranelift-codegen/arm64 --all-targets --all-features -j4 --no-deps \
  -- -D warnings
```

Scoped Clippy passes with warnings denied. The cross-ISA test is explicitly
marked with its required dependency feature; the command above executes it,
with zero ignored cases. No native numerical or compiler checks are disabled.
The fixed twenty-model MSL/OpenModelica canary and application evidence remain
separate qualification gates. Raw receipts/logs stay ignored in the consumer's
`.cache/codex-evidence/performance-20261023/rumoca-pure-call-relocation-*`.

## Tier 1 canary

The fixed twenty-model roster uses MSL 4.1.0 and OpenModelica 1.27.1. Twelve
models compare strict-high, including 248 initial-condition channels, with zero
deviations, missing traces, skipped comparisons, non-finite runs or simulation
timeouts. Six models receive ToDAE refusals; two receive EL005 lowering refusals
before solver runtime. All twenty phase/error rows, error-code counts and all
twelve channel-band rows match the retained `d47a5bf3` fixed-canary snapshot.
This also retains the documented predecessor `45d66677` counts. The intermediate
predecessor's raw result directory is absent; no bitwise trajectory delta or
runtime baseline delta for refused models is asserted. The focused run skips
the full-cohort quality gate; no Tier 2/full-cohort parity is claimed.

The exact census delta is ignored under
`rumoca-pure-call-relocation-msl-delta.json` in the consumer evidence directory.
Native raw results are under `target/msl/pure-call-relocation-canary`. Validation
uses the existing standard stage and solver budgets, four workers and one
attempt per case. No oracle, reference, tolerance or exclusion is modified.
