# Native compilation artifact reuse

An owned LunCoSim Griffin Tracy capture attributes 49.20% of constructor-thread
leaf samples to Cranelift. Repeated constructors emit equivalent native function
stencils while each FMI session owns fresh mutable state. Reusing a live FMI
component would cross that lifetime boundary; retaining relocatable compiler
artifacts addresses the actual repeated compiler work.

`rumoca-exec-cranelift` installs an explicit synchronous thread-local cache scope.
Its existing owned JIT module routes definitions through the maintained
Cranelift incremental-cache API and rebinds module relocations with
`ModuleReloc::from_mach_reloc`. The curated `rumoca-sim` facade exposes the scope
and upstream store trait. No upstream source, Modelica IR, flags or math changes.
The host owns immutable artifact storage, eviction, persistence and rejection
reporting. Scope lifetime does not cross worker threads or async tasks. Solver
instances, numeric workspaces and executable arenas are never cached here.
Pure-call result cells are zero-initialized writable data in each current JIT
module. Functions use full-width symbolic data relocations; cache keys retain
program identity without process-specific storage addresses. The compiled
table preserves its non-Sync boundary and drops its code/data arena last.

Five focused native tests pass: bitwise output equivalence with changed
coordinates/parameters/constants, nested/unwound/thread boundaries, a typed
occupied-store error, shifted module function IDs with changed host addresses,
and malformed artifact rejection followed by correct fresh compilation.

The exact checked Griffin FMI component (v13 source key
`5c7237cb45d93c71`) exposes 6,053 values at initialization and twelve 1/180 s
steps. Uncached, cold, warm and process-restart diagnostics produce identical
f64 bits, digest
`8def1a60ed868e958272691c30a8f0406747498e55e75526b8f88e28836bb409`.
One matched series measures uncached 7,269.063 ms, cold 7,190.815 ms and warm
3,472.063 / 3,236.436 ms. A separate restart series measures uncached
7,219.292 ms and its first persisted-cache construction 3,519.132 ms, with
16,408 hits and 68 fresh compilations. Pure-call storage addresses are bound
through module data relocations; no pointer normalization or cache-key
relaxation is performed.

These are constructor diagnostics, not full-Twin loading or FPS acceptance.
The high-level JIT bytes API does not populate optional unwinder/code-range
metadata; the current dependency has no unwinder feature enabled and Rumoca
does not consume those ranges. A host enabling that feature needs separate
qualification. AArch64 relocation behavior uses the existing emitter, but macOS
has not been executed.

The fixed twenty-model MSL canary has zero model/diagnostic or oracle-band
delta against clean baseline `bda1e01ff223`. Both compile fourteen models,
simulate twelve and compare all twelve strict-high against OpenModelica
1.27.1. The same six ToDae rejections and two solver failures remain visible;
there are no timeouts or non-finite results. This is a focused tripwire, not a
full-cohort parity claim. The two runs use the identical committed target list,
four stage/simulation workers and the standard parent/solver budgets.
The durable delta is `native-cache-msl-canary-delta.json` in the evidence area.

The application passes sixteen cache/codec/admission tests and its production
source-target gate passes twenty checks. The browser worker compiles. A full
Griffin restart consumes 16,408 of 16,476 native artifacts for its large
constructor, with no evictions or rejections. The host's measured working set
exceeds 32 MiB, so its default remains bounded at 64 MiB and 32,768 entries;
evictions and resident bytes are exposed with construction facts. Ready time
falls from 28.642 s to 20.584 s, and that constructor from 8.628 s to 4.450 s.
Source hashes, readiness, touchdown, retained diagnostics and screenshots are
checked through owned sessions. Physics remains around 60 ms, frames around
76 ms; the whole-task performance targets are not met. Prepared-FMI disk reuse
is correctly disabled for this dirty local compiler build, so combined-cache
clean-identity qualification remains open.

The emitter passes Clippy with lib/tests and the facade passes lib Clippy,
both scoped with `--no-deps -D warnings`. Facade test Clippy stops at an
unchanged unnecessary cast in `solve_lowering/tests.rs:1419`; dependency-inclusive
Clippy also encounters existing core nesting. No unrelated lint is suppressed.
The required notices are regenerated with cargo-about 0.9.2 and the existing
accepted-license policy. Its preserved license text includes CRLF, so the
notice writer now emits canonical LF before Git normalization and check;
its focused write/check/negative regression passes. No license policy changes.

Evidence resides in LunCoSim optimization's
`.cache/codex-evidence/performance-20261020/native-cache-*`, including source
probes, full sampled states, comparison JSON and cross-process logs.
