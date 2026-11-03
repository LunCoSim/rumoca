# Checked initialization preparation

The consumer is LunCoSim's generation-scoped coupled-initialization preparation
job. It needs settled observations and accepted free continuous coordinates
without creating a numerical integrator or exposing a mutable runtime kernel.
Rumoca owns that initialization and the checked state-coordinate admission.

MLS §8.6 supplies the initialization boundary. SPEC_0007 and SPEC_0036 keep the
DAE/Solve/FMI ownership and checked aggregate constructors authoritative;
SPEC_0043 catalogs the external free-coordinate constraint. SPEC_0029 keeps
scoped declaration/type queries and numerical evaluation at their existing
owners. SPEC_0008 retains original diagnostic source maps. No source equation,
projection, residual, or update writer is replaced by an initialization guess.

The public initialization-only operation uses the same initialized host as live
construction, requests checked finite outputs in caller order, and stops before
integrator creation. Selective output access maps each logical scalar to its
owning program; it does not evaluate unrelated output programs.

A free-state request validates exact source identity, finite value, domain and
uniqueness. A request identical to checked `initial_y` retains the immutable
component and every initialization/event owner. A changed supported coordinate
gets given ownership through the existing checked initialization and FMI
constructors. Changed hybrid/history/clock/chart/initial-parameter-solve state is
explicitly refused. The consumer verifies all accepted source-state observations
after initialization/events; identity normalization is not permission to accept a
different resulting state. Cached prepared components stay unchanged.

Scoped parameter queries reuse maintained segmented-name, inheritance, type and
variability resolution. Binding-only attribute evaluation uses admitted parameter
bindings, never initialization start guesses. Structural-override provenance uses
the current source map even when resolution planning is reused. Detailed DAE
failure returns its original source map, allowing located diagnostics without a
second strict compile.

Focused evidence in the consumer checkout's
`.cache/codex-evidence/modelica-initialization-review-20261103/` includes two
initialization-only tests, selective-output program mapping, seven free-state
constraint tests, scoped reference queries, structural-override provenance and
binding-only evaluation. Consumer compiler/cache, lifecycle, scene and live Twin
tests exercise the public APIs. Griffin reaches readiness with a certified
initial solve and no initializer telemetry channels.

The fixed 20-model canary ran with MSL 4.1.0 and OpenModelica
1.27.1~2-g6db4671 using `CARGO_BUILD_JOBS=4 RUST_TEST_THREADS=4
RAYON_NUM_THREADS=4 cargo xtask verify msl-parity --sim-targets-file
infra/verification/msl-canary-20.json`. It exits 0: 20 parse/flatten, 14 DAE/Solve,
12 initialized/simulated traces, all 12 compared in the high agreement band;
zero skipped/missing traces and zero deviations. Six ToDae and two Solve
refusals remain visible. These aggregate counts match the documented historical
fixed-roster result in `luncosim-output-program-index.md` and
`native-compilation-cache-2026-10-21.md`; historical raw results were unavailable
for a new per-model diagnostic delta. Current raw results, refusal census,
roster digest and the explicitly aggregate delta are retained in
`rumoca-canary-evidence/`. This is partial Tier 1 evidence, not a 566-model
cohort or full-workspace claim.

The combined revision `64d99750f850ed1134588ad0ea1fcfd8728d28f4` also includes
the native torn-tangent branch and all seven newer main commits. All local and
fetched origin branch commits dated October 2026 or later are ancestors of that
revision. Older April–June branches remain separate. The combined tangent
filter passes 12 tests across evaluator, native execution and solver owners;
all seven initial-state tests pass. Its repeated fixed canary compares the same
12 traces with high agreement and no deviations.

The full pre-landing gate passed formatting and entered Clippy, then was stopped:
its runner hardcodes eight jobs and documents approximately 50 GB for coverage,
exceeding the consumer's four-job limit and available disk space. No full
Clippy/coverage/workspace, full MSL, ModelicaTest or pinned external corpus pass
is claimed. These remain landing gates under SPEC_0025 unless explicitly waived
by the integrating maintainer.

Performance scope is explicit: initialization-only avoids the integration plugin
and consumer metadata/telemetry publication, but `SolveRuntime::construct` still
prepares native dynamics, derivatives and event programs eagerly. Selective
observation does not remove that earlier work. Consumer checked-DAE sharing is
verified separately from native JIT reuse. The observed native cache reached its
32,768-entry bound below its 64 MiB byte bound and evicted programs; no overall
startup speedup or benefit from unimplemented lazy preparation is claimed.
