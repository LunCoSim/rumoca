# Exact native power reuse

LunCoSim's owned standard-release Griffin capture attributes 2,079 of 5,170
slow-lane integration samples to native math. A separate bounded argument
diagnostic records two 200,000-call power streams with only 1,913 and 3,201
distinct f64 input pairs. A conceptual 1,024-entry LRU can reuse 99.04% and
98.40%; that calculation is diagnostic, not the implemented cache's hit rate.
The argument diagnostic overlaps a build and supplies no FPS evidence.

`rumoca-exec-cranelift::emit::host_runtime::power_cache` owns exact finite
power reuse. Its only production reader is the existing native `pow` callback.
A 4,096-entry direct-mapped table occupies 128 KiB per thread. Full input bits
authorize a hit; hash collisions cannot return another pair's result. Non-finite
inputs/results retain the normal math path. No Modelica IR, equations, solver
tolerance, precision, step lattice, finite check or numerical history changes.
The table is thread-local, bounded, non-persistent, and independent of component
and scene lifetimes. It retains no instance facts or worker completion order.

DRY review checks the existing typed `InputResults` storage first. That owner
already reuses complete Modelica pure-function calls at their construction-issued
input coordinates. Scalar `pow` imports have a separate native math ABI, so
they cannot directly consume that owner without manufacturing function IR.
The new table remains private to that math import and adds no public API or
dependency. This is generic numerical hot-path work, not changeable scenario
policy or a hook candidate.

Two focused native tests pass for finite values, signed zero, subnormals,
collision replacement and exceptional results. Owner lib/tests Clippy passes
with `--no-deps -D warnings`. A standalone diagnostic imports the actual source
module and checks complete output bits against ordinary `powf` before timing
eight alternating pairs. Median throughput is approximately 2.84 times the
uncached recorded stream; this is a math-call diagnostic, not solver speedup.

The production standard-release comparison uses unchanged model source and
High/2560×1440 rendering:

| Run | MainPropulsion mean worker service | Transactions | Scene FPS |
|---|---:|---:|---:|
| Baseline A1 | 4.3046 ms | 2,268 | 48.11 |
| Candidate B | 3.8224 ms | 2,249 | 48.66 |
| Baseline A2 | 4.1393 ms | 2,260 | 48.68 |

The source change reduces mean worker service by 7.7–11.2% against these two
baselines. Whole-scene FPS is unchanged within run variation. The requested
1–2 ms worker and 150 FPS targets remain unmet. Readiness, touchdown handoff,
terrain, advancing physics and retained diagnostics qualify; all 3,035
MainPropulsion variables remain present and all 528 model hashes are unchanged.
No Cargo build overlaps these three performance windows. The candidate image
is inspected. Every owned application exits and releases its API port.

Both binaries pass the 17-check production radiative-exchange gate, including
invalid-emissivity rejection. All 400 observable values and 100 timestamps
across cooling, heating, equilibrium and mirror trajectories match bit-for-bit.
The candidate also passes 47 production initialization/clock/receipt checks and
the terminal initialization-fault gate. These are focused numerical and runtime
checks, not full-MSL parity or exact complete-Griffin replay.

The fixed twenty-model MSL canary runs separately with and without the native
power cache, using MSL 4.1.0 and OpenModelica 1.27.1~2-g6db4671. Both compile
fourteen models, simulate twelve and compare the same twelve models strictly
high, with zero deviations, missing traces or skipped comparisons. Six ToDAE
refusals and two typed solve-IR refusals remain unchanged. All twenty owner
classifications/errors match. The twelve Rumoca traces also retain identical
bits across **562,411 data values and 16,841 timestamps**. This is a fixed
canary delta, not a full-cohort parity claim. Its durable record is
`pow-cache-candidate/msl-canary-delta.json` in the evidence directory below;
the two native result trees are `target/msl/power-cache-{baseline,canary}`.
The baseline temporarily restores only the existing math callback and module
registration, then verifies and restores the candidate source exactly.

Application baseline SHA-256:
`a763f890ece1395d5fa883f8e7250bbb24827948ed3eaaf7350a5189172158aa`.
Candidate SHA-256:
`6f39553f2e955105e81e27e165a41a338b4c1755d9bba698ea97171a457fe684`.
Evidence is retained in LunCoSim optimization's ignored
`.cache/perf/modelica-critical-path-20261104/pow-cache-candidate/` and
`pow-probe/corrected/`. The application uses an explicit ignored local source
override; canonical dependency pins remain unchanged. No publication is made.
