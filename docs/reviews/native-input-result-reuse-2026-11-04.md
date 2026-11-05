# Bounded reuse of complete native function results

The Griffin plume computes five streamtube stations through
`computeFreeJetProfile`, each querying `supersonicExitMach`. Nozzle thermal
zones also query that function. Repeated interleaved coordinates need retained
results beyond the latest call. An owned standard-release Tracy capture locates
1,734 of 4,344 slow-lane integration samples in one native pure-function call
subtree. Native disassembly identifies its station calculations and the nested
Mach inversion; the latter has 504 leaf samples in its body. These are sampled
diagnostic counts, with incomplete JIT caller unwinding, not product timings.

The existing `emit::typed_program::InputResults` is the authoritative owner.
Its closed `SolvePureCallInputCoordinate` includes every input cell at the
owner's fixed arithmetic profile; directional coordinates include tangent
cells. The adapter now retains power-of-two direct-mapped entries for small
coordinates. It hashes all input bits to select an entry, then compares the
complete coordinate before reusing the whole result tuple, including assertion
predicates and message values. Collisions replace entries without authorizing
another input's output. A selected entry is invalidated before body evaluation
and published only after success.

Coordinates of one to sixteen cells use at most 64 entries within a 16 KiB
storage target. Constant or larger input coordinates retain one complete entry;
a large tuple's single payload can exceed that target. The emitted hash work
is bounded. Byte extents remain checked, and the selected byte offset is within
the module data object's extent on both pointer widths. Module-owned writable
storage, symbolic relocation, non-Sync table ownership, independent primal and
directional storage, and fresh data on compiled-artifact replay remain intact.
No new public API, dependency, model-name branch or parallel cache owner is
introduced. This is numerical execution work, not scenario policy or a hook
candidate. No equation, arithmetic operation, precision, tolerance or step
cadence changes.

The focused interleaving regression fails before the change: four native
evaluations occur for two alternating input tuples. It passes with two after
the change. Nine focused native tests pass, covering replacement collisions,
complete tensor inputs, assertion predicates, seeds, failed execution, nested
calls, independent modules and compiled-artifact identity. Each failed input
must execute and report its failure again. Owner lib/tests Clippy passes with
`--no-deps -D warnings`.

The production standard-release comparison uses unchanged Griffin source and
High/2560×1440 rendering:

| Cohort | Mean worker service | Last worker service | Maximum | Transactions | FPS |
|---|---:|---:|---:|---:|---:|
| Candidate 1 | 2.5190 ms | 2.1944 ms | 32.0503 ms | 2,297 | 48.36 |
| Fresh baseline | 3.6505 ms | 3.2181 ms | 9.7783 ms | 2,287 | 48.28 |
| Candidate 2 | 2.3322 ms | 1.9017 ms | 9.2978 ms | 2,279 | 48.51 |

These are the exact `/Griffin1SurfaceOps/Lander/MainPropulsion` participant's
worker receipts, not response latency or total coupled-step wall time. The mean
service reduction is 31.0–36.1%. All three runs complete landing handoff,
qualify terrain and advancing physics, retain all 3,035 variables without a
Modelica error, and preserve all 528 model hashes. No Cargo build overlaps the
performance windows. Every owned app exits and releases its API port. Scene
FPS is unchanged within variation. The 32 ms maximum is not reproduced and
remains unexplained; these runs do not establish a tail-latency improvement.
The requested 1–2 ms mean worker and 150 FPS targets remain unmet.

A follow-up composition check removes only scalar power reuse while retaining
the complete-function cache. MainPropulsion averages 2.3714 ms without it
(API 5588, 2,281 transactions), versus 2.4336 ms in a fresh qualified repeat
with both caches (API 5589, 2,269 transactions). Both are within the existing
repeat range. This does not establish a benefit from removing scalar reuse;
the qualified implementation and normal binary above remain selected. Model
hashes are unchanged and both processes and ports are released. Evidence is in
optimization's `.cache/perf/modelica-cache-composition-20261104/`.

The production plume gate passes 52 conservation/optical assertions, then five
invalid-gas assertions in the same owned session: invalid gamma faults at the
Modelica owner and retains the last transported sample without crashing. The
47-check initialization/clock/receipt gate and terminal initialization-fault
gate pass separately. The fixed MSL 4.1.0/OpenModelica 1.27.1 canary compiles
fourteen of twenty targets and compares the same twelve simulated models
strictly high against the baseline, with no deviations, missing traces or
skipped comparisons. All twenty owner classifications/errors match; six ToDAE
and two typed solve-IR refusals remain visible. All 562,411 trace data values and
16,841 timestamps match bit-for-bit. This is a fixed canary delta, not
full-cohort parity or exact complete-Griffin replay.

Baseline binary SHA-256:
`6f39553f2e955105e81e27e165a41a338b4c1755d9bba698ea97171a457fe684`.
Candidate SHA-256:
`50868737c46d4b573461dbd8a356130b80b2050042796c8e78c978e040108020`.
Evidence is retained in optimization's ignored
`.cache/perf/modelica-jit-critical-path-20261104/`, including the failing/fixed
tests, build provenance, normal receipts, numerical delta and native capture.
Native canary results are in `target/msl/input-results-canary`, compared with
the unchanged `target/msl/power-cache-canary` baseline.

The matching Tracy 0.14.1 capture validates native export and framing:
92,198,423 file bytes comprise 92,178,481 payload bytes, a six-byte header and
4,984 four-byte block lengths. Sampling is verified at 1 ms. The maintained
Cranelift perf-map facility initially emits interleaved rows under concurrent
construction. A diagnostic file-open interposer serializes only the owned
`/tmp/perf-<PID>.map` writes; a two-thread probe validates 1,000 complete rows.
The corrected run has 5,683 valid map records and no malformed rows. Current
read-only executable mappings bound function attribution; matching sampled
addresses have no ambiguous labels. The helper and code snapshots remain
ignored diagnostic evidence and are absent from normal product measurements.

A separate post-change standard-release Tracy capture (API 5590/PID 394340)
uses the same 1 ms sampling period and bounded diagnostic settings. It admits
5,683 valid JIT map records, with 3,247 function records in current
executable mappings. Nineteen adjacent range pairs overlap after code-arena
reuse; none of the selected integration sample addresses has multiple labels. Native CSV/sample export succeeds; framing accounts
for all 83,883,041 bytes (83,865,063 payload bytes, six header bytes, 4,493
four-byte block lengths). The 20–40 second count-10 MainPropulsion cohort has
1,157 transactions: mean instrumented integration is 2.4516 ms, observation
construction 0.7519 ms, and complete worker service 3.4291 ms. These are
diagnostic timings, distinct from the normal worker receipts above.

The slow lane has 2,849 integration samples. No sampled PC falls in the current
Mach inversion helper; the free-jet helper retains 172 inclusive samples.
Its code contains the source's Gauss node constant, and the Mach helper retains
the two-input/48-bisection body. This supports the intended cache effect; zero
samples does not prove zero evaluations. Native Jacobian/assignment execution
is also visible. Observation construction remains measurable; unresolved
stripped static symbols prevent a precise allocation-versus-equation split.
No additional optimization is selected from that attribution alone. The
capture, current executable code, source-preservation receipt and qualification
are retained in optimization's ignored
`.cache/perf/modelica-cache-composition-20261104/post-cache-tracy/`. Its app exits
and releases the port, and the qualified normal executable is restored.
