# Selected-output execution workspace

`SolveRuntime::eval_selected_outputs_with_native` groups a request by checked
program owner and copies each program's selected outputs in request order. Its
request records now reuse runtime-owned storage, validated before evaluation and
sorted by `(program, request ordinal)`. Duplicate requests retain their position.
Both compiled and interpreter copying consume the same selected iterator. The
old per-call grouping implementation is removed; no numerical state, coordinate,
specialization decision, guard, diagnostic or output value is reused by this
workspace. It is engine execution work, so no changeable Rhai policy is involved.

The runtime owns its allocation, refills it before every read, and releases it
on teardown. A runtime clone owns independent storage. Failed selection admission
cannot execute a partial request; subsequent requests replace those records.
Capacity follows the largest admitted request. The owning current contract is
in `docs/dev-guide/src/runtime/simulation-runtime.md`.

The post-shared-label LunCoSim Tracy capture identifies 34 selected-output
function leaves and 27 tree-entry insertion leaves among 1,573 integration
samples. Allocator caller recovery is incomplete, so it does not attribute all
221 allocator leaves to grouping. This is the concrete profiling basis for the
fixture and the localized execution change.

The inline 512-program/1,024-value fixture tests complete selected execution
through the actual owner, including duplicate rows and signed zero. It alternates
five independent optimized executable pairs. Ordered median time for 512 reads
is 28.8937 → 19.5102 ms (32.48% lower); reversed requests are 27.4929 → 21.0161 ms
(23.56% lower). These are fixture results, not full-worker/FPS numbers. All sixteen
visibility-module tests pass, including invalid selection, copy order, sparse
requests and allocation reuse. The fixed twenty-model canary retains twelve
strict-high comparisons, zero missing/skipped traces and eight unchanged typed
refusals. All 562,411 values and 16,841 timestamps match its retained pre-change
traces bit-for-bit. This is Tier 1 evidence, not full-cohort parity.

Solver lib/tests Clippy reports only three existing excessive-nesting sites in
unchanged `relation_memory.rs` and `tangent_evaluators.rs`, once per target; no
new diagnostic is introduced or suppressed. The full Clippy gate remains red.
The production binary `08ce42b0ce88cfd075af5d38ab44f459ede899f1997220ba18be85a558b93be3` passes 111 authored observation, clock/receipt and plume checks plus two expected-fault scenes. Its normal baseline/candidate/baseline Griffin bracket reduces MainPropulsion
mean worker service from 2.3803/2.4165 to 2.2384 ms (5.96–7.37%) and aggregate
worker work from 3.6334/3.6429 to 3.5216 ms per physics tick (3.08–3.33%). No
compiler overlap is observed in guarded measurement windows. Drifting FPS
references (49.55/45.81 versus candidate 46.91) establish no whole-scene gain,
and the candidate maximum does not establish a tail improvement. Source,
terrain, handoff, advancing physics, complete variables and diagnostics gates
pass. All owned processes/ports close; application quality settings stay High/2K. Application evidence lives under
`optimization/.cache/perf/modelica-selected-output-workspace-20261104/`; canary
traces are `target/msl/selected-output-workspace-after` against the unchanged
`target/msl/shared-label-after` baseline. Local SDK/consumer overrides are
explicit; nothing is published.
