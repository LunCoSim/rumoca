# Fixed root-condition execution batches

The authoritative mechanism is documented in
[Simulation Runtime](../dev-guide/src/runtime/simulation-runtime.md#root-condition-execution).
`rumoca-solver::SolveRuntime` owns construction and lifetime;
`write_planned_root_conditions` and `write_planned_root_search_conditions` are
its production readers. The standard execution backend owns native kernel
chunking. No application adapter, new dependency or Modelica rule is added.

A fresh LunCoSim capture identified selected-expression/native dispatch as a
material integration cost. Root membership was already fixed by its root plan,
but numerical reads traversed individual specialization caches and crossed the
native expression boundary per selected program. A 64-root regression records
64 native calls and demonstrates that an admitted native error was hidden by
interpreter retry. The current contract issues one batch dispatch for that
selection and propagates execution failure.

The owner prepares checked public/search projections once. Complete selected
programs retain their operations, order and source provenance; a private output
map preserves grouped, sparse and repeated identities. The retired root-row
specialization cache, refusal set and dispatcher are removed. Other scalar-row
owners retain their own contracts. Exact immutable code can be shared by
identical selections and alternate charts; changed selections execute only their
own programs. Runtime scratch is refilled at each coordinate. There is no value
cache, parameter-static reclassification or COM assumption.

This is a continuous-engine execution mechanism. Root membership and indicator
semantics remain with their existing owner, so no changeable Rhai hook applies.
The governing boundaries are SPEC_0029's solver/backend separation, SPEC_0007's
checked executable IR and SPEC_0044's common ME/root and bounded-buffer contract.
No root search, projection, initialization, precision or integration rule changes.

## Evidence

The solver lib passes 562 tests with two manual diagnostics ignored. New tests
cover batch cardinality, terminal native errors, grouped/sparse/duplicate output
ownership, exact f64 and signed-zero bits, skipped source programs, missing
owners and short destinations. Existing root, chart-sharing, event and lifecycle
tests pass. Clippy reports only three unchanged excessive-nesting sites in
`relation_memory.rs` and `tangent_evaluators.rs`; no lint suppression is added.

Five alternating native executable pairs over 384 roots and 512 fresh-time
reads reduce the diagnostic median from 11.938011 to 0.822214 ms (93.11%). Every
output bit is asserted. This is a root-selection fixture, not whole-worker or
FPS performance.

The fixed 20-model canary retains all classifications: twelve completed traces,
six ToDAE refusals and two solve-IR refusals. All twelve comparisons remain
strict-high; there are no missing/skipped comparisons. All 562,411 data values
and 16,841 timestamps match the admitted baseline bit for bit. This is Tier 1,
not full 566-model cohort parity. Results are
`target/msl/root-batch-after-20261105` against `target/msl/main-sync-tangent-after`.

The production consumer passes 264 authored checks and two dedicated expected-
fault scenes. Source-matched production headless runs retain thirteen participants
and complete physics/handoff admission. Changing external compiler/memory load
prevents a stable whole-Modelica timing claim; a further source-changing run is
rejected. See LunCoSim's
`docs/reviews/modelica-root-batches-2026-11-05.md` for measured worker costs,
concurrent-load limits and the excluded graphics run. Detailed source, executable,
benchmark, canary and API receipts are retained in the optimization checkout at
`.cache/perf/modelica-current-20261105/`. Changes remain local and unpublished.
