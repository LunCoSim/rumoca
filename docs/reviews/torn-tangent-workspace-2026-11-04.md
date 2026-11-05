# Torn tangent evaluation workspace

The canonical owner contract is in
`docs/dev-guide/src/runtime/simulation-runtime.md`. `TornTangentPreparation`
binds one immutable JVP source and prepares the solver's block evaluators with
shared directional preparation and one retained evaluation workspace. Checked-out
buffers allow nested callbacks without borrowing or overwriting active seeds.
The largest allocated seed storage remains available across lane shapes, while
every callback receives its exact current logical extent. Previous-plan write
ranges are cleared before registering the next plan, including interrupted
sweeps. No result, coefficient, numerical coordinate or evaluation decision is
reused. This is continuous engine math and temporary storage, with no changeable
Rhai policy or hook decision.

The retained post-shared-label Tracy capture identifies 117 integration samples
at libc offset `0x1a1efa` (`rep stos`) and smaller copy samples at `0x1a197f`
(`rep movsb`). The loaded inode matches the inspected library and its change
metadata predates the capture. Caller stacks are incomplete, so this identifies
bulk fills/copies, not their owning source. The tangent evaluator's source
independently establishes repeated dense zero-filled seed allocation as a
candidate. No libc, Cranelift dependency source or platform settings are changed.

The existing native tangent fixture supplies a compiled-kernel diagnostic at
3,035 coordinates and 8,192 complete evaluations per sample. Five alternating
independent executable pairs reduce multi-lane median time from 17.9529 to
11.4596 ms (36.17%) and directional time from 13.2180 to 10.0704 ms (23.81%).
These are fixture timings, not a whole-worker or FPS claim. All five native
boundary tests pass: exact bits and invocation count, terminal native errors,
recovery after partial execution and non-finite coefficients, varying coordinate
lengths, reentry, shared lane shapes and typed short-input rejection. Unused seed
coordinates retain positive zero, and callback lengths match their logical shape.

Evaluator/backend lib/tests Clippy passes with warnings denied. The fixed
20-model canary retains twelve strict-high comparisons, zero missing/skipped
traces and eight unchanged typed refusals. All 562,411 values and 16,841 timestamps
match the retained pre-change traces bit-for-bit. This is Tier 1, not full-cohort
parity. The production candidate passes 111 authored observation, clock/receipt
and plume checks plus two expected-fault scenes. Normal binary SHA-256 is
`d8c7002761ba8dd181674677571a606ba3b1ef96e1cdd9c3aa65a5d7091649a6`.

The normal High/2K baseline/candidate/baseline bracket gives MainPropulsion mean
service 2.0242 / 1.8018 / 2.2446 ms, 10.99–19.73% lower against both references.
Aggregate Modelica work is 3.0479 / 2.8931 / 3.1588 ms per physics tick across
thirteen participants, 5.08–8.41% lower. Candidate warm largest-participant mean
service has median 1.4350 ms. No compiler overlap is observed in guarded windows.
FPS is 50.53 / 49.19 / 47.68; drifting references establish no whole-scene gain.
Candidate process peak RSS is 4.014 GiB versus reference 4.011/4.041 GiB; this
records no large measured retention increase in this cohort, not a general memory
bound or a GPU-memory measurement. Source, complete variables, terrain, handoff,
physics and diagnostics gates pass; the final native image is inspected. All
owned apps/ports close and the canonical application lock is restored. The
150 FPS and 1–2 ms aggregate-work targets remain unmet. Full solver Clippy still
reports only three preexisting nesting sites, without a new lint or suppression.

Evidence is under optimization's
`.cache/perf/modelica-tangent-workspace-20261104/`; current canary traces are
`target/msl/tangent-workspace-after` against
`target/msl/selected-output-workspace-after`. SDK/application changes remain
local and unpublished.
