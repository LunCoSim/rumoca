# Shared fixed-output execution

`rumoca-solver::SolveRuntime` owns fixed output preparation through its private
`PreparedOutputSelection`. Prepared root and visible-value plans contain that
executable capability with their semantic metadata. Runtime readers cannot
receive a bare fixed membership or attach a separate visible/root row cache.
Dynamic guarded event/discrete selections retain their own membership semantics
and use the common selected-program/output admission helper.

The existing checked scalar program and standard native backend remain the
execution owners. Complete selected programs retain source order, provenance,
grouped/sparse/duplicate output identity and current state/parameter/time/table
inputs. Exact structural equality permits immutable alternate-chart sharing;
changed selections cannot execute superseded programs through an overlay.
Compilation decline uses the same checked interpreter. Admitted native errors
are terminal. No result, COM term or input-dependent numerical history is cached.
Direct observations and existing expression deduplication are preserved. This is
an engine invariant and hot loop, with no changeable Rhai policy hook.

The concrete regression issues 64 native calls for one fixed visible selection,
returns successful interpreted values after an admitted native error, and leaves
grouped observations on the interpreter. The shared reader issues one batch and
propagates the native error. The root-specific implementation and separate visible
specialization state are removed in the same cutover.

## Verification

- 567 solver tests pass; two manual diagnostics remain ignored. Tests cover fixed
  root/visible call counts, terminal errors, compilation decline, alternate
  replacement, grouped/sparse/duplicate placement, signed zero and buffer bounds.
- Solver and native facade Clippy pass for both library and test targets.
- The fixed 20-model canary preserves all classifications: twelve strict-high
  comparisons, zero missing/skipped comparisons, six ToDAE refusals and two
  structural solve refusals. All 562,411 values and 16,841 timestamps match the
  fresh baseline bit for bit. This is Tier 1, not full 566-model cohort parity.
- Five alternating native executable pairs reduce the visible fixture median
  from 15.572587 to 0.832622 ms (94.65%) for 384 outputs at 512 fresh times. The
  root control has overlapping ranges (0.832–1.395 and 0.839–2.133 ms) and a
  higher candidate median (0.945062 versus 1.046080 ms); no root speedup is claimed.
  Both diagnostics assert every value bit. These are fixture measurements.

Observable consumer behavior belongs to LunCoSim's authored Rhai initialization,
observation, reset and expected-fault gates. Low-level native call counts, bounds
and executable ownership remain asset-free SDK tests. The consumer's current
measurement/review owns production acceptance and FPS claims. Evidence is retained
in the optimization checkout at `.cache/perf/latest-20261105/`, with canaries at
`target/msl/fixed-output-before-20261105` and `target/msl/fixed-output-after-20261105`.
The authoritative execution/lifetime contract is in
[Simulation Runtime](../dev-guide/src/runtime/simulation-runtime.md#fixed-output-execution).
