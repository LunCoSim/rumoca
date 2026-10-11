# LunCoSim runtime owner verification

Based on official v0.10.2, commit 276e82bd0fd48960ac201e782918d28bf19cf7ce.
The additional host capabilities keep checked FMI construction and ME session
execution authoritative: immutable prepared-component borrowing, canonical
construction-wire persistence, exact compiler source contributions, paired
source/syntax bundle records, cooperative cancellation and retained-output
admission. Optional timing uses rumoca-core's instrumentation owner.

Focused checks used nightly-2026-02-27, four Cargo jobs, four test threads and
four Rayon threads:

- `cargo test -p rumoca-eval-solve -p rumoca-solver --lib`: 240 evaluator and
  550 solver tests passed.
- `cargo test -p rumoca-compile --lib successful_strict_compile_carries_only_reachable_source_contributions`:
  passed; unrelated document bytes are excluded.
- `cargo test -p rumoca-solver --lib admitted_output_count_matches_exact_grid_and_rejects_invalid_requests`:
  passed after restoring fallible output-vector allocation.
- LunCoSim's checked-component persistence tests passed, including malformed
  construction-wire rejection, bounded decoding and directory-quota publication.

## Fixed canary delta

Both unmodified v0.10.2 and these owner changes ran:
`cargo xtask verify msl-parity --sim-targets-file infra/verification/msl-canary-20.json`.
MSL was the official v4.1.0 release; the comparator was
OpenModelica 1.27.1~2-g6db4671. Each run used the same fixed twenty targets.

| Observed boundary | Official v0.10.2 | Host capability changes |
|---|---:|---:|
| Parsed and flattened targets | 20 | 20 |
| Checked DAE targets | 14 | 14 |
| Simulated targets | 12 | 12 |
| Initialization successes | 12 | 12 |
| Trace comparator strict-high target identities | 12 | 12 |
| Additional unexcepted non-high targets | 0 | 0 |

The successful target identities were identical. Six targets failed before
checked DAE construction. Two more reported typed solve-IR failures:
`Modelica.Electrical.Analog.Examples.OpAmps.OpAmpCircuits.Add` was structurally
singular and `Modelica.Electrical.Polyphase.Examples.TransformerYD` had a
non-numeric affine derivative coefficient. No new failure was introduced.
This is a focused canary delta, not a full-cohort parity measurement.

## Static variability inside input components

Checked DAE role construction gives constant and parameter variability precedence
over inherited input causality. A nested parameter remains a static coordinate;
its enclosing input component cannot turn it into an external runtime slot.
The focused checked-preparation test evaluates a declaration default depending
on that nested parameter, a function default argument, and an array input.

After the static-variability correction, the same fixed canary again parsed all
2555 sources, compiled fourteen targets and initialized/simulated twelve. The
compared model identities and agreement bands remained identical to official
v0.10.2. The only feature-specific build warning was corrected by gating the ME
artifact import on its actual RK45 consumer.

## Source class query ownership

`rumoca-core` owns quote/bracket-aware path boundaries.
`StoredDefinition` owns exact read and mutable class traversal and normalization
of its authored `within` package. `Session::class_definition_query` borrows the
current indexed declaration; `class_component_declarations_query` returns a
complete scope-resolved member set with declaring identity and typed source
facts. Unresolved or cyclic inheritance has no complete result. Primitive
bases have no named component members.

The focused core boundary tests passed (2), and the session class query tests
passed (4). The existing membership/cache tests also passed (3) using the
same built test binary. Coverage includes quoted identities, exact package boundaries, source edits,
document removal, pre-parsed library replacement, inherited typed declarations,
primitive-derived connectors, missing bases, and inheritance cycles.
Commands used `--locked`, four Cargo jobs, four test threads and four Rayon
threads:

- `cargo test -p rumoca-core --lib qualified_name_boundary_tests -j4`
- `cargo test -p rumoca-compile --lib class_definition_query -j4`

The final same-target canary command was
`cargo run -p xtask --bin xtask -j4 -- verify msl-parity --sim-targets-file infra/verification/msl-canary-20.json --results-dir target/msl/class-lookup-current`.
The baseline was `87df7f6bcaafc4aace0e875497bb731f8f50f220`;
the fixed target file SHA-256 was
`536004933ce502f7a8a4317e5dd6de2b700a6b176e84d2d6e7bbe0dcdf61eafa`.
Both used MSL v4.1.0 and OpenModelica 1.27.1~2-g6db4671.

| Observed boundary | Baseline | Current source queries |
|---|---:|---:|
| Parsed and flattened targets | 20 | 20 |
| Checked DAE / solve IR targets | 14 | 14 |
| Initialized and simulated targets | 12 | 12 |
| Strict-high trace target identities | 12 | 12 |
| Refused before simulation | 6 | 6 |
| Typed solver failures | 2 | 2 |

All twenty per-model band rows were identical. No additional failure or weaker
trace band appeared. This remains a focused canary, not a full-cohort claim.
The cached OMC reference run did not reproduce its twelve initialization XML
files in the new results directory, so the current harness reported zero
state-selection comparisons. Direct comparison of the twelve before/after
Rumoca trace state metadata sets found them all unchanged. Timings came from
concurrent builds and the live application; they do not establish a performance
improvement.
