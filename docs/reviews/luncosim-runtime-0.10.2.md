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
