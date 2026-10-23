# Shared-value proof construction

The owner is `rumoca-ir-solve::linear_op`, consumed by shared-value segment
construction and its symbolic equivalence checker. SPEC_0029 §3/§3b permits the
read-only query and requires its single owner; SPEC_0043 §6a requires ordered
output and final-slot equivalence. No Modelica semantics or wire schema changes.

An isolated operation's reads are collected during one traversal of the same
exhaustive structural operand checks used by whole-program dominance validation.
The initialized-register context still rejects every source without a preceding
write. The collection context records exact operands without an allocation
indexed by the absolute register number. It returns sorted, unique register IDs.
Range shape, overflow, projection and enclosing-context failures still reject
the query. Shared-value output/final-slot checks retain their symbolic owner.

The term table interns each complete operation key (shape and ordered operand
terms) once. A scalar output term holds that table-local value ID and its output
ordinal. Hash collisions still compare the complete structural key. The builder
retains its table across segment boundaries, including capped and opaque
segments; the checker evaluates both sequences in one independent table. IDs
never leave their owning derivation/check. Register normalization uses one
operation clone, with a read-only discovery pass before mutation.

## Qualification

Use the production consumer's Rust 1.97.1 with ordinary Cargo targets:

```sh
cargo test -p rumoca-ir-solve --lib -j4 -- --nocapture
```

All 420 tests pass. Focused read-set cases cover scalar aliases, strided and
broadcast reads, matrix multiplication, tensor transpose, output ranges, empty
and overflowing ranges, and a nested load outside its required context.
The ordinary `shared_read_collection_range_benchmark` verifies 64 operations
with 1,024 reads each: the local diagnostic measures 36,838,415 ns before and
227,143 ns after. This is an isolated constructor-path measurement, not a
whole-simulation speedup.

The ordinary `shared_wide_value_proof_benchmark` derives and checks eight pairs
of 16×16 matrix programs, each with 512 inputs and 256 outputs. It measures
37,767,766 ns before operation-key interning and 6,113,756 ns after. The paired
negative test changes operand order and output selection separately; both are
rejected. Existing tests retain cap boundaries, effects and final-slot checks.
These measurements qualify isolated proof construction, not scene loading.

The fixed twenty-model Tier 1 canary runs through `cargo xtask verify msl-parity`
with `infra/verification/msl-canary-20.json`. MSL 4.1.0 and OpenModelica
1.27.1~2-g6db4671 produce exactly the same twenty band/diagnostic rows as the
native-factory qualification: twenty parse/flatten, fourteen DAE/solve and
twelve initialized, simulated and compared high-band traces. The six DAE and
two solve refusals remain visible. This is a zero-delta canary, not full-cohort
qualification. The operation-key change also preserves all twenty band and
diagnostic rows against the single-pass source-discovery baseline. Raw results
remain ignored in the consumer checkout under
`.cache/codex-evidence/performance-20261023/shared-{proof,value-key}-*`.

The owner-only all-feature Clippy check passes with `--no-deps -D warnings`.
The dependency-inclusive invocation stops at three pre-existing excessive-
nesting diagnostics in `rumoca-core/src/lib.rs` lines 291–295; that file is
unchanged. A full workspace lint/cohort sweep is not claimed.
