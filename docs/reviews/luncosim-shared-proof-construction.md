# Shared-value proof construction

The owner is `rumoca-ir-solve::linear_op`, consumed by shared-value segment
construction and its symbolic equivalence checker. SPEC_0029 §3/§3b permits the
read-only query and requires its single owner; SPEC_0043 §6a requires ordered
output and final-slot equivalence.

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

`shared_values/shape.rs` derives an operation's identity through its existing
typed IR serialization. The proof-local key contains the complete binary
encoding, preserving floating-point bits. Slot and slot-range reads have
distinct typed variants. The shape interner uses hash lookup and compares the
complete key on collisions. IDs follow insertion in program order; hash-table
iteration does not select them. Encoded shapes and their IDs remain local to
the proof table. Encoding failure refuses the symbolic operation through the
existing unevaluable result. The existing workspace `bincode` dependency owns
the encoding; this adds no persistent format or decoder.

## Qualification

Use the production consumer's Rust 1.97.1 with ordinary Cargo targets:

```sh
cargo test -p rumoca-ir-solve --lib -j4 -- --nocapture
```

All 422 tests pass. The constant-bit regression checks destination-independent
sharing, then requires the checker to reject changed finite, signed-zero and
NaN payload bits. The baseline checker accepted a changed NaN payload; the
current checker rejects it. Focused read-set cases cover scalar aliases,
strided and broadcast reads, matrix multiplication, tensor transpose, output
ranges, empty and overflowing ranges, and a nested load outside its required context.
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

The scalar diagnostic derives and checks eight sequences of 256 programs.
It measures 7,067,212 ns in the baseline's focused test run and 6,620,901 ns in
the current full owner suite. The wide diagnostic measures 4,520,923 ns and
5,205,547 ns respectively. The differing suite scopes and concurrent workloads
prevent a causal timing claim from these small differences; production
construction timing remains the performance gate.

The fixed twenty-model Tier 1 canary runs through `cargo xtask verify msl-parity`
with `infra/verification/msl-canary-20.json`. MSL 4.1.0 and OpenModelica
1.27.1~2-g6db4671 produce exactly the same twenty phase/error and channel-band
rows as the published `b9720023` baseline: twenty parse/flatten, fourteen
DAE/solve and twelve initialized, simulated and compared high-band traces. The six DAE and
two solve refusals remain visible. This is a zero-delta canary, not full-cohort
qualification. There are no timeouts, skipped or missing candidate traces.
Raw results remain ignored in the consumer checkout under
`.cache/codex-evidence/performance-20261023/structural-shape-msl-canary/`;
owner tests and diagnostic timings are in `native-cache-audit/` beside it.

The owner-only all-feature Clippy check passes with `--no-deps -D warnings`.
A full workspace lint/cohort sweep is not claimed.
