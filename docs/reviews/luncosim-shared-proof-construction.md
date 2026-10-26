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

Symbolic terms carry the original slot index or the operation ID plus output
ordinal directly. An optional nonzero operation ID keeps those domains disjoint
in two machine words. The allocated operation table bounds its one-based IDs;
the term does not require a separate interning table. This representation is
private to construction/checking and changes no persistent format, executor
operation or floating-point evaluation. The checker still independently walks
the complete original and transformed sequences.

`shared_values/shape.rs` derives an operation's identity through its existing
typed IR serialization. The proof-local key contains the complete binary
encoding, preserving floating-point bits. Slot and slot-range reads have
distinct typed variants. The shape interner uses hash lookup and compares the
complete key on collisions. IDs follow insertion in program order; hash-table
iteration does not select them. Encoded shapes and their IDs remain local to
the proof table. Encoding failure refuses the symbolic operation through the
existing unevaluable result. The existing workspace `bincode` dependency owns
the encoding; this adds no persistent format or decoder.

Private proof lookup tables use the workspace's `rustc-hash` implementation.
Full structural equality still resolves hash collisions. Program traversal
assigns operation IDs; table iteration does not choose execution order,
output order or IDs. The independent symbolic check, register cap, effect
preservation and exact floating-point keys retain their existing owners.

The Griffin production constructor capture attributes 3,463 selected 1 kHz
CPU samples to the 4.369 s coupled-initialization construction interval.
Available frames include shared-value interning and standard SipHash; some
symbols are unavailable. This motivates the private lookup change but does
not itself measure its production benefit.

## Qualification

The direct-term candidate passes 423 active IR tests (one pre-existing ignored
test) and the owner-only Clippy command with `--no-deps -D warnings`, using Rust
1.97.1 and four build/test workers. The dependency-inclusive Clippy command
stops on three existing `rumoca-core` nesting findings; that broader command is
not green. The eighteen focused shared-value cases include changed operand and
output selection, final-slot equivalence, effects, register caps, signed zero
and NaN payload rejection.

Independent baseline/candidate test binaries run ABBA, ten processes per block,
for each existing scalar and wide proof fixture. Scalar median time changes
from 5.280 to 5.065 ms (4.1%); wide median changes from 2.613 to 2.478 ms (5.2%).
All eighty fixture runs pass their derivation/check assertions. These are local
proof measurements, not a claim about end-to-end initialization or simulation.
Binary hashes and complete outputs remain in the consumer checkout's ignored
`.cache/codex-evidence/performance-20261026/term-identity-compact-matched.json`.

The fixed twenty-model canary completes with MSL 4.1.0 and OpenModelica
1.27.1~2-g6db4671: twenty parse/flatten, fourteen DAE/solve, and twelve initialized
and simulated traces. All twelve compare in the high band with accurate initial
conditions; there are no timeouts, skipped comparisons or missing traces.
Six DAE refusals and two solve refusals remain visible. These counts match the
previous published qualification at `5febdef6`; its deleted raw results are not
available for an exact row-by-row comparison. Current raw evidence is ignored
under `target/msl/direct-term-identity-canary`. No full-cohort or workspace
coverage qualification is claimed for this private representation change.

Commands:

```sh
CARGO_BUILD_JOBS=4 RUST_TEST_THREADS=4 cargo +1.97.1 test --locked -p rumoca-ir-solve --lib -j 4
CARGO_BUILD_JOBS=4 RUST_TEST_THREADS=4 cargo +1.97.1 clippy --locked -p rumoca-ir-solve --no-deps -j 4 -- -D warnings
CARGO_BUILD_JOBS=4 RUST_TEST_THREADS=4 RAYON_NUM_THREADS=4 cargo +1.97.1 xtask verify msl-parity --sim-targets-file infra/verification/msl-canary-20.json --results-dir target/msl/direct-term-identity-canary
```
