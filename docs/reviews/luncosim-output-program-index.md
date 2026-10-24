# Immutable scalar output ownership index

`ScalarProgramBlock` owns the checked program/output pairing. Its private
immutable data now indexes each logical output to its first producing program.
Construction uses the same output-binding iterator as existing readers. Sparse
indices, including `usize::MAX`, consume storage proportional to the admitted
output count. Repeated logical indices retain their first stored occurrence.
Clones share the complete immutable owner.

The index is private and never enumerated to select execution, output order,
serialization or IDs. It uses the existing workspace `rustc-hash` dependency.
The checked constructor derives it on wire replay; the three-field wire remains
unchanged. Stored-output count reads the already-validated output identity
length. There is no new public API, serialized field, evaluator, numerical
change, alternate lookup path or compatibility contract.

SPEC_0029 section 3 permits the read-only IR query; SPEC_0036 and SPEC_0043 retain
constructor ownership and checked wire replay. SPEC_0021's private lookup rule
permits the non-enumerated hash index. The live output bindings remain the
authority for programs, offsets, output counts and logical identities.

## Qualification

Use the production consumer's Rust 1.97.1, with `CARGO_BUILD_JOBS=4`,
`RUST_TEST_THREADS=4`, `RAYON_NUM_THREADS=4` and normal sccache:

```sh
cargo +1.97.1 test --locked -p rumoca-ir-solve --lib scalar_program_tests -j 4 -- --include-ignored --nocapture
cargo +1.97.1 clippy --locked -p rumoca-ir-solve --all-targets --all-features --no-deps -j 4 -- -D warnings
cargo +1.97.1 xtask verify msl-parity --sim-targets-file infra/verification/msl-canary-20.json --results-dir <ignored-evidence-directory>
```

Thirty focused tests pass before and after the implementation. They include
sparse and repeated identities, distinct source spans, shared clones, checked
serialization replay, empty blocks and malformed wire/provenance cases. The
ignored diagnostic measures 131,072 lookups over 1,024 four-output programs:
490,615,183 ns before and 462,373 ns after. This isolates the lookup algorithm;
it does not establish the complete constructor or application loading time.
The owner-only all-target/all-feature Clippy check passes without warnings.

The fixed twenty-model canary uses MSL 4.1.0 and OpenModelica 1.27.1~2-g6db4671.
All twenty phase/error rows and channel-band rows match the preceding
private-hasher canary exactly: twenty parse/flatten, fourteen DAE/solve, and
twelve initialized/simulated/high-band traces. The six DAE and two solve
refusals remain visible. There are no timeouts, skipped or missing compared
traces. The roster digest is
`536004933ce502f7a8a4317e5dd6de2b700a6b176e84d2d6e7bbe0dcdf61eafa`.
Raw results and source hashes remain ignored in the consumer checkout under
`.cache/codex-evidence/performance-20261023/output-index-msl-canary/`.
This is Tier 1 focused/canary qualification, not a full workspace or 566-model
cohort claim. Production loading and coupled-tick targets remain separate gates.
