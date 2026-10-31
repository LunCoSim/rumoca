# Shared immutable dependency sets

The warm Griffin initializer spends 845.9 ms reconstructing its checked prepared
model. CPU samples identify dependency propagation, set copies and destruction
inside `rumoca-ir-solve::structural_pattern`; JSON parsing is a smaller part.
The measurement comes from the owned integrated LunCoSim Tracy capture at
`optimization/.cache/griffin-loading-20261030/main-integration/loading-tracy-warm/`.
It is diagnostic evidence, not a product loading-time result.

Under the existing SPEC_0039 dependency rules and SPEC_0021 deterministic
collection contract, register copies share immutable `Arc<BTreeSet<usize>>`
storage. Union detaches through `Arc::make_mut` before modification. Empty-set
union and union with the exact same allocation reuse the existing set. The
refresh dependency reader retains those shared sets; public result APIs still
return owned, ordered sets. Sharing is scoped to the existing construction and
reader lifetime, with no global cache, changed wire format or skipped checks.
The reader retains its thread-safety traits. All existing operation dispatch,
provenance, malformed-register and checked replay boundaries remain active.

All 427 IR library tests pass. The new fixture verifies that modifying one
register does not change an earlier copy and rejects an uninitialized register.
Scoped library/test Clippy passes. Three alternating before/after diagnostic
pairs use the same repository toolchain and seven samples per case. For 1,024
register copies, median-of-run-medians changes as follows:

| Dependency width | Before | After | Ratio |
| --- | ---: | ---: | ---: |
| 1 | 75.361 us | 17.233 us | 4.37x |
| 64 | 759.902 us | 45.976 us | 16.53x |
| 256 | 5138.735 us | 332.011 us | 15.48x |

The fixed 20-model MSL canary exits zero with no per-model phase, diagnostic,
simulation-status or agreement-band delta from `ff830339`. Twelve models compare
strict-high against OpenModelica, with zero skipped or missing traces. The same
six compilation failures and two solver failures remain visible. The durable
comparison is `shared-dependency-sets-canary.json`. Production Griffin
measurement is pending; these checks establish neither full-cohort parity nor
the five-second loading target.
