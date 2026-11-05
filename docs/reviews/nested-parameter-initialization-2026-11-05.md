# Nested parameter initialization

The modifier `Child child(p=p)` must keep a dependency on the enclosing `p`,
including when the root parameter has an explicit compilation-time modification.
Changing a tunable value when preparing a new instance must reach the child's
initial equations without recompiling the model.

## First divergent boundary

At revision `5a8dbefbdb41384a62361e9b012cd4693fefbded`, an explicit root
modification `p=60` caused `inherited_modifier_source_metadata` to treat a
resolved enclosing declaration reference as forwarded modifier metadata.
It replaced the child binding's source with the parent's literal modification.
Flat therefore contained `child.p = 60`, rather than a VarRef to the enclosing
instance. Later FMI preparation could change root `p` to 1, but the lost
binding edge prevented that change reaching the child. The native simulator
correctly rejected the mismatched initialized state.

Instantiation owns modifier provenance. The fix checks resolved declaration
identity against the enclosing components before consulting forwarded metadata.
An enclosing declaration keeps its lexical source expression and scope. Flatten
then resolves that expression to the correct concrete instance. Genuine
inherited modifier forwarding keeps its existing path. This implements
MLS section 7.2 lexical modification scope, INST-001 and FLAT-C01. It adds no
model names, parameter-specific propagation, runtime repair or cache exception.

## Regression evidence

- The original scalar counterexample passed without a root modification but
  failed with explicit root `p=60` followed by instance override `p=1`.
- The repaired test asserts the Flat VarRef targets the exact enclosing instance,
  then prepares and initializes one compiled model at 60, 1, 2, -4 and 60.
- Nested-chain coverage includes inherited declarations, arithmetic expressions,
  opposite sibling inputs, an independent literal branch, and a child-specific
  override that propagates only through its own branch.
- Array coverage includes `each`, element-wise modifiers and qualified references.
- Existing modifier-scope suite: 33 passed. The simulator's 13 artifact-contract
  tests passed, including structural/tunable cache separation, indexed array
  overlays and a repeated initialization test asserting the same compiled Arc
  is reused for every tunable value.

## Fixed canary delta

The required fixed 20-model Tier 1 run was performed once before and once after
with build, test and Rayon concurrency set to 4. Both commands exited 0.
Both runs flattened 20 models, produced 14 DAE models and completed 12 simulations.
Six existing DAE failures and two existing solver failures remain; they are not
counted as successes. All 12 comparable traces were in the comparator's highest
agreement band, with no missing or skipped candidate traces, before and after.
There was no coverage/status regression. This is a partial canary result, not a
566-model cohort parity claim. Full cohort verification was not run.

See [machine-readable evidence](nested-parameter-initialization-canary.json) for
per-model results and hashes of the original result and comparator artifacts.

```sh
CARGO_BUILD_JOBS=4 RUST_TEST_THREADS=4 RAYON_NUM_THREADS=4 \
  cargo xtask verify msl-parity \
  --sim-targets-file infra/verification/msl-canary-20.json \
  --stage-parallelism 4 --sim-parallelism 4 --no-remote-quality-baseline \
  --results-dir <fresh-evidence-directory>
```

Native acceptance passed in one headful LunCoSim session: 60-metre descent,
1-metre drop, 2-metre drop and another 60-metre descent, without a process
restart. The first drop samples report exactly 1 and 2 metres, zero initial
speed and zero thrust. All four authored recording-cadence verdicts passed.
The new executable SHA-256 is
`b8ab6d64a0c6df5ffa1b59154832ec256ce97cc1b9f22d62230721a9c690cb46`.
Array and qualified-reference initialization coverage passed with the complete
five-test initialization suite. This runtime evidence establishes scene
initialization and cadence; the finished film still needs its own visual QA.
