# Initialized parameter observations

`MeSimulationSession::parameter_values` reads the settled numeric parameter
inventory from the checked FMI metadata and current component observation.
The runtime resolves storage runs and branded read-only references; callers do
not infer positions from names or inspect lowered initialization seeds. RK45
forwards the same reader. String parameters, inputs, constants and outputs keep
their separate inventories. This follows the common ME observation owner in
SPEC_0044 and the crate boundaries in SPEC_0029.

The LunCoSim consumer needs this result to populate its numeric parameter editor
after compilation and reset. Expression, inherited and initialization-solved
values therefore come from the initialized instance.

## Verification

The focused `rumoca-sim` `initialized_parameters` integration test passes with
`--no-default-features --features solver-rk45`. It covers expressions, inherited
declarations, arrays, Integer, Boolean, String exclusion, initialization equations,
overrides, reset and refusal to treat a parameter as a writable input. The
`rumoca-solver --lib fmi_me` filter passes all 184 tests.

The fixed 20-model canary ran on both a fresh detached `c8fce575` checkout and
this patch, using `cargo xtask verify msl-parity --sim-targets-file
infra/verification/msl-canary-20.json`. All 20 classification rows match exactly.
Both produce 14 DAE/compiled models, 12 successful simulations and 12 strict-high
OpenModelica comparisons, with no near/deviation bands, skipped comparisons or
missing traces. Six ToDAE refusals and two simulation failures remain visible.
The identical band-row digest is
`07fb475915d477780bcca32fcb59a9e4268ad71e7d6312bb32184116374a8a6d`.
This is a focused canary delta, not a full-cohort claim or baseline promotion.
Candidate artifacts are in `target/msl/initialized-parameter-canary`.

The combined solver/SDK library Clippy check stops on three existing excessive
nesting errors in unchanged `runtime/solve_runtime/relation_memory.rs` and
`tangent_evaluators.rs`. No suppression or unrelated dependency change is added.
