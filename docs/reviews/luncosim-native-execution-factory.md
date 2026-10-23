# Native execution for external Model Exchange hosts

`rumoca_sim::fmi_execution_backend` accepts a checked `FmiComponent` and
`SimOptions`, then returns the opaque `MeExecutionBackend` issued by the same
native-admission owner as the built-in solvers. External numerical plugins do
not inspect Solve IR or implement another Cranelift adapter. Explicit
Interpreter policy and targets without native compilation retain the existing
admission semantics. This follows SPEC_0038's Internal Solver Boundary and
SPEC_0041's simulation-facade ownership.

The production consumer is LunCoSim's `FixedStepSession`. It supplies this
handle to `MeRetainedComponent::instantiate`, while its fixed RK4 numerical
plugin, communication lattice, input transactions and full checked observation
retain their owners. No compiler IR, wire format, numerical method or FMI
lifecycle rule changes.

## Qualification

The focused `fmi_execution_backend_honors_external_host_policy` test passes.
LunCoSim's six existing fixed-step numerical tests also pass, including atomic
invalid-input rejection, exact communication points, analytical accuracy and
bitwise replay over 512 ordered input changes. Its nonlinear-array differential
agrees with explicit Interpreter execution within 1e-12 through 100 input
changes. These checks use Rust 1.97.1, the production consumer's toolchain.

The regular Tracy application build passes. The production Modelica drivetrain
scene passes all eight authored checks, and the non-finite-observation scene
passes its expected terminal `modelica-step-failed` fault. Owned Griffin, Tesla
and Summer loading observations retain clear diagnostics and inspected final
images. All owned application sessions are closed. These short observations
do not qualify full missions or the application's strict performance targets.

The Tier 1 fixed twenty-model canary runs from this Rumoca checkout. Set
`LUNCOSIM_ROOT` to the consumer checkout and `RESULTS_DIR` to its ignored
evidence directory:

```sh
CARGO_TARGET_DIR="$LUNCOSIM_ROOT/target" RUSTUP_TOOLCHAIN=1.97.1 \
  CARGO_BUILD_JOBS=4 RUST_TEST_THREADS=4 RAYON_NUM_THREADS=4 \
  cargo run -p xtask --bin xtask -j4 -- verify msl-parity \
  --sim-targets-file infra/verification/msl-canary-20.json \
  --results-dir "$RESULTS_DIR" --serialize
```

It completes with MSL v4.1.0 and OpenModelica 1.27.1~2-g6db4671: 20 targets
parse/flatten, 14 reach checked DAE/solve, and 12 initialize and simulate. All
12 candidate traces compare in the high band with zero channel deviations,
skipped models or missing traces. Six DAE refusals and two solve refusals remain
failures. These counts match the recorded aggregate canary baseline in
`luncosim-runtime-0.10.2.md`; the aggregate boundary delta is zero. No previous
per-model band table was retained for an automatic identity delta. This is
Tier 1 evidence, not a full-cohort parity or performance claim.

Raw evidence is local and ignored under the consumer checkout's
`.cache/codex-evidence/performance-20261023/`: `fixed-native-*` logs,
`fixed-native-gates/`, `*-fixed-native-profile/` and
`fixed-native-msl-canary-fork/`. The canary's `qualification-summary.json`
records the tested working-tree digest, target-list digest and aggregate
reference. Application path overrides disable prepared-model disk caching;
portable Git-pin loading acceptance requires a separate run after publication.
