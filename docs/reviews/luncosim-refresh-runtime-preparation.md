# Shared refresh preparation and runtime sweep workspace

`SharedAssignmentSchedule` owns checked immutable scalar programs, source
provenance and per-program solver targets. The exact schedule constructor runs
the shared-value equivalence checker, then moves the proven operations into
`ScalarProgramBlock`. The runtime retains that result for interpreter and native
preparation. Native compilation accepts the proven schedule directly; generated
C reads the same IR contract. Each program commits its ordered targets before
the next executes. Mutable evaluator outputs remain runtime-owned.

`TornSweepCache` owns reusable raw residuals, dense native outputs and, with debug
assertions, the independent per-row reference state and residuals. Each sweep
refills its live data; dense native outputs are zeroed before every call.
Cloning a runtime resets this pointer-keyed cache and its workspace. Reentrant
workspace use and allocation failure report typed runtime errors. The existing
independent per-row replay, full f64 bit comparisons, finite checks and native
decline protocol are unchanged.

## Verification

- 130 focused solver runtime tests pass, including ordered schedule execution,
  borrowed proof storage, changed inputs, overflow decline/recovery, warm buffer
  reuse, clone isolation, explicit reentry rejection and signed-zero output.
- Five Cranelift assignment tests pass, including ordered dependencies,
  correlated target commits and validation before writes.
- All-target/all-feature Clippy passes for the five changed owning packages
  with `--no-deps -D warnings`, using application toolchain Rust 1.97.1.
- The fixed twenty-model Tier 1 MSL canary completes: twelve models are strict
  high against OpenModelica 1.27.1, with no trace deviations or simulation
  timeouts. Six stop before solve construction and two receive structural
  lowering refusals (EL005). This focused run skips the full-cohort quality
  delta gate and is not Tier 2 parity qualification.

No tolerance, numerical check, operation order or external dependency changes
are introduced. Application timing and complete Twin qualification are owned
by LunCoSim's pinned production build; these native checks alone establish no
scene-level speedup.
