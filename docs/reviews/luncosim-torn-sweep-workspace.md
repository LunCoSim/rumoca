# Runtime-owned torn-sweep workspace

`rumoca-solver::TornSweepCache` retains scratch vectors for raw residuals,
dense compiled outputs and the independent development-build reference state
and residuals. Every live value is initialized for each evaluation. Numerical
answers are never cached, and cloning a runtime starts with empty independent
storage. Checked reservation failures and reentry return runtime errors.

Causal evaluation order, native decline behavior, finite checks, independent
per-row replay and complete f64 bit comparisons are unchanged. The public
Model Exchange consumer needs no additional cache or API. This is a numerical
runtime mechanism; it introduces no application policy or hook.

## Qualification (2026-10-27)

Based on `41637b1eba187ce1324aa1c511ad55e2f27815f4`, retaining its compact
assignment-producer index and checked immutable assignment schedule.

- `cargo +1.97.1 test --locked -p rumoca-solver --lib runtime::solve_runtime -j 4`:
  130 pass. Focused workspace tests cover changed inputs, overflow and recovery,
  capacity/pointer reuse, clone isolation, typed reentry failure, dense output
  initialization after a backend decline and signed-zero bits.
- `cargo +1.97.1 clippy --locked -p rumoca-solver --all-targets --all-features
  --no-deps -j 4 -- -D warnings`: passes.
- SPEC_0033 fixed twenty-model MSL canary, four stage workers and two simulation
  workers: twelve high-agreement traces against OpenModelica 1.27.1, zero
  deviations/timeouts, six ToDae refusals and two EL005 solve refusals. The
  fourteen simulation outcomes and refusal messages match the prior current-pin
  canary log; its agreement-band counts also match. The prior raw traces are no
  longer available, so no cross-run bit equality is claimed. This partial run
  does not qualify the full cohort or promote its quality baseline.

Commands use the normal target, sccache and four Cargo jobs. The native canary
uses its canonical msl-fast profile. Logs and result JSON remain ignored under
`.cache/codex-evidence/torn-workspace-20261027/`. LunCoSim production replay and
Twin performance acceptance belong to its integration review; no isolated
performance gain is claimed from native tests.
