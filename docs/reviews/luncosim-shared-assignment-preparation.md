# Shared immutable assignment preparation

`SharedAssignmentSchedule` owns checked immutable scalar programs, source
provenance and per-program solver targets. The exact schedule constructor runs
the shared-value equivalence checker, then moves the proven operations into
`ScalarProgramBlock`. The runtime retains that result for interpreter and native
preparation. Native compilation accepts the proven schedule directly; generated
C reads the same IR contract. Each program commits its ordered targets before
the next executes. Mutable evaluator outputs remain runtime-owned.

This construction-only change preserves the runtime's existing sweep storage
and numerical evaluation. Source/owner schedules, proof checks, target order,
finite checks and precision remain unchanged. The owning contracts are
SPEC_0007's checked target boundary, SPEC_0029's IR/runtime ownership, and
SPEC_0036's immutable construction and borrowed proof transitions.

## Verification

- 128 focused solver-runtime tests pass, including ordered assignment execution
  and shared proof storage.
- Five Cranelift assignment tests pass, including correlated commits, ordered
  dependencies and prevalidation before mutation.
- The five affected owner packages pass all-target/all-feature Clippy with
  `--no-deps -D warnings` on Rust 1.97.1.
- The fixed twenty-model Tier 1 canary compares twelve models strictly against
  OpenModelica 1.27.1, with zero deviations, missing traces or skipped comparisons.
  Six models receive ToDAE refusals; two receive structural lowering refusals
  (EL005) before solver runtime. There are no simulation timeouts. This matches
  the preceding combined patch's canary counts; the eight refusals have no
  asserted runtime baseline delta. The focused runner skips the full-cohort
  quality-delta gate. Raw receipts are retained in ignored
  `target/msl/schedule-only-canary`.

Application loading and exact replay are qualified independently through
LunCoSim's pinned production build. These native checks establish no
scene-level timing gain or full-cohort parity.
