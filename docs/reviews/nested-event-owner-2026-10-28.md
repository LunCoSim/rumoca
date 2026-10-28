# Nested sampled event owners

Normative anchors: MLS 11.1.2 and 11.2.7 / Appendix B; SPEC_0007 DAE-C21 and SOLVE-C57/58; SPEC_0033 sections 2–4; SPEC_0036 construction-owned complete producer coverage. An instantiated model event algorithm retains its ordered target transaction. Equal trigger/guard expressions do not identify the algorithm owner.

A two-instance counter compiled to checked DAE with two model event transactions, then failed at Solve lowering: `discrete.event_iteration_plan row 0 ... a transaction does not cover one complete guarded producer program`. The single-instance conditional counter passed. This locates the earliest divergence after DAE, rather than at parsing, component name resolution or runtime clock dispatch.

`lower_guarded_targets` fused independent targets with equal clock, pre mode and guards into one producer, even when different model event transactions owned those targets. `claim_event_transaction` then attached each part of the shared guarded program to a different transaction. The unchanged complete-program checker correctly refused the result.

The fix indexes constructor-owned DAE transaction identity by target variable, then requires that identity to agree before fusion. Unowned equation targets retain their prior fusion. No checker, clock, arithmetic, tolerance or source-name special case changed.

Focused before: one conditional counter passed; the two-instance algorithm test failed at Solve lowering. After: both tests pass, including the two-instance trajectory through interpreter and native automatic execution. Expected final states are 3, timers 30 seconds, and the exposed alias agrees. The test asserts that DAE retains two transaction owners before Solve construction.

Application integration uses approved sibling source paths to the repaired Rumoca checkout. No dependency patch is published or pushed. The factory's separately observed zero-production result still requires production verification; this counter fix alone does not close that counterexample.

## Silent counterexample: structural replay dropped the ordered owner

The complete factory still returned zero production after the fusion fix. Its isolated weighing stage behaved identically under the interpreter and native backend, with correct parameters and true admission conditions. This rules out a native-code-only bug and downstream backpressure.

Before repair, the source checked DAE had one ordered transaction, while the final Solve artifact had none. `semantic_owners::rebuild_semantic_owners` copied discrete equations and value owners but omitted the DAE-C21 model-event transaction arena. Alias quotient reconstruction therefore discarded the source order; the initial unconditional hold assignment became the first active guarded branch. The ordered start update never executed.

The shared semantic replay now reconstructs every transaction's typed targets, ordered steps, conditions, clocks, definitions, expressions and provenance through the checked `model_events` constructor. The checker is unchanged. A generic vector counter with a continuous alias proves that quotient reconstruction preserves its transaction before checking the trajectory.

Artifacts retained by the factory caller show Solve ordered transactions changing from zero to one for the isolated weighing stage. It now starts at 5 seconds through both backends. A full eight-hour baseline through both backends yields 19,507 kg packed, 19,262 kg stored, 21 batches started, and species-balance errors below 1e-6 kg. This closes the observed silent numerical counterexample at the library facade; native workbench reintegration is a separate acceptance step.

## Same-tick intermediate read of an invalid relation buffer

MLS 8.5 / Appendix B and SOLVE-C57 require an issued intermediate to read the coordinate tuple produced earlier in the partition. A minimal child computes `moved = if enabled then 25 else 0` and `blocked = enabled and moved < 25`. A sampled parent enables it through `pre(state)` and increments a counter only when blocked. The expected counter is zero. Both interpreter and native execution instead incremented it once.

Checked DAE retains the correct expressions. The issued Solve order correctly refreshes enabled, moved and blocked before the parent transaction, but blocked's program loads tick-entry relation memory instead of comparing the freshly issued moved coordinate. The root predicate was selected for the old tuple. The first divergence is therefore scalar construction of the C57 intermediate, not scheduling order, transaction source order, floating point tolerance or observation output.

The issuer now carries the set of coordinates already issued into intermediate scalar construction. A buffered relation that reads one of those coordinates is compiled against the fresh work tuple. Relations that do not read refreshed coordinates retain their existing buffered side, and history reads remain excluded by the shared structural same-instant read traversal. The runtime, certificate checker and ordinary continuous relation compiler are unchanged. This applies only to already-issued C57 intermediates; it neither inlines their producer graph nor repeats a transaction.

The generic regression passes through Interpreter and Auto. All three factory scenarios (baseline, faster mixing and slow packing) now agree with OpenModelica at all 5,761 unique output times for 23 quantities through both execution policies. Maximum difference is 3.64e-12; blocked-time counters agree exactly. Before/after Solve artifacts are retained by the factory caller under `results/rumoca/boolean-dependency/`. The full event selection passes 97 tests; the fixed 20-model MSL canary retains 12/12 compared HIGH matches, 14 compile successes, 12 simulations, two solver failures and six ToDAE refusals. These counts are a bounded canary result, not full-cohort acceptance or an isolated parent-baseline comparison.
