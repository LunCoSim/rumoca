# Complete visible observations with canonical variable labels

`MeSimulationSession` remains the public-observation and input owner
(SPEC_0044 §6, ME-HOST-001 and ME-BUF-001). Its visitor evaluates the complete
component observation and warnings, validates cardinality and recorded-input
positions, then borrows `rumoca_core::VarName` handles and f64 values in output
order. Applied inputs absent from outputs follow in admission order. The common
host resolves output labels once during construction and input labels on first
admission. Reset retires each applied input's value, output position and label.
No numeric operation, tolerance, time coordinate or integration policy changes.

SPEC_0029 §3a owns variable identity in `rumoca-core`. The host reuses that
interner and its shared immutable payload; no second interner or per-observation
text lookup is added. Clients can clone handles into typed result messages.
The named-map SDK reader collects the same visitor and renders text for its
real named consumers. RK45 forwards the visitor through its common backend.
Native source text and serialized variable names remain unchanged.

LunCoSim's fixed/adaptive stepper validates the complete finite sample before
publication. Its worker, accepted samples and binary worker protocol retain
canonical handles; serde writes names as text. Reflected variable snapshots
retain String keys at their existing owner. Metadata and step rows use one
normalized admission mechanism, preserving input/parameter classification,
reordering, omissions, duplicate last writes and signed zero. This is an engine
representation boundary; no changeable Rhai policy or hook belongs here.

A fresh matching Tracy 0.14.1 capture on integrated LunCoSim main `dc92300ce`
records 2.1792 ms mean transactions in the ten-input cohort, including 1.5890 ms
integration and 0.5749 ms observation collection. Allocation/free-list routines
account for 431 of 856 selected observation leaf samples. Unwinding does not
recover their domain callers, so the allocation is not attributed to individual
call sites from those samples alone. The common visitor's text copies are
measured separately with the production collection diagnostic.

Six alternating independent release executables measure fifteen samples per
producer: 512 complete reads of 2,305 values take median 36.3211 ms with copied
text and 14.5298 ms with shared handles (60.00 percent lower). Ranges are
35.5682–39.5360 and 13.9639–15.4763 ms. Every point's name/value checks pass;
reported input, time and terminal value bits also match between executables.
This is a collection result, not whole-worker, coupled-tick or FPS acceptance.

Nine fixed-step numerical tests and worker admission/fingerprint tests pass.
Native input/session checks preserve hidden inputs, shared payloads, signed
zero, atomic rejection, failed-session refusal and reset. The SDK RK45-enabled
lib/tests Clippy gate passes. The fixed twenty-model MSL canary compares twelve
traces strictly high with zero missing/skipped traces; six ToDAE and two typed
solve-IR refusals remain unchanged. All twenty classifications match, and all
562,411 data values plus 16,841 timestamps match the fresh baseline bit-for-bit.
This is Tier1 evidence, not full-cohort parity or full-Griffin exact replay.

Canary artifacts are retained in `target/msl/shared-label-before` and
`target/msl/shared-label-after`. Application diagnostics and independent
executables are in optimization's ignored
`.cache/perf/modelica-shared-labels-20261104/`; the new pre-change Tracy capture
is in `.cache/perf/modelica-after-main-20261104/tracy/`. Earlier ignored
observation-stage evidence was removed by cleanup; its paths are not retained.
Whole-scene qualification and current executable identity belong in
optimization's `docs/reviews/griffin-2k-performance-2026-11-04.md`.
The API and consumer remain local patches through explicit overrides; no
published or portable dependency promotion is claimed.
