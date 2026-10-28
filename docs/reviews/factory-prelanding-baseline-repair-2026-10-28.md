# Factory pre-landing baseline repair

Anchors: SPEC_0021 module size and Cargo-selected tests; SPEC_0029 and SPEC_0041 facade ownership; SPEC_0036 checked construction; SPEC_0037 unchanged proof checks; SPEC_0043 section 6a shared-value construction; SPEC_0033 focused evidence before broad gates.

The pre-landing gate reached the architecture suite with 239 passes and six failures. The repaired suite passes all 245 checks without raising limits or removing checks.

- Parser contract expectations now name the active exact Parol 4.7.0 / runtime 4.5.0 pins. The staging, generated-file drift and regeneration-documentation checks remain intact.
- The simulation facade whitelist includes its existing checked runtime FMI wire producer. The documented external `rk45::SimulationSession::from_fmi_component` contract remains public; LunCoSim calls it. Its explicit disposition points to SPEC_0041, and an external-component regression exercises periodic events, continuous state and reset through interpreter and native policies. The uncalled `sample_capacity` method is removed and pinned as a removed declaration-only surface.
- Class navigation now lives in a real session submodule. The query/cache file is below the 2,000-line action threshold; source resolution and cache ownership are unchanged. No file-size exception or source-path attribute was added.
- Ownership timing diagnostics use `rumoca-ir-solve/diagnostic-benchmarks`. The native relocation proof uses `rumoca-exec-cranelift/cross-isa-tests`, which selects Cranelift's ARM64 backend. There are no ignored tests; the cross-ISA proof and diagnostics remain executable with Cargo.
- Symbolic operation identities retain their compact nonzero representation. Allocation is checked at the local constructor; exhausted identity space returns `SharedValueError::IdentitySpaceExhausted`. `SharedValueSegments::derive` returns a result, and schedule construction propagates failure. Single-program admission retains its existing counted proof-failure behavior. The totality ceiling remains 15, and the added runtime assertion is removed; no key can be silently saturated, wrapped or reused.

The public derivation signature is a clean internal API cutover: its schedule caller and all tests consume the result. No compatibility wrapper, ignored failure or relaxed proof checker was introduced. Full pre-landing gate results are recorded against the final committed tip, separately from the bounded twenty-model canary and the factory's native production evidence.

Focused validation: 400 compiler, 114 native execution, 426 Solve IR and 154 simulation library tests pass (1,094 total), with both optional diagnostic/cross-ISA features selected. Workspace clippy passes. The fixed twenty-model canary retains 14 compile successes, 12 simulations and 12/12 compared HIGH traces; two solver failures and six ToDAE refusals remain visible. This is bounded evidence, not full-cohort acceptance.
