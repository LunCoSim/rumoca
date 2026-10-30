# Bounded native program dispatch

The execution adapter owns native emission under SPEC_0029 and SPEC_0041 §3.
This change preserves Solve IR and Modelica semantics. An owned Griffin capture
attributes substantial constructor work to per-function hashing and native
preparation even when every artifact is cached. The initializer contains 5,040
scalar residual programs; separate selectable functions multiply that work.

`rumoca-exec-cranelift::emit::selected_dispatch` emits up to 32 independent
programs per function, bounded by 4,000 operation cost except for an indivisible
program. A private integer selector branches to exactly one arm. Each arm uses
the shared independent-row lowering implementation with fresh register, loaded
value and conditional-result maps. Output offsets and selected-program input
requirements remain with the existing compiled row metadata. Invalid selectors
return a backend failure without touching output.

Shared conditional residual owners keep their full-call execution boundary;
large residuals keep their chunked register tape. Projection application batches
retain their separate full-application ABI. Each compiled owner retains its own
code arena and mutable state; no live solver or executable memory is shared
between runtimes. Native artifact identity and relocation checks are unchanged.

The focused regression fails before this change with 64 definitions instead of
2. All 115 execution-adapter tests pass after the change, including exact selected
outputs, invalid selector rejection, skipped failable programs, shared conditional
ownership, external table and tensor failures, projection and native cache reuse.
The fixed 20-model canary exits zero with no model-status, diagnostic-code or
agreement-band delta from `257ee265`: 14 compile, 12 simulate and all 12 compare
strict-high against OpenModelica, with zero skipped or missing traces. Six
compilation failures and two solver failures remain visible. This is not full
cohort parity. The durable comparison is
`selected-native-dispatch-canary.json`; memory-aware scheduling admitted one
worker under concurrent workloads despite a configured ceiling of four.
Production Griffin measurements belong in the application loading-budget
review. Function-count reduction is not a product loading-time result.
