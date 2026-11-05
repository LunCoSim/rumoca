# Simulation Runtime

The simulation runtime turns a compiled model into trajectories. It is
organized around shared solver interfaces: generic simulation policy lives
in the shared runtime layer, and solver backends are thin adapters.

## Layering

| Crate | Role |
|---|---|
| `rumoca-sim` | Simulation orchestration over the compiled model |
| `rumoca-solver` | Shared solver API, result types, report payloads |
| `rumoca-solver-rk45` | Explicit Runge–Kutta-style backend (`rk-like`) |
| `rumoca-solver-diffsol` | Implicit backend via diffsol (`bdf`) |
| `rumoca-input`, `rumoca-input-keyboard`, `rumoca-input-gamepad` | Interactive input devices |
| `rumoca-signal-frame` | Signal payload types |
| `rumoca-transport-udp`, `rumoca-transport-websocket` | External coupling and viewer transport |
| `rumoca-web` | Browser viewer assets |

## Shared Responsibilities

`MeSimulationSession::parameter_values()` reads initialized numeric parameters
through the component's checked FMI variable inventory and read-only value
references. It includes inherited, array and initialization-solved values at
the current accepted coordinate. Inputs, outputs and constants retain their
separate inventories. A terminated or failed session rejects the observation.
The live `rumoca_sim::rk45::SimulationSession` forwards this common-host API.
`crates/rumoca-sim/tests/initialized_parameters.rs` exercises dependent defaults,
an initial-equation parameter, an override, reset and rejected parameter writes.

These belong in the shared runtime/solver API layer, never duplicated in a
backend:

- event schedules and root handling (zero-crossing location, event
  iteration, `__pre__.*` slot updates at event entry),
- input routing and zero-order-hold behavior,
- result collection and the report payload,
- termination (`terminate`) and assertion handling,
- pacing: `as_fast_as_possible`, `realtime`, and `lockstep` modes.

If two solvers need the same code, it moves into the shared layer. Solver
backends implement the integration method and consume resolved input
values — they must not know about keyboards, gamepads, or transports.

## Native projection preparation

Every native module uses `rumoca-exec-cranelift`'s shared `host_jit_builder`.
It enables Cranelift's inline stack probes so large generated frames commit
stack pages in order on a fresh worker stack. Probe emission belongs to native
code generation; hosts do not pre-touch stacks or enlarge them to admit a model.
Inline emission requires no external probe helper. Cranelift includes these
flags in native artifact identity, so an artifact compiled without probes
cannot satisfy a probed compilation request. The `large_jit_frame_on_fresh_worker_stack`
regression compiles a 768 KiB frame before creating its executing thread and
checks both frame edges across a native helper call.

`rumoca-exec-cranelift` emits each construction-issued colored projection in
the compiled source's shared JIT module. Operation reuse storage belongs to
that application: allocate it only for selected source programs, once across
all colors, and release it after emission. Unselected programs must not incur
per-operation retention allocations. Program identities and color order come
from the checked application; the lookup table never chooses execution order.
Retained native values remain local to emission and are never simulation state.

The native `pow` host call retains finite results in a bounded, thread-local
4,096-entry table. Both f64 input bit patterns must match before reuse; a hash
collision only replaces an entry. Signed zero and result bits are preserved.
Exceptional inputs or results use the ordinary math call and are not retained.
This table contains no component identity, solver coordinate or numerical
history, and is not persisted. It does not change power evaluation, precision,
integration policy or convergence checks. The typed pure-call input/result
storage remains the owner for complete Modelica function calls; scalar math
imports do not carry those function-owner coordinates.

Complete typed pure-function results use the native module's `InputResults`
storage. Coordinates of one to sixteen cells use up to 64 direct-mapped entries
within a 16 KiB storage target; constant or larger coordinates retain one
complete entry. A large tuple can exceed that target for its single entry.
Hashing selects storage only: the complete input bits must still match before
all result cells, assertion predicates and message values are copied. Each
selected entry is invalidated before evaluation and published only after the
whole body succeeds. Primal and directional owners have independent storage,
and tangent cells are part of directional keys. Module lifetime and symbolic
data relocation remain authoritative; mutable results are not persisted with
compiled artifacts.

Native hosts can wrap synchronous construction in
`rumoca_sim::native_code_cache::with_native_code_cache`. The host supplies a
bounded `CacheKvStore` containing trusted Cranelift artifacts and owns storage
and persistence. The scope restores on return or unwinding and stays on its
calling thread. Cranelift's incremental cache owns stencil/version/ISA/flag
identity; each new JIT module applies its current relocations through the
maintained module API. Executable arenas and FMI state retain their ordinary
session lifetimes. The result includes attempts, hits, compilations and rejected
entries; rejections are included in compilations. Hosts must report rejections
and continue with the newly compiled result rather than conceal invalid cache
data. Calls without an installed scope compile normally. The adapter does not
change Modelica equations, compiler flags, solver selection or tolerances.

## Events at Runtime

The DAE hands the runtime explicit metadata: relations (event condition
surfaces), scheduled events (`sample`), event actions (`assert`,
`terminate`), and the `pre`-slot bindings. The runtime's job is mechanical:
detect or schedule the event, advance to the event instant, write pre
slots, apply discrete updates, and restart integration. No backend
rediscovers event structure from expressions.

## Failure Diagnostics

When a simulation produces a non-finite value, the runtime re-runs with NaN
tracing to name the offending variables — the user-facing behavior
documented in the handbook's troubleshooting chapter. Keep that path
working when touching solver internals; it is the difference between a
useful bug report and "solver error".

## Torn tangent workspaces

`TornTangentPreparation` binds one immutable JVP source. Its prepared evaluators
share directional-program preparation and one retained seed/output/coefficient
workspace. Standalone `TornTangentEvaluator::new` uses that same preparation
owner. Each invocation checks out its buffers, so callbacks may reenter the
same evaluator or another evaluator in the group without aliasing active work.
The group retains one returned buffer set; recursive scratch is released when
the outer invocation returns, and teardown releases the group.

The seed storage grows to the largest requested extent without shrinking.
Callbacks receive only the current logical seed slice. Before a sweep, the
workspace clears every range the previous plan could write, including ranges
registered before an error or non-finite coefficient. It then registers the
current plan's tear/causal seed ranges before execution. Coordinates outside
those ranges retain positive zero. This preserves a fresh dense seed's values
without clearing the entire global coordinate matrix per sweep. Output and
coefficient lengths reset each time; all programs and arithmetic still execute
in the checked plan's original order. Jacobian results remain independently
owned. Invalid coordinate extents fail before invoking a tangent program.

## Selected-output execution

`SolveRuntime` owns reusable scratch for dynamic selected-program outputs. Each dynamic request
clears and refills `(program, request order, output, offset)` records, validates
every requested output owner, then sorts by program and request order. Programs
execute in ascending order; each program copies outputs in original request
order, including duplicates. The same iterator feeds compiled and checked
interpreter output copying, retaining diagnostics and f64 bits. No computed value
or plan identity is cached in this scratch. Its capacity follows the largest
admitted selection, drops with the runtime, and is independently owned by a
runtime clone. Every reader refills it before use, including after a failed
selection or numerical reset.

## Root condition execution

The existing root plan owns the public and root-search memberships, including
announced-time and continuously static defaults. `SolveRuntime` prepares each
fixed selection as a checked `ScalarProgramBlock` at construction. Selected
source programs retain ascending program order, original operations and source
spans; their complete outputs feed a retained map back to the selected global
positions. Unselected programs do not execute. Grouped, sparse and repeated
output requests retain their declared identity and f64 bits.

The standard whole-block native expression backend receives one dispatch for
each nonempty selection; backend kernel chunking remains its own mechanism.
Identical public/search selections and alternate-chart programs share immutable
code only after structural equality. Changed selections compile independently,
so a replacement cannot execute a superseded program through an overlay.
Numerical buffers remain runtime-owned and are overwritten at every evaluation.
Current state, parameters, time, external tables and evaluator context are read
for every call; no input-dependent value or COM term is memoized.

A backend compilation decline uses the prepared interpreter over the same
selected programs. An admitted native execution error propagates without
recompilation or interpreter retry. Destination extents are checked before
execution, and failed evaluation publishes no selected values. The state/root
refresh, root-location algorithm and indicator-default rules are unchanged.

The solver's `root_batches` and `root_condition_batches::tests` cover native
batch cardinality, terminal errors, grouped/sparse/duplicate ownership, signed
zero, skipped programs and invalid destinations. The native adapter's ignored
`native_execution::tests::root_batches::fixed_root_selection_diagnostic` measures
384 complete root values over 512 fresh-time evaluations and asserts every f64
bit. Pair that diagnostic with the fixed MSL canary and a source-matched
production co-simulation run; it is not whole-worker or FPS acceptance.

## Incremental visible observations

The common ME session exposes `visit_visible_values` for consumers that need
ordered rows rather than a name map. It evaluates the complete public
observation and warnings, and checks output cardinality and recorded input
positions before visiting any value. The consumer still owns its required
finite-value check. Declared outputs retain their order; host-applied inputs absent from
outputs follow in admission order. Input recording resolves an output index
once against immutable component metadata. The host retains output labels as
canonical `rumoca_core::VarName` handles at construction and input labels at
first admission; reset clears each input's value, index and label together.
The visitor borrows those handles, so consumers can clone their shared payloads
without interning or copying text on every observation. Serialized labels stay
textual. The named `visible_values` reader collects the same visitor and renders
names as text. RK45 forwards through its existing common backend; no integration
policy or second observation evaluator is introduced. Values retain f64 precision.
