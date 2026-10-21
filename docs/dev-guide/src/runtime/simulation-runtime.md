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
