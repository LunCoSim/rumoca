# Inspecting and Debugging Models

When a model misbehaves — fails to compile, fails to initialize, or produces
wrong dynamics — Rumoca gives you structured views into every stage of the
pipeline. All of these work with both `rumoca compile` and `rumoca sim`.

## Observe Initialization from Rust

`rumoca_sim::initialize_fmi_component(&component, &options, &requested_outputs)`
initializes an already prepared `FmiComponent` and returns finite values in the
requested name order without attaching an integration plugin. Prepare the
component with the admitted parameter and initial-input overrides through
`prepare_fmi_component` first. The API requires `solver-rk45` or
`solver-diffsol`, which provide the common execution-backend composition.

The common Model Exchange host settles initialization and its discrete events.
Only requested output programs are evaluated for the result; a grouped program
remains one execution unit. Initialization and public-coordinate settlement
still perform their required equation work. Unknown names, non-finite results
and initialization termination return errors. This operation releases its
initialized instance after observation; it does not export state for another
instance or establish a coupled-participant state-transfer contract.

For an admitted continuous initial solution, a prepared `FmiComponent` exposes
`free_initial_state_names()` and `initial_state_constraint_features()`.
Construct each `InitialStateConstraint` with its exact scalar name and finite
accepted value. `with_free_initial_state_constraints(&constraints)` validates
names, domains and duplicates first; given, equation-owned and update-owned
coordinates cannot be replaced. Values bitwise equal to the checked prepared
`initial_y` require no change. An all-identity request returns a clone sharing
the original component program, with its initialization and event owners intact.
This requires no event/history/chart or parameter-solve transfer.

Only nonredundant values pass the continuous-state capability guard and acquire
given initialization ownership in a per-instance component. Every original
initial equation remains active and the cached base stays unchanged. This is
not a start override or an FMU snapshot import. Callers must verify all accepted
observations after initialization events, including identity requests: preserving
an event program does not certify that its final observations match another
instance. Unsupported nonredundant transfer reports each coordinate's requested
and prepared values alongside the component features.

## Dump an Intermediate Representation

`--emit` prints the model as the compiler sees it after each stage:

| Stage | What you see |
|---|---|
| `ast-json` | The parsed, resolved semantic tree (no lossy Modelica reconstruction) |
| `flat-mo` / `flat-json` | The flattened model: hierarchy and `connect`s expanded |
| `dae-mo` / `dae-json` | The DAE system: equations partitioned, ready for analysis |
| `solve-json` | The solver IR: sorted, torn, scheduled for execution |

```bash
rumoca compile Model.mo --emit flat-mo          # to stdout
rumoca compile Model.mo --emit dae-json -o m.json
```

Reading `flat-mo` answers "what did my modifications and connects actually
produce?". Reading `dae-mo` answers "what equation system is the solver
given?" — the live examples in this book expose the same view through their
**Show DAE** button.

The JSON dumps are for debugging and for replay with the same compiler build,
not a stable interface. `dae-json` carries a `schema_version` that changes as
the IR evolves, and its reader accepts only the current version; `flat-json`
has no version. For a model's inputs and outputs, read the `modelDescription.xml`
of an `fmi2` or `fmi3` export instead.

In `dae-json`, a variable's `role` is its runtime classification (`state`,
`algebraic`, `output`, `discrete_value`, ...), not the declared prefix: a
declared `output` that is a state has role `state`, and a discrete one has
role `discrete_value`. The declared prefix is `declared_causality` (`none`,
`input`, or `output`), kept for nested components too, while `causality` is
`input` or `output` only at the top level. FMI exports mark a nested
declaration's prefix with a per-variable annotation and keep its causality
`local`:

```xml
<!-- FMI 2, inside the ScalarVariable after its type element -->
<Annotations><Tool name="rumoca"><DeclaredCausality value="output"/></Tool></Annotations>
<!-- FMI 3, first child of the variable -->
<Annotations><Annotation type="rumoca.declaredCausality">output</Annotation></Annotations>
```

## Structural Analysis

```bash
rumoca compile Model.mo --inspect structure
```

Prints the structural preparation of the system: the matching between
equations and unknowns, the block lower-triangular (BLT) ordering,
simultaneous (coupled) blocks, and tearing decisions. This is the first
place to look when compilation fails with *structurally singular system* —
it names the unmatched equations and unknowns.

## Numerical Evaluation at a Point

```bash
rumoca sim Model.mo --inspect eval
rumoca sim Model.mo --inspect eval --at "x=1.5,v=0@2.0"
```

Evaluates all solver values and state derivatives at a point and names any
non-finite results — the fastest way to find the division-by-zero or domain
error behind a NaN. With no `--at`, it evaluates at the initial state (which
also discovers the state names for you).

The `--at` syntax is `<name=value,...@t>`: states by name, unset states keep
their initial values, time after `@` (default 0).

## Jacobian Analysis

```bash
rumoca sim Model.mo --inspect jacobian --at "x=1.0@0"
```

Prints the dense state Jacobian at a point and flags singular columns and
zero pivots — useful for diagnosing initialization failures and stiff or
degenerate dynamics.

## NaN Tracing

When a simulation fails with a non-finite value, `rumoca sim` automatically
re-runs with NaN tracing to locate the offending variables, so the
diagnostic names the variable instead of just reporting a solver failure.

## Performance

```bash
rumoca sim bench Model.mo            # compile / prepare / hot-loop timing
rumoca cache status                  # compilation cache usage
```

## Verbose Compilation

```bash
rumoca compile Model.mo --target c-ode -o out -v
```

`-v` prints friendly `[rumoca] Phase ...` progress lines, which localizes
slow or failing phases on large models.
