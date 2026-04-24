# Per-Equation Annotations in the Rumoca IR

Status: design note, written 2026-04-21. Captures the reasoning behind the
`feat/equation-connect-annotation` branch and the review that killed
several over-designed variants before anything shipped. Read before
touching equation-level metadata or the annotation surface.

## Problem

Modelica sources attach an annotation to every equation:

```modelica
connect(a, b) annotation(Line(points={{0,0},{10,10}}, color={0,0,255}));
x = 1 "a docstring" annotation(__Vendor_foo = true);
```

The parol grammar `Description` node captures both the docstring and the
`annotation(...)` clause. The `TryFrom<&SomeEquation> for rumoca_ir_ast::Equation`
conversion in `crates/rumoca-phase-parse/src/equations.rs` drops them.

Downstream consumers — specifically `lunco-modelica`'s diagram panel — need
the `connect` annotation to render user-authored wire routing
(`Line(points=...)`). With the data dropped at the IR layer, the consumer
regex-scans raw source as a workaround (see
`crates/lunco-modelica/src/ui/panels/diagram.rs` ~L1592).

## What industry tools actually consume

Surveyed OMEdit (via OMC's `Absyn`), Dymola, SystemModeler, ITI SimulationX.
Every diagram editor reads the same set of annotations:

| Annotation | Host | Rumoca status |
|---|---|---|
| `Placement(transformation(extent=, origin=, rotation=))` | `Component.annotation` | reachable |
| `Icon(graphics={Line, Rectangle, Polygon, Text, Ellipse, Bitmap, coordinateSystem})` | `Class.annotation` | reachable |
| `Diagram(graphics=..., coordinateSystem=...)` | `Class.annotation` | reachable |
| `Documentation(info=, revisions=)` | `Class.annotation` | reachable |
| `Dialog(group=, tab=, enable=)` on parameters | `Component.annotation` | reachable |
| `experiment(StartTime=, StopTime=, Tolerance=, Interval=)` | `Class.annotation` | reachable |
| `DynamicSelect(...)` on graphics | nested in `Class.annotation` | reachable (generic modification tree) |
| Vendor `__OpenModelica_` / `__Dymola_` prefixes | anywhere | reachable (permissive grammar) |
| **`connect(...) annotation(Line(points=, color=, thickness=, smooth=, pattern=))`** | **equation-level** | **MISSING** — the gap |

OMC's internal AST does carry an optional `Comment { annotation, string }` on
every equation (structural symmetry). OMEdit *uses* only the connect case.
No editor reads per-equation docstrings or annotations on non-Connect
equations for diagram rendering. Symmetry in the IR is correctness-by-
aesthetics, not a feature enabler.

**Takeaway:** the only equation variant that needs an annotation slot is
`Connect`. Other variants are YAGNI until a concrete consumer appears.

## Regex inventory in lunco-modelica

Found ~20 regex call sites across `crates/lunco-modelica/`. Categorised:

### A — Already solvable by existing rumoca IR (no new work)

- `ui/panels/diagram.rs:1496` — component declarations (`Class.components`)
- `ui/panels/canvas_diagram.rs:1012` — `connect(a.b, c.d)` endpoints (`Equation::Connect`)
- `ui/panels/canvas_diagram.rs:3062-3074` — section counters (iterate IR)
- `ui/panels/diagram.rs:1668`, `bin/msl_indexer.rs:100` — `Placement(transformation(extent=, origin=))` (`Component.annotation`)
- `examples/dup_probe.rs:39` — component scan

### B — Solvable by the connect-annotation fix

- `ui/panels/diagram.rs:1592` — the `connect(...) annotation(Line(points=...))` regex

### C — Hidden coupling: parsing `format!("{:?}", expression)` as a data format

`bin/msl_indexer.rs:446-573` treats rumoca's `Debug` output as a stable
format. A `Debug` tweak in rumoca silently breaks the indexer. Shapes it
parses: `Minus("-")`, `UnsignedInteger`, `String`, `FunctionCall { comp:
Line, args: ... }`, `Rectangle`, `Polygon`, `Text`, `textString="..."`.
This is a bigger latent risk than the missing Connect annotation. It goes
away the moment there's a typed graphical-primitive view.

### D — Legitimate: source text mutation

- `ast_extract.rs:295,322` — parameter/input value rewrite
- `ui/commands.rs:1345-1523` — class opener/closer/within rewrites

No fix possible until rumoca has a Modelica source printer. Document as
`TODO(source-printer)` and move on.

## Design variants considered

### V1 — Narrow field on `Equation::Connect` (commit 445d177)

```rust
Equation::Connect { lhs, rhs, annotation: Option<Annotation> }
```

Touched 12 files. Populated at parse. **Pros:** minimal. **Cons:** does
not propagate through flatten into `InstanceConnection`; docstring still
dropped. Asymmetric: only one variant has annotation support.

### V2 — `EquationItem { equation, description }` wrapper

```rust
pub struct EquationDescription { docstring: Vec<Token>, annotation: Option<Annotation> }
pub struct EquationItem { equation: Equation, description: EquationDescription }
Class.equations: Vec<EquationItem>
```

**Pros:** symmetric, docstring + annotation, every variant covered.
**Cons:** 50+ call sites across resolve/typecheck/flatten/session/lsp,
grammar `nt_type` change + regeneration, `Visitor` trait migration
(~8 overrides + default walker), serde snapshot re-baseline. Large blast
radius for features nobody requests.

### V3 — Chosen: narrow field + flatten propagation + typed annotation API

Keep 445d177's `Option<Annotation>` on `Connect`. Plug the flatten gap
(`InstanceConnection.annotation`). Add a typed query API that works over
`&[Expression]` so **zero migration** is needed for existing
`Component.annotation` / `ClassDef.annotation` users.

Why this over V2:
- Delivers the actual consumer need (`diagram.rs` Line points).
- Kills category C entirely by giving `msl_indexer` typed accessors.
- ~18 files vs. 50+, no grammar regen, no trait changes, no snapshot churn.
- V2 remains a clean upgrade path if a second variant ever needs
  equation-level metadata.

## Decision record — what we are NOT doing

| Rejected | Why |
|---|---|
| `EquationItem` wrapper covering all equation variants | Symmetry for its own sake. No consumer needs docstrings on equations or annotations on non-Connect variants today. |
| Lifting `For.equations` / `If.else_block` / `EquationBlock.eqs` to the wrapper | Same reason. Array-connect inside a `for` with per-index annotations is theoretically legal Modelica, practically never authored. |
| Grammar `nt_type some_equation = EquationItem` change | Would trigger regeneration of `crates/rumoca-phase-parse/src/generated/`. Not required if we populate at the IR conversion site. |
| `Visitor::visit_equation_item` + breaking `visit_equation` | Trait surface change with no payoff — the typed annotation API is consumed outside the visitor pattern. |
| Normalizing `Component.annotation: Vec<Expression>` → `Option<Annotation>` | Would cascade through every annotation call site in the codebase. The typed API's `from_modifications(&[Expression])` signature abstracts over both shapes so consumers don't care. Leave normalization for a dedicated cleanup later. |
| Docstring preservation on `Equation::Connect` alongside annotation | Not consumed by any tool in the workspace. Add when someone asks. |

## The plan (as committed)

1. **`Equation::Connect { lhs, rhs, annotation: Option<Annotation> }`**, populated at parse from `SomeEquation.description.description_opt.annotation_clause.class_modification.class_modification_opt.argument_list.args`. Source: 445d177.
2. **`InstanceConnection.annotation: Option<Annotation>`**, populated at the `Connect → InstanceConnection` conversion in `crates/rumoca-phase-flatten/src/connections/equation_generation.rs`. Array-connect expansions inherit the same annotation (documented invariant — all expanded wires render with the same authored route).
3. **Typed annotation API** in a new `crates/rumoca-ir-ast/src/annotation.rs`:
   - `find_modification(&[Expression], &str) -> Option<&Expression>`
   - `find_modification_path(&[Expression], &[&str])`
   - `graphical::{Placement, Transformation, Line, Rectangle, Polygon, Text, Icon, CoordinateSystem, Point, Extent, Color}` with `from_modifications(&[Expression]) -> Option<Self>` constructors.
   - Accessors: `Component::placement()`, `ClassDef::icon()`, `ClassDef::diagram_layer()`.
   - Shaped to accept `&[Expression]` so it works unchanged on `Component.annotation`, `ClassDef.annotation`, and `Annotation.modifications`.

Downstream lunco-modelica migration lands in a **separate PR** after the
rumoca dep bumps: kill category A + B regexes, replace `msl_indexer.rs`
Debug-format scraping with typed graphical accessors (kills category C).

## Grammar navigation (for future edits)

The parser uses parol (`build.rs` regenerates on `modelica.par` change).
Path from `SomeEquation` to per-equation metadata:

```
SomeEquation.description                        // modelica_grammar_trait::Description
  .description_string.tokens                    // Vec<Token> — docstring
  .description_opt                              // Option<DescriptionOpt>
    .annotation_clause                          // AnnotationClause
      .annotation.annotation                    // literal "annotation" keyword token (Location source)
      .class_modification                       // ClassModification
        .class_modification_opt                 // Option<ClassModificationOpt>
          .argument_list.args                   // Vec<Expression> — Annotation.modifications
```

Existing parser helpers to reuse:
- `crates/rumoca-phase-parse/src/elements.rs:141` — `extract_annotation` for `Component.annotation` shape (returns `Vec<Expression>`).
- `crates/rumoca-phase-parse/src/elements.rs:520`, `definitions.rs:66,602,667,905` — docstring extraction examples.

`Annotation` struct lives at `crates/rumoca-ir-ast/src/lib.rs:1622-1627`.
Before this work, it was defined but unused — `Component.annotation` and
`ClassDef.annotation` both use raw `Vec<Expression>`. Connect's
`Option<Annotation>` is the first live user; the typed API accommodates
both shapes.

## Propagation paths (for future edits that need to track something through phases)

Equation lifecycle, useful reference for anyone adding a new equation-level
field:

```
source
  └─ parol grammar: SomeEquation (has description)
      └─ TryFrom<&SomeEquation> for Equation   [crates/rumoca-phase-parse/src/equations.rs:113]
          └─ class.equations: Vec<Equation>    [crates/rumoca-ir-ast/src/lib.rs:689]
              └─ equations_to_instance_cloned  [crates/rumoca-phase-instantiate/src/lib.rs:863]
                  └─ InstanceEquation          [crates/rumoca-ir-ast/src/instance.rs:495]
                      └─ flatten's extract_connections_from_equation
                          └─ InstanceConnection [crates/rumoca-ir-ast/src/instance.rs:519]
```

For a class-level equation field (description/annotation/metadata), the
write sites are:
- Parser: ~1 site (`equations.rs` `TryFrom` impl).
- Instantiate: 2 sites (`equations_to_instance_cloned`, `equations_to_instance_without_connections` at `lib.rs:863,890`).
- Flatten: `connections/equation_generation.rs:649` + `conditional_and_eval.rs:925,1461,1497` + `vcg.rs:1015-1029` (synthesized, default-None).
- Tests: `phase-instantiate/src/tests.rs`, `phase-instantiate/src/connections.rs:993,1023,1105`, `phase-flatten/src/connections/tests.rs:30,50`.

## Open questions left for later

- When should `Component.annotation` / `ClassDef.annotation` normalize to `Annotation` struct? Not now; the typed API hides the inconsistency. Revisit if a third `Vec<Expression>` annotation appears, or when cleanup PRs stack up.
- When should equation docstrings be preserved? When a consumer needs them. Until then, the parser drops them. Trivial to add alongside the annotation if asked.
- A Modelica source printer would eliminate category D regexes. Out of scope here; large independent initiative.

## Key references

- Modelica Language Specification 3.6, §3.1 (comments), §8 (equations), §18 (annotations).
- OMEdit source — OpenModelica's `OMEdit/ClientManager/*ConnectionLineAnnotation*` for connect-rendering reference behaviour.
- `crates/rumoca-phase-parse/src/modelica.par` — parol grammar with `%nt_type` directives.
- `crates/rumoca-phase-parse/build.rs` — automatic parser regeneration.
