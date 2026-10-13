# LunCoSim runtime owner verification

Based on official v0.10.2, commit 276e82bd0fd48960ac201e782918d28bf19cf7ce.
The additional host capabilities keep checked FMI construction and ME session
execution authoritative: immutable prepared-component borrowing, canonical
construction-wire persistence, exact compiler source contributions, paired
source/syntax bundle records, cooperative cancellation and retained-output
admission. Optional timing uses rumoca-core's instrumentation owner.

Focused checks used nightly-2026-02-27, four Cargo jobs, four test threads and
four Rayon threads:

- `cargo test -p rumoca-eval-solve -p rumoca-solver --lib`: 240 evaluator and
  550 solver tests passed.
- `cargo test -p rumoca-compile --lib successful_strict_compile_carries_only_reachable_source_contributions`:
  passed; unrelated document bytes are excluded.
- `cargo test -p rumoca-solver --lib admitted_output_count_matches_exact_grid_and_rejects_invalid_requests`:
  passed after restoring fallible output-vector allocation.
- LunCoSim's checked-component persistence tests passed, including malformed
  construction-wire rejection, bounded decoding and directory-quota publication.

## Fixed canary delta

Both unmodified v0.10.2 and these owner changes ran:
`cargo xtask verify msl-parity --sim-targets-file infra/verification/msl-canary-20.json`.
MSL was the official v4.1.0 release; the comparator was
OpenModelica 1.27.1~2-g6db4671. Each run used the same fixed twenty targets.

| Observed boundary | Official v0.10.2 | Host capability changes |
|---|---:|---:|
| Parsed and flattened targets | 20 | 20 |
| Checked DAE targets | 14 | 14 |
| Simulated targets | 12 | 12 |
| Initialization successes | 12 | 12 |
| Trace comparator strict-high target identities | 12 | 12 |
| Additional unexcepted non-high targets | 0 | 0 |

The successful target identities were identical. Six targets failed before
checked DAE construction. Two more reported typed solve-IR failures:
`Modelica.Electrical.Analog.Examples.OpAmps.OpAmpCircuits.Add` was structurally
singular and `Modelica.Electrical.Polyphase.Examples.TransformerYD` had a
non-numeric affine derivative coefficient. No new failure was introduced.
This is a focused canary delta, not a full-cohort parity measurement.

## Static variability inside input components

Checked DAE role construction gives constant and parameter variability precedence
over inherited input causality. A nested parameter remains a static coordinate;
its enclosing input component cannot turn it into an external runtime slot.
The focused checked-preparation test evaluates a declaration default depending
on that nested parameter, a function default argument, and an array input.

After the static-variability correction, the same fixed canary again parsed all
2555 sources, compiled fourteen targets and initialized/simulated twelve. The
compared model identities and agreement bands remained identical to official
v0.10.2. The only feature-specific build warning was corrected by gating the ME
artifact import on its actual RK45 consumer.

## Source class query ownership

`rumoca-core` owns quote/bracket-aware path boundaries.
`StoredDefinition` owns exact read and mutable class traversal and normalization
of its authored `within` package. `Session::class_definition_query` borrows the
current indexed declaration; `class_component_declarations_query` returns a
complete scope-resolved member set with declaring identity and typed source
facts. Unresolved or cyclic inheritance has no complete result. Primitive
bases have no named component members.

The focused core boundary tests passed (2), and the session class query tests
passed (4). The existing membership/cache tests also passed (3) using the
same built test binary. Coverage includes quoted identities, exact package boundaries, source edits,
document removal, pre-parsed library replacement, inherited typed declarations,
primitive-derived connectors, missing bases, and inheritance cycles.
Commands used `--locked`, four Cargo jobs, four test threads and four Rayon
threads:

- `cargo test -p rumoca-core --lib qualified_name_boundary_tests -j4`
- `cargo test -p rumoca-compile --lib class_definition_query -j4`

The final same-target canary command was
`cargo run -p xtask --bin xtask -j4 -- verify msl-parity --sim-targets-file infra/verification/msl-canary-20.json --results-dir target/msl/class-lookup-current`.
The baseline was `87df7f6bcaafc4aace0e875497bb731f8f50f220`;
the fixed target file SHA-256 was
`536004933ce502f7a8a4317e5dd6de2b700a6b176e84d2d6e7bbe0dcdf61eafa`.
Both used MSL v4.1.0 and OpenModelica 1.27.1~2-g6db4671.

| Observed boundary | Baseline | Current source queries |
|---|---:|---:|
| Parsed and flattened targets | 20 | 20 |
| Checked DAE / solve IR targets | 14 | 14 |
| Initialized and simulated targets | 12 | 12 |
| Strict-high trace target identities | 12 | 12 |
| Refused before simulation | 6 | 6 |
| Typed solver failures | 2 | 2 |

All twenty per-model band rows were identical. No additional failure or weaker
trace band appeared. This remains a focused canary, not a full-cohort claim.
The cached OMC reference run did not reproduce its twelve initialization XML
files in the new results directory, so the current harness reported zero
state-selection comparisons. Direct comparison of the twelve before/after
Rumoca trace state metadata sets found them all unchanged. Timings came from
concurrent builds and the live application; they do not establish a performance
improvement.

## Original UTF-8 source ownership

`rumoca-phase-parse` admits a single file-prefix UTF-8 BOM through its grammar
(MLS 13.4, SPEC_0022 LEX-014). Parser tokens, recovery and diagnostics address
the original bytes (SPEC_0008). The grammar rejects misplaced or repeated marks;
marks in strings/comments are source content. CR, CRLF and LF preserve source
coordinates. Hosts pass the unmodified UTF-8 text to strict and recovery APIs.

The scanner uses `scnr2` 0.5.3 with the matched `parol` 4.7.0 and
`parol_runtime` 4.5.0. The scanner maintainer's Unicode literal fix is the
producer correction; no host rewrite is part of parsing or editor paste.

The baseline failed two focused encoding assertions, and LunCoSim's production
batch compiler rejected a valid BOM-prefixed document with EP001. Five encoding
checks, all 61 remaining Modelica parser tests and 25 GALEC parser tests passed
with the corrected dependency and grammar. Verification used the same four-job
repository toolchain and regular sccache.

The unchanged twenty-target MSL canary parsed/flattened all targets, admitted
14 checked DAE models and simulated 12. All twenty per-model agreement bands
and exit-reason classifications match the source-class-query evidence commit
`e773472180ded10dc6aab0e3f84f99330f8398ae`; twelve retain strict-high agreement.
The six DAE refusals and two EL005 solver refusals remain visible. Results live
under `target/msl/source-encoding-current`, with the same fixed canary SHA-256
`536004933ce502f7a8a4317e5dd6de2b700a6b176e84d2d6e7bbe0dcdf61eafa`.
This is bounded canary evidence; cached OMC references do not provide fresh
initialization XML or a new state-selection comparison.

## Canonical source declaration queries

`StoredDefinition` owns structural declaration lookup, canonical names and
declaration enumeration (SPEC_0029 section 3; SPEC_0041; MLS 13.4.3). Exact
qualified or document-relative paths preserve nested and quoted identifiers.
Unknown paths have no canonical declaration. Source-tree enumeration includes
the authored `within` prefix and all declaration kinds without resolving imports
or inherited members. Session qualification and completion consume these readers.

Three parser-owned source-query checks and two session-consumer checks passed
through `cargo test --locked -p rumoca-phase-parse -p rumoca-compile --lib source_class_query -j4`.
These are focused structural-query checks, not a solver-cohort claim.

The fixed twenty-target canary in `target/msl/source-admission-current` parsed
and flattened all twenty, admitted fourteen checked DAE models and simulated
twelve. All twenty model identities, agreement bands and exit classifications
match `target/msl/source-encoding-current` at `b349ebff0fc47225dd070a8020158688b4318ed2`:
twelve strict-high traces, six DAE refusals and two solver refusals remain
visible. The canary working-tree digest is
`ae8d95a365068915887bd00af268fc29308dc60a51ff07af529674d6cdf46e69`.
The target SHA-256 is
`536004933ce502f7a8a4317e5dd6de2b700a6b176e84d2d6e7bbe0dcdf61eafa`.
Cached OMC references do not provide fresh initialization XML or a new
state-selection comparison. This is Tier 1 canary evidence, not a full-cohort
or performance claim.

## Complete authored class extents

The parser retains declaration semicolons and combines their locations with
class and element qualifier tokens (SPEC_0008; MLS Annex A.2.1/A.2.2).
`ClassDef.location` is the complete declaration, including a constraining
clause for a replaceable element. Name-token spans preserve their separate
rename contract. Autodiff inserts generated functions at that exact endpoint.

Four focused class-span checks failed before the producer correction. The
corrected parser passed all 73 parser checks. All 64 autodiff checks, including
generation after a comment before a declaration semicolon, and all fourteen
dependency-fingerprint checks passed. Seven focused session navigation/body
consumer checks also passed, preserving separate identifier locations.
Verification used four jobs and threads.

The fixed twenty-target canary in `target/msl/class-source-span-current`
retained all twenty agreement-band rows and phase/failure classifications from
`target/msl/source-admission-current`: fourteen checked DAE models and twelve
strict-high simulated traces. Six DAE refusals and two solver refusals remain
visible. The working-tree digest is
`a22ec14983e2c8a5ce0faf34894866c02a334febe681cdf9364fd471e7763eff`.
The fixed target SHA-256 is
`536004933ce502f7a8a4317e5dd6de2b700a6b176e84d2d6e7bbe0dcdf61eafa`.
This is Tier 1 canary evidence. Cached OMC references do not establish fresh
initialization XML or state-selection parity, and concurrent timings do not
establish a performance improvement.

## Enclosing package imports

`within_import_resolves_enclosing_package_alias_in_both_compile_paths` passes
for `Session::compile_model` and `Session::compile_model_strict`, with parent
and child documents admitted in both orders. A child declared with `within`
resolves a type alias imported by its enclosing package through Rumoca's normal
lexical scope. Hosts must retain the admitted parent source in the compiler
session used by that child.

The focused command was
`cargo test --locked -p rumoca-compile --lib within_import -j4`.

## Scoped declaration type facts

`Session::class_component_declarations_query` returns immutable
`ClassComponentDeclaration` records with authored components kept separate from
effective causality, primitive type, and type-dimension facts. Aliases use the
shared lexical/import candidate query, including inherited nested classes and
package aliases. Encapsulation stops outward scope lookup; an unrelated unique
suffix cannot satisfy a type reference. The declaration result is cached by
session revision under the existing semantic-navigation cache bound.

`cargo test --locked -p rumoca-compile --lib class_ -j4` passed 51 checks,
including alias chains, records with signal-looking names, alias dimensions,
parent imports, encapsulation, inherited types, package aliases, negative
missing/cyclic types, lexical/import head shadowing, shared result reuse,
and source-revision invalidation.

## Qualified declaration identities

MLS 5.3 and SPEC_0002 govern type lookup. `ClassComponentDeclaration.type_name`
now publishes the exact identity selected by the scoped type-prefix query.
`component.type_name` retains the authored spelling. Built-in types retain their
primitive names. This changes source-query output only; checked compilation and
solver execution do not consume this presentation field.

`cargo test --locked -p rumoca-compile --lib typed_model_query_tests -j 4`
passed all 11 focused tests with the standard four-thread budgets. Coverage
includes an import alias, inherited members, two different packages declaring
`Port`, exact declaring scope, and unchanged authored spelling.

A declaration-query canary used the fixed `infra/verification/msl-canary-20.json`
roster and MSL 4.1.0. All twenty queries resolved; 125 declarations published a
qualified identity different from their authored spelling. Every non-builtin
identity resolved back to itself. Cold queries, including lazy source hydration,
took 116.141 seconds; repeated cached queries took 20.748 microseconds. This is
query coverage and a bounded cache observation, not a numerical canary delta or
an uncontended performance claim. The temporary canary probe was removed.

## Namespace membership cache

Qualified declaration lookup reuses a package membership map validated at the
current session revision. It computes the full source-set signature when the
revision changes. Cache adoption from another session clears the local
validation stamp, so equal revision numbers cannot authorize foreign membership.
Source-root invalidation retains its existing cache retirement boundary.

Focused namespace, package-membership, and source-root cache checks cover
edits, removals, split packages, source-set revisions, and foreign-cache adoption.

## Git dependency numerical closure

The Diffsol backend directly pins the corrected source retained in this repository.
Cargo ignores dependency-workspace patches in a downstream application, so the
backend manifest owns this dependency edge. Git consumers now execute the same
accepted-state and Newton corrections as native workspace validation.
