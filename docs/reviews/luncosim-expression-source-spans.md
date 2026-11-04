# Complete parsed expression spans

LunCoSim reproduced a parameter edit of `gain=2*3` becoming `gain=9.0*3`.
Its splice selected `Expression::get_location()`, whose contract is the first
token, rather than `Expression::span()`. Correcting that caller exposed a parser
producer defect: stored spans omitted `if`, parentheses, nonempty array
delimiters and trailing subscripts. Replacing those ranges would still leave
authored syntax behind. The source edit owner must consume one complete range.

The parse phase now retains these grammar tokens and builds complete expression
spans at their syntax owner. Component-reference parts stay in the generated
grammar representation until their parent captures the final subscript range;
the AST reference retains its existing representation. Synthetic matrix row
arrays keep row spans while their enclosing constructor owns the brackets.
There is no source scanner, extra parse, or downstream range repair.

The governing contracts are SPEC_0007 Stage 1 (AST source structure), SPEC_0008
source identity/traceability and SPEC_0029 phase ownership. This is source
metadata correction: accepted syntax and Modelica evaluation are unchanged.
`get_location()` keeps its documented first-token contract. The main risk is
dropping syntax or source identity while forming a wrapper range, covered by
exact byte-slice regressions with comments, Unicode, CRLF, arrays, tuples,
conditionals, references and subscripts.

## Verification

`cargo test --locked -p rumoca-phase-parse --lib expression_source_span_tests
-j 4` passed all three exact-span regressions. The remaining parser library
suite passed all 73 tests with `-- --skip expression_source_span_tests`.
`cargo clippy --locked -p rumoca-phase-parse --all-targets -j 4 -- -D warnings`
passed. Normal parser builds regenerated both checked-in generated files.

The fixed 20-model canary ran through `cargo xtask verify msl-parity
--sim-targets-file infra/verification/msl-canary-20.json --results-dir
target/msl/expression-source-spans-canary` with concurrency four. Its canonical
row digest and all counts match the retained fresh-main baseline at `c8fce575`:
`07fb475915d477780bcca32fcb59a9e4268ad71e7d6312bb32184116374a8a6d`.
There are 14 DAE/compiled models, 12 successful simulations and 12 strict-high
OpenModelica 1.27.1 comparisons, zero near/deviation/skipped/missing traces,
six ToDAE refusals and two simulation failures, unchanged. This is a Tier 1
canary delta, not full-cohort evidence or a baseline promotion.

The LunCoSim splice/reparse regression and production lifecycle gate consume
this API at the application owner. Full repository pre-landing coverage has
not been claimed; the separate initialized-parameter ledger records existing
solver Clippy failures outside this parser delta.
