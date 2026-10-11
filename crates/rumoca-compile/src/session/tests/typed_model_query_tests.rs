use super::*;

fn assert_typecheck_failure(result: PhaseResult, expected_code: &str) {
    match result {
        PhaseResult::Failed {
            phase, error_code, ..
        } => {
            assert_eq!(phase, FailedPhase::Typecheck);
            assert_eq!(error_code.as_deref(), Some(expected_code));
        }
        other => panic!("expected typecheck failure, got {other:?}"),
    }
}

#[test]
fn typed_model_query_cache_is_reused_by_compile_and_diagnostics() {
    let mut session = Session::default();
    session
        .add_document(
            "target.mo",
            "model Target\n  Real x(startd = 1.0);\nequation\n  der(x) = -x;\nend Target;\n",
        )
        .expect("target should parse");
    session
        .add_document("other.mo", "model Other\n  Real y;\nend Other;\n")
        .expect("other should parse");

    let first = session
        .compile_model_phases("Target")
        .expect("compile should return phase result");
    assert_typecheck_failure(first, "ET001");

    let cache_key = standard_typed_cache_key(&mut session, "Target");
    match session
        .query_state
        .flat
        .typed_models
        .artifacts
        .get(&cache_key)
        .expect("typed query should be cached")
        .outcome
        .clone()
    {
        TypedModelOutcome::TypecheckError(diags) => {
            assert!(
                diags
                    .iter()
                    .any(|diag| diag.code.as_deref() == Some("ET001")),
                "initial typed artifact should preserve the real typecheck failure"
            );
        }
        other => panic!("expected cached typecheck error, got {other:?}"),
    }

    session
        .query_state
        .flat
        .typed_models
        .artifacts
        .get_mut(&cache_key)
        .expect("typed query should be cached")
        .outcome = TypedModelOutcome::TypecheckError(vec![CommonDiagnostic::error(
        "TTEST",
        "typed cache sentinel",
        PrimaryLabel::new(Span::DUMMY).with_message("cache sentinel"),
    )]);

    let parse_error = session.update_document(
        "other.mo",
        "model Other\n  Real y;\n  Real z;\nend Other;\n",
    );
    assert!(parse_error.is_none(), "unrelated edit should remain valid");

    session
        .query_state
        .dae
        .compile_results
        .shift_remove("Target");
    let dae_cache_key = standard_dae_cache_key(&mut session, "Target");
    session
        .query_state
        .dae
        .dae_models
        .shift_remove(&dae_cache_key);
    let flat_cache_key = standard_flat_cache_key(&mut session, "Target");
    session
        .query_state
        .flat
        .flat_models
        .artifacts
        .shift_remove(&flat_cache_key);

    let warm_compile = session
        .compile_model_phases("Target")
        .expect("warm compile should return phase result");
    assert_typecheck_failure(warm_compile, "TTEST");

    let warm_diagnostics = session.compile_model_diagnostics("Target");
    assert_diagnostics_have_code(&warm_diagnostics, "TTEST");

    let parse_error = session.update_document(
        "target.mo",
        "model Target\n  Real x(start = 1.0);\nequation\n  der(x) = -x;\nend Target;\n",
    );
    assert!(parse_error.is_none(), "target edit should remain valid");

    let rebuilt = session
        .compile_model_phases("Target")
        .expect("rebuild should return phase result");
    assert!(
        matches!(rebuilt, PhaseResult::Success(_)),
        "target edit should rebuild the typed model instead of reusing the stale typecheck failure"
    );

    let rebuilt_diagnostics = session.compile_model_diagnostics("Target");
    assert_diagnostics_lack_code(&rebuilt_diagnostics, "TTEST");
}

#[test]
fn class_definition_query_preserves_exact_within_and_quoted_identity() {
    let mut session = Session::default();
    session
        .add_document(
            "within.mo",
            "within Space; package Root model 'A.B' Real marker; end 'A.B'; end Root;",
        )
        .expect("valid source");
    let selected = session
        .class_definition_query("Space.Root.'A.B'")
        .expect("exact qualified declaration");
    assert_eq!(selected.name.text.as_ref(), "'A.B'");
    assert!(selected.components.contains_key("marker"));
    let selected_ptr = selected as *const ast::ClassDef;
    let parsed = session
        .parsed_file_query("within.mo")
        .expect("parsed document");
    assert_eq!(
        selected_ptr,
        parsed
            .get_class_by_qualified_name("Space.Root.'A.B'")
            .expect("same declaration") as *const ast::ClassDef
    );
    assert!(parsed.get_class_by_qualified_name("Root.'A.B'").is_some());
    for invalid in [
        "Other.Space.Root.'A.B'",
        "Space.Root..'A.B'",
        "Space.Root.'A.B'.",
        "Space.Root.A.B",
    ] {
        assert!(
            session.class_definition_query(invalid).is_none(),
            "{invalid}"
        );
    }
}

#[test]
fn class_definition_query_rejects_ambiguous_bare_targets_and_tracks_revision() {
    let mut session = Session::default();
    session.add_document("roots.mo", "package A model Shared Real first; end Shared; end A; package B model Shared Real second; end Shared; end B;").expect("valid source");
    assert!(session.class_definition_query("Shared").is_none());
    assert!(
        session
            .class_definition_query("A.Shared")
            .expect("A")
            .components
            .contains_key("first")
    );
    assert!(
        session
            .class_definition_query("B.Shared")
            .expect("B")
            .components
            .contains_key("second")
    );
    assert!(
        session
            .update_document(
                "roots.mo",
                "package B model Shared Real edited; end Shared; end B;"
            )
            .is_none()
    );
    assert!(session.class_definition_query("A.Shared").is_none());
    assert!(
        session
            .class_definition_query("Shared")
            .expect("unique current declaration")
            .components
            .contains_key("edited")
    );
    session.remove_document("roots.mo");
    assert!(session.class_definition_query("B.Shared").is_none());
}

#[test]
fn class_definition_query_tracks_preparsed_source_set_replacement() {
    let mut session = Session::default();
    session.replace_in_memory_source_set(
        "library",
        SourceRootKind::DurableExternal,
        vec![parse_source_document(
            "within Library; package P model Leaf Real original; end Leaf; end P;",
            "library.mo",
        )],
        None,
    );
    assert!(
        session
            .class_definition_query("Library.P.Leaf")
            .expect("installed parsed source")
            .components
            .contains_key("original")
    );
    session.replace_in_memory_source_set(
        "library",
        SourceRootKind::DurableExternal,
        vec![parse_source_document(
            "within Library; package P model Leaf Real replacement; end Leaf; end P;",
            "library.mo",
        )],
        None,
    );
    let current = session
        .class_definition_query("Library.P.Leaf")
        .expect("replacement declaration");
    assert!(current.components.contains_key("replacement"));
    assert!(!current.components.contains_key("original"));
}

#[test]
fn class_definition_query_component_declarations_are_typed_and_complete() {
    let mut session = Session::default();
    session.add_document("members.mo", "within Space; package P model Base parameter Real k=2; input Real u; end Base; model Derived extends Base; output Real y; end Derived; model Broken extends Missing; Real local; end Broken; model Cyclic extends Cyclic; end Cyclic; model Empty end Empty; type Scalar = Real; connector ScalarInput = input Scalar; end P;").expect("valid source");
    let members = session
        .class_component_declarations_query("Space.P.Derived")
        .expect("resolved declarations");
    assert_eq!(members.len(), 3);
    let (origin, _, parameter) = members
        .iter()
        .find(|(_, _, c)| c.name == "k")
        .expect("inherited parameter");
    assert_eq!(origin, "Space.P.Base");
    assert!(matches!(
        parameter.variability,
        rumoca_core::Variability::Parameter(_)
    ));
    assert_eq!(
        parameter
            .binding
            .as_ref()
            .map(ToString::to_string)
            .as_deref(),
        Some("2")
    );
    let (_, _, input) = members
        .iter()
        .find(|(_, _, c)| c.name == "u")
        .expect("inherited input");
    assert!(matches!(input.causality, rumoca_core::Causality::Input(_)));
    for invalid in ["Space.P.Broken", "Space.P.Cyclic", "Space.P.Unknown"] {
        assert!(
            session
                .class_component_declarations_query(invalid)
                .is_none(),
            "{invalid}"
        );
    }
    assert!(
        session
            .class_component_declarations_query("Space.P.Empty")
            .expect("valid empty class")
            .is_empty()
    );
    for primitive in ["Space.P.Scalar", "Space.P.ScalarInput"] {
        assert!(
            session
                .class_component_declarations_query(primitive)
                .expect("primitive-derived declaration has no named members")
                .is_empty()
        );
    }
    let mut parsed = session
        .parsed_file_query("members.mo")
        .expect("parsed file")
        .clone();
    parsed
        .get_class_by_qualified_name_mut("Space.P.Base")
        .expect("exact mutable declaration")
        .components
        .clear();
    assert!(
        parsed
            .get_class_by_qualified_name_mut("Other.Space.P.Base")
            .is_none()
    );
}
