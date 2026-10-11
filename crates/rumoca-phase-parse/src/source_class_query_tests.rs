use crate::parse_to_ast;

#[test]
fn canonical_identity_preserves_nested_and_quoted_segments() {
    let definition = parse_to_ast(
        "within 'Source.Space'; package Box model 'Part.Dot' end 'Part.Dot'; end Box;",
        "inline.mo",
    )
    .unwrap();
    for request in ["Box.'Part.Dot'", "'Source.Space'.Box.'Part.Dot'"] {
        assert_eq!(
            definition.qualified_class_name(request).as_deref(),
            Some("'Source.Space'.Box.'Part.Dot'")
        );
    }
    assert_eq!(
        definition.declared_class_names(),
        vec!["'Source.Space'.Box", "'Source.Space'.Box.'Part.Dot'"]
    );
}

#[test]
fn unknown_and_invalid_source_paths_have_no_canonical_identity() {
    let definition = parse_to_ast("within Source; model Part end Part;", "inline.mo").unwrap();
    for request in [
        "",
        "Source..Part",
        ".Part",
        "Other.Part",
        "Sources.Part",
        "Part.More",
    ] {
        assert_eq!(definition.qualified_class_name(request), None, "{request}");
    }
    assert_eq!(
        definition.qualified_class_name("Part").as_deref(),
        Some("Source.Part")
    );
}

#[test]
fn declared_names_include_all_class_kinds_in_source_tree_order() {
    let definition = parse_to_ast(
        "within; package P partial model M end M; connector C = Real; end P; model Last end Last;",
        "inline.mo",
    )
    .unwrap();
    assert_eq!(
        definition.declared_class_names(),
        vec!["P", "P.M", "P.C", "Last"]
    );
    assert_eq!(
        definition.qualified_class_name("P.M").as_deref(),
        Some("P.M")
    );
}
