use super::*;

#[test]
fn source_class_query_qualifies_exact_nested_and_quoted_document_paths() {
    let mut session = Session::default();
    session
        .add_document(
            "inline.mo",
            "within 'Source.Space'; package Box model 'Part.Dot' end 'Part.Dot'; end Box;",
        )
        .unwrap();
    for name in ["Box.'Part.Dot'", "'Source.Space'.Box.'Part.Dot'"] {
        assert_eq!(
            session.qualify_model_name("inline.mo", name),
            "'Source.Space'.Box.'Part.Dot'"
        );
    }
    assert_eq!(
        session.qualify_model_name("inline.mo", "Other.Model"),
        "Other.Model"
    );
}

#[test]
fn source_class_query_reuses_declaration_enumeration_for_session_completion() {
    let mut session = Session::default();
    session
        .add_document(
            "inline.mo",
            "within Source; package Box model Part end Part; end Box;",
        )
        .unwrap();
    let names = session.all_class_names_for_completion().unwrap();
    assert!(names.contains(&"Source.Box".to_owned()));
    assert!(names.contains(&"Source.Box.Part".to_owned()));
}
