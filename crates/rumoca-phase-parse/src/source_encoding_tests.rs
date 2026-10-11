use super::{parse_to_ast, parse_to_recovered_ast, parse_to_syntax};

#[test]
fn leading_bom_keeps_original_source_coordinates() {
    let source = "\u{feff}within Space;\r\nmodel 'R.1' \"café\"\r\n  Real value;\r\nend 'R.1';";
    let ast = parse_to_ast(source, "encoding.mo").expect("leading UTF-8 BOM is file metadata");
    assert_eq!(ast.within.as_ref().unwrap().to_string(), "Space");
    let class = ast.get_class_by_qualified_name("Space.'R.1'").unwrap();
    let start = source.find("'R.1'").unwrap();
    assert_eq!(class.name.location.start as usize, start);
    assert_eq!(class.name.location.start_line, 2);
    assert_eq!(&source[start..class.name.location.end as usize], "'R.1'");
    assert_eq!(class.components["value"].name_token.location.start_line, 3);
    assert!(!parse_to_syntax(source, "encoding.mo").has_errors());
}

#[test]
fn bom_is_only_admitted_at_the_file_start() {
    for source in [
        " \u{feff}model A end A;",
        "\u{feff}\u{feff}model A end A;",
        "model A \u{feff} Real x; end A;",
        "model A end A;\u{feff}",
    ] {
        assert!(
            parse_to_syntax(source, "encoding.mo").has_errors(),
            "{source:?}"
        );
    }
}

#[test]
fn bom_in_string_and_comment_is_source_content() {
    let source = "model A \"\u{feff}\" // \u{feff}\r\n Real x; end A;";
    assert!(parse_to_ast(source, "encoding.mo").is_ok());
}

#[test]
fn recovered_bom_source_retains_offsets_and_diagnostics() {
    let source = "\u{feff}model A\r\n Real x\r\nend A;";
    let syntax = parse_to_syntax(source, "encoding.mo");
    assert!(syntax.has_errors());
    assert!(
        syntax
            .parse_errors()
            .iter()
            .all(|error| error.span().start.0 >= 3)
    );
    let ast = parse_to_recovered_ast(source, "encoding.mo");
    let class = ast.get_class_by_qualified_name("A").unwrap();
    assert_eq!(
        class.name.location.start as usize,
        source.find("A").unwrap()
    );
    assert_eq!(class.name.location.start_line, 1);
}

#[test]
fn source_line_endings_keep_byte_coordinates() {
    for ending in ["\n", "\r\n", "\r"] {
        let source = format!("model A \"café\"{ending}Real value;{ending}end A;");
        let ast =
            parse_to_ast(&source, "encoding.mo").expect("Modelica whitespace accepts line endings");
        let token = &ast.classes["A"].components["value"].name_token;
        assert_eq!(token.location.start as usize, source.find("value").unwrap());
        assert_eq!(
            &source[token.location.start as usize..token.location.end as usize],
            "value"
        );
    }
}
