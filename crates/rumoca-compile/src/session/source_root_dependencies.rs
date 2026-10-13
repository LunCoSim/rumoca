use std::collections::BTreeSet;

use super::FileId;
use super::class_interface::{
    FileClassInterfaceIndex, LookupKind, external_type_resolution_candidates,
    type_resolution_candidates,
};
use super::file_summary::{FileSummary, ImportSummary};
use rumoca_ir_ast as ast;

/// External namespace roots required by a parsed document's declarations.
///
/// Uses the session's lexical/import lookup for nested classes, component types,
/// extends and constraints. A `within` clause places declarations in a namespace;
/// it does not itself request a library. This admission query does not replace
/// full resolution: unresolved members are diagnosed when the closure is compiled.
pub fn source_root_dependencies(definition: &ast::StoredDefinition) -> BTreeSet<String> {
    let summary = FileSummary::from_definition(FileId::default(), definition);
    let interfaces = FileClassInterfaceIndex::from_summary(&summary);
    let mut roots = BTreeSet::new();
    for (key, class) in summary.iter() {
        let enclosing = key.qualified_name();
        for import in &class.imports {
            let (path, global_scope) = match import {
                ImportSummary::Qualified { path, global_scope }
                | ImportSummary::Renamed {
                    path, global_scope, ..
                }
                | ImportSummary::Unqualified { path, global_scope }
                | ImportSummary::Selective {
                    path, global_scope, ..
                } => (path, global_scope),
            };
            let absolute;
            let path = if *global_scope {
                absolute = format!(".{path}");
                &absolute
            } else {
                path
            };
            collect_roots(
                &interfaces,
                &enclosing,
                path,
                LookupKind::Type,
                true,
                &mut roots,
            );
        }
        for base in &class.extends {
            collect_roots(
                &interfaces,
                &enclosing,
                &base.base_name,
                LookupKind::BaseClass,
                false,
                &mut roots,
            );
        }
        for component in class.components.values() {
            collect_roots(
                &interfaces,
                &enclosing,
                &component.type_name,
                LookupKind::Type,
                false,
                &mut roots,
            );
            if let Some(constraint) = &component.constrainedby {
                collect_roots(
                    &interfaces,
                    &enclosing,
                    constraint,
                    LookupKind::Type,
                    false,
                    &mut roots,
                );
            }
        }
        if let Some(constraint) = &class.constrainedby {
            collect_roots(
                &interfaces,
                &enclosing,
                constraint,
                LookupKind::Type,
                false,
                &mut roots,
            );
        }
    }
    roots
}

fn collect_roots(
    interfaces: &FileClassInterfaceIndex,
    enclosing: &str,
    name: &str,
    kind: LookupKind,
    include_unqualified: bool,
    roots: &mut BTreeSet<String>,
) {
    if rumoca_core::is_builtin_type(name) {
        return;
    }
    let lookup = |candidate: &str| interfaces.class_interface(candidate).cloned();
    if type_resolution_candidates(enclosing, name, kind, lookup)
        .iter()
        .any(|candidate| interfaces.class_interface(candidate).is_some())
    {
        return;
    }
    for candidate in external_type_resolution_candidates(enclosing, name, kind, lookup) {
        let mut segments = rumoca_core::qualified_name_segments(&candidate);
        if let Some(root) = segments.next()
            && (include_unqualified || segments.next().is_some())
            && !rumoca_core::is_builtin_type(root)
        {
            roots.insert(root.to_owned());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dependencies(source: &str) -> BTreeSet<String> {
        let ast = rumoca_phase_parse::parse_to_ast(source, "inline.mo").unwrap();
        source_root_dependencies(&ast)
    }

    #[test]
    fn nested_aliases_request_the_import_target() {
        assert_eq!(
            dependencies(
                "within RunSpace; package P model M import Units = Library.Units; Units.Temperature t; end M; end P;"
            ),
            BTreeSet::from(["Library".to_owned()])
        );
    }

    #[test]
    fn local_and_quoted_names_do_not_request_libraries() {
        assert!(
            dependencies(
                "within RunSpace; package P type 'A.B' = Real; model M P.'A.B' x; end M; end P;"
            )
            .is_empty()
        );
        assert!(dependencies("within RunSpace; model M Real x; end M;").is_empty());
        assert!(dependencies("package P model M P.Missing x; end M; end P;").is_empty());
        assert!(
            dependencies("package P type T = Real; model M import U = P; U.T x; end M; end P;")
                .is_empty()
        );
    }

    #[test]
    fn import_forms_and_missing_qualified_types_keep_external_roots() {
        assert_eq!(
            dependencies(
                "model M import Bare; import A.B.*; import C.D.{T,U}; import .Global.T; Missing.Type x; end M;"
            ),
            BTreeSet::from(["A", "Bare", "C", "Global", "Missing"].map(str::to_owned))
        );
    }
}
