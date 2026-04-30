use super::{ClassDef, DefId, SourceId, SourceMap, StoredDefinition};
use rumoca_ir_ast::ClassType;
use std::{cell::RefCell, collections::HashMap, sync::Arc};

thread_local! {
    static ACTIVE_SEMANTIC_SOURCE_IDS: RefCell<Option<HashMap<String, SourceId>>> = const { RefCell::new(None) };
    static ACTIVE_SEMANTIC_LOOKUP: RefCell<Option<SemanticLookupIndex>> = const { RefCell::new(None) };
}

struct SemanticLookupIndex {
    def_ptr: *const StoredDefinition,
    qualified_names: HashMap<String, Arc<[String]>>,
    /// Multi-candidate index: simple class name → every qualified
    /// path that matches, paired with its `ClassType`. The previous
    /// implementation kept only the first insertion via
    /// `or_insert_with`, which made the global lookup non-
    /// deterministic when multiple classes share a simple name
    /// (e.g. `Modelica.Units.SI.Temperature` (Type) vs
    /// `Modelica.Fluid.Sensors.Temperature` (Model)). Resolving a
    /// record-component type then sometimes returned the Model and
    /// produced a spurious ER023 "record component has type X which
    /// is a model" error. Storing every candidate lets
    /// [`find_class_by_name`] pick the most-likely intended class
    /// (Type/Record/Connector first, then Model/Block/Function/
    /// Package) — a pragmatic stand-in for full MLS §5.3
    /// scope-aware lookup until that lands.
    nested_short_names: HashMap<String, Vec<(Arc<[String]>, ClassType)>>,
    by_def_id: HashMap<DefId, Arc<[String]>>,
}

impl SemanticLookupIndex {
    fn build(def: &StoredDefinition) -> Self {
        let mut lookup = Self {
            def_ptr: def as *const StoredDefinition,
            qualified_names: HashMap::new(),
            nested_short_names: HashMap::new(),
            by_def_id: HashMap::new(),
        };
        let mut path = Vec::new();
        for (name, class) in &def.classes {
            path.push(name.clone());
            lookup.index_qualified_class(class, &path);
            path.pop();
        }
        lookup
    }

    fn index_qualified_class(&mut self, class: &ClassDef, path: &[String]) {
        let qualified_path: Arc<[String]> = path.to_vec().into();
        self.qualified_names
            .insert(path.join("."), qualified_path.clone());
        if path.len() > 1 {
            self.nested_short_names
                .entry(class.name.text.to_string())
                .or_default()
                .push((qualified_path.clone(), class.class_type.clone()));
        }
        if let Some(def_id) = class.def_id {
            self.by_def_id.insert(def_id, qualified_path.clone());
        }
        for (name, nested) in &class.classes {
            let mut nested_path = path.to_vec();
            nested_path.push(name.clone());
            self.index_qualified_class(nested, &nested_path);
        }
    }

    /// Pick the best candidate for a simple-name lookup.
    ///
    /// When several classes share a simple name, prefer Type > Record
    /// > Connector > Package > Block > Model > Function > Operator.
    /// Component types in records and connectors are virtually always
    /// Type aliases or other Records — preferring those resolves the
    /// SI.Temperature vs Fluid.Sensors.Temperature collision (and
    /// every analogous one across the MSL) without a full §5.3 walk.
    fn pick_best_candidate(
        candidates: &[(Arc<[String]>, ClassType)],
    ) -> Option<Arc<[String]>> {
        let preference = |kind: &ClassType| -> u8 {
            match kind {
                ClassType::Type => 0,
                ClassType::Record => 1,
                ClassType::Connector => 2,
                ClassType::Package => 3,
                ClassType::Block => 4,
                ClassType::Model => 5,
                ClassType::Function => 6,
                ClassType::Operator => 7,
                ClassType::Class => 8,
            }
        };
        candidates
            .iter()
            .min_by_key(|(path, kind)| (preference(kind), path.len(), path.join(".")))
            .map(|(p, _)| p.clone())
    }
}

pub(super) struct ActiveSemanticContextGuard;

impl Drop for ActiveSemanticContextGuard {
    fn drop(&mut self) {
        ACTIVE_SEMANTIC_SOURCE_IDS.with(|slot| {
            *slot.borrow_mut() = None;
        });
        ACTIVE_SEMANTIC_LOOKUP.with(|slot| {
            *slot.borrow_mut() = None;
        });
    }
}

pub(super) fn activate_semantic_context(
    def: &StoredDefinition,
    source_map: &SourceMap,
) -> ActiveSemanticContextGuard {
    ACTIVE_SEMANTIC_SOURCE_IDS.with(|slot| {
        *slot.borrow_mut() = Some(source_map.source_ids());
    });
    ACTIVE_SEMANTIC_LOOKUP.with(|slot| {
        *slot.borrow_mut() = Some(SemanticLookupIndex::build(def));
    });
    ActiveSemanticContextGuard
}

fn with_active_lookup<T>(
    def: &StoredDefinition,
    f: impl FnOnce(&SemanticLookupIndex) -> T,
) -> Option<T> {
    ACTIVE_SEMANTIC_LOOKUP.with(|slot| {
        let lookup = slot.borrow();
        let active = lookup.as_ref()?;
        std::ptr::eq(active.def_ptr, def).then(|| f(active))
    })
}

pub(super) fn source_id_for(file_name: &str) -> Option<SourceId> {
    ACTIVE_SEMANTIC_SOURCE_IDS.with(|slot| {
        let ids_ref = slot.borrow();
        let ids = ids_ref.as_ref()?;
        ids.get(file_name).copied()
    })
}

pub(super) fn find_class_by_name<'a>(
    def: &'a StoredDefinition,
    type_name: &str,
) -> Option<&'a ClassDef> {
    if type_name.contains('.') {
        match with_active_lookup(def, |lookup| lookup.qualified_names.get(type_name).cloned()) {
            Some(Some(class)) => return find_class_by_path(def, &class),
            Some(None) => return None,
            None => {}
        }
        return find_class_by_qualified_name(def, type_name);
    }

    if let Some(cls) = def.classes.get(type_name) {
        return Some(cls);
    }

    match with_active_lookup(def, |lookup| {
        lookup
            .nested_short_names
            .get(type_name)
            .and_then(|cands| SemanticLookupIndex::pick_best_candidate(cands))
    }) {
        Some(Some(class)) => return find_class_by_path(def, &class),
        Some(None) => return None,
        None => {}
    }

    for class in def.classes.values() {
        if let Some(found) = find_nested_class_by_name(class, type_name) {
            return Some(found);
        }
    }

    None
}

pub(super) fn find_class_by_def_id(
    def: &StoredDefinition,
    target_def_id: DefId,
) -> Option<&ClassDef> {
    match with_active_lookup(def, |lookup| lookup.by_def_id.get(&target_def_id).cloned()) {
        Some(Some(class)) => return find_class_by_path(def, &class),
        Some(None) => return None,
        None => {}
    }

    for class in def.classes.values() {
        if class.def_id == Some(target_def_id) {
            return Some(class);
        }
        if let Some(found) = find_nested_class_by_def_id(class, target_def_id) {
            return Some(found);
        }
    }

    None
}

fn find_nested_class_by_name<'a>(class: &'a ClassDef, type_name: &str) -> Option<&'a ClassDef> {
    for nested in class.classes.values() {
        if nested.name.text.as_ref() == type_name {
            return Some(nested);
        }
        if let Some(found) = find_nested_class_by_name(nested, type_name) {
            return Some(found);
        }
    }
    None
}

fn find_nested_class_by_def_id(class: &ClassDef, target_def_id: DefId) -> Option<&ClassDef> {
    for nested in class.classes.values() {
        if nested.def_id == Some(target_def_id) {
            return Some(nested);
        }
        if let Some(found) = find_nested_class_by_def_id(nested, target_def_id) {
            return Some(found);
        }
    }
    None
}

fn find_class_by_qualified_name<'a>(
    def: &'a StoredDefinition,
    type_name: &str,
) -> Option<&'a ClassDef> {
    let mut current = def.classes.get(type_name.split('.').next()?)?;
    for part in type_name.split('.').skip(1) {
        current = current.classes.get(part)?;
    }
    Some(current)
}

fn find_class_by_path<'a>(def: &'a StoredDefinition, path: &[String]) -> Option<&'a ClassDef> {
    let mut current = def.classes.get(path.first()?)?;
    for part in path.iter().skip(1) {
        current = current.classes.get(part.as_str())?;
    }
    Some(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rumoca_ir_ast as ast;
    use rumoca_phase_parse::parse_to_ast;

    fn resolved_definition(source: &str) -> ast::ClassTree {
        let ast = parse_to_ast(source, "lookup_test.mo").expect("parse should succeed");
        let resolved = crate::resolve_parsed(ast).expect("resolve should succeed");
        resolved.into_inner()
    }

    #[test]
    fn active_semantic_lookup_finds_nested_classes_by_short_name_and_def_id() {
        let tree = resolved_definition(
            r#"
model Outer
    model Inner
    end Inner;
end Outer;
"#,
        );
        let def = &tree.definitions;
        let inner = &def.classes["Outer"].classes["Inner"];
        let inner_def_id = inner.def_id.expect("Inner should have DefId after resolve");
        let _context = activate_semantic_context(def, &tree.source_map);

        let by_name = find_class_by_name(def, "Inner").expect("nested short-name lookup");
        let by_def_id = find_class_by_def_id(def, inner_def_id).expect("nested def-id lookup");

        assert_eq!(by_name.name.text.as_ref(), "Inner");
        assert_eq!(by_def_id.name.text.as_ref(), "Inner");
        assert!(std::ptr::eq(by_name, inner));
        assert!(std::ptr::eq(by_def_id, inner));
    }
}
