//! Class navigation queries over the session's authoritative parsed sources.
use super::class_body::FileClassBodyIndex;
use super::declaration_index::ItemKind;
use super::session_impl::{
    NavigationReadContext, collect_navigation_class_reference_locations_in_definition,
    collect_navigation_class_rename_locations_in_definition, navigation_location_contains_position,
};
use super::session_impl_queries::QueryClassNavigationTarget;
use super::*;

struct NavigationTargetSearch<'a> {
    uri: &'a str,
    file_id: FileId,
    class_bodies: &'a FileClassBodyIndex,
    line: u32,
    character: u32,
}

impl Session {
    pub(super) fn navigation_class_references_query(
        &mut self,
        uri: &str,
        line: u32,
        character: u32,
        include_declaration: bool,
    ) -> Option<Vec<(String, rumoca_core::Location)>> {
        let target = self.navigation_class_target_at_position(uri, line, character)?;
        let document_uris = self.parsed_navigation_document_uris();
        self.prewarm_navigation_read_queries(&document_uris);
        let read = NavigationReadContext::new(self);
        let mut locations = Vec::new();
        for document_uri in document_uris {
            let Some(definition) = self
                .documents
                .get(&document_uri)
                .and_then(|doc| doc.parsed())
            else {
                continue;
            };
            collect_navigation_class_reference_locations_in_definition(
                &read,
                &document_uri,
                definition,
                &target.qualified_name,
                include_declaration,
                &mut locations,
            );
        }
        (!locations.is_empty()).then_some(locations)
    }

    pub(super) fn navigation_class_prepare_rename_query(
        &mut self,
        uri: &str,
        line: u32,
        character: u32,
    ) -> Option<rumoca_core::Location> {
        self.navigation_class_target_at_position(uri, line, character)
            .map(|target| target.location)
    }

    pub(super) fn navigation_class_rename_locations_query(
        &mut self,
        uri: &str,
        line: u32,
        character: u32,
    ) -> Option<Vec<(String, rumoca_core::Location)>> {
        let target = self.navigation_class_target_at_position(uri, line, character)?;
        let document_uris = self.parsed_navigation_document_uris();
        self.prewarm_navigation_read_queries(&document_uris);
        let read = NavigationReadContext::new(self);
        let mut locations = Vec::new();
        for document_uri in document_uris {
            let Some(definition) = self
                .documents
                .get(&document_uri)
                .and_then(|doc| doc.parsed())
            else {
                continue;
            };
            collect_navigation_class_rename_locations_in_definition(
                &read,
                &document_uri,
                definition,
                &target,
                &mut locations,
            );
        }
        (!locations.is_empty()).then_some(locations)
    }

    pub(super) fn navigation_qualified_class_name_at_position(
        &mut self,
        uri: &str,
        line: u32,
        character: u32,
    ) -> Option<String> {
        let dotted_name = {
            let source = self.documents.get(uri)?.content.clone();
            qualified_class_name_at_position(&source, line, character)?
        };
        self.class_lookup_query(&dotted_name)
    }

    pub(super) fn navigation_class_target_info(
        &self,
        target_uri: &str,
        qualified_name: &str,
    ) -> Option<NavigationClassTargetInfo> {
        let definition = self
            .documents
            .get(target_uri)
            .and_then(|doc| doc.parsed().or(doc.recovered()))?;
        let class = definition.get_class_by_qualified_name(qualified_name)?;
        Some(NavigationClassTargetInfo {
            target_uri: target_uri.to_string(),
            qualified_name: qualified_name.to_string(),
            class_name: class.name.text.to_string(),
            class_type: class.class_type.clone(),
            description: class.description.first().map(|desc| desc.text.to_string()),
            component_count: class.components.len(),
            equation_count: class.equations.len() + class.initial_equations.len(),
            declaration_location: class.name.location.clone(),
        })
    }

    pub(super) fn navigation_class_target_at_position(
        &mut self,
        uri: &str,
        line: u32,
        character: u32,
    ) -> Option<QueryClassNavigationTarget> {
        let file_id = self.file_id_for_uri(uri)?;
        let class_bodies = self.file_class_body_query(uri)?.clone();
        let definition = self.documents.get(uri)?.parsed().cloned()?;
        let within_prefix = definition
            .within
            .as_ref()
            .map(ToString::to_string)
            .filter(|prefix| !prefix.is_empty())
            .unwrap_or_default();
        let search = NavigationTargetSearch {
            uri,
            file_id,
            class_bodies: &class_bodies,
            line,
            character,
        };
        for (name, class) in &definition.classes {
            if let Some(target) =
                self.navigation_class_target_in_class(&search, &within_prefix, name, class)
            {
                return Some(target);
            }
        }
        None
    }

    fn navigation_class_target_in_class(
        &mut self,
        search: &NavigationTargetSearch<'_>,
        container_path: &str,
        class_name: &str,
        class: &ast::ClassDef,
    ) -> Option<QueryClassNavigationTarget> {
        let class_item_key =
            ItemKey::new(search.file_id, ItemKind::Class, container_path, class_name);
        let qualified_name = class_item_key.qualified_name();
        if navigation_location_contains_position(
            &class.name.location,
            search.line,
            search.character,
        ) {
            return Some(QueryClassNavigationTarget {
                qualified_name: qualified_name.clone(),
                token_text: class.name.text.to_string(),
                location: class.name.location.clone(),
            });
        }
        if let Some(end_name) = &class.end_name_token
            && navigation_location_contains_position(
                &end_name.location,
                search.line,
                search.character,
            )
        {
            return Some(QueryClassNavigationTarget {
                qualified_name: qualified_name.clone(),
                token_text: end_name.text.to_string(),
                location: end_name.location.clone(),
            });
        }
        if let Some(constrainedby) = &class.constrainedby
            && let Some(target) = self.navigation_class_target_in_type_name(
                search.uri,
                &qualified_name,
                constrainedby,
                search.line,
                search.character,
            )
        {
            return Some(target);
        }
        for import in &class.imports {
            if let Some(target) =
                self.navigation_class_target_in_import(import, search.line, search.character)
            {
                return Some(target);
            }
        }
        for extend in &class.extends {
            if let Some(target) = self.navigation_class_target_in_type_name(
                search.uri,
                &qualified_name,
                &extend.base_name,
                search.line,
                search.character,
            ) {
                return Some(target);
            }
        }
        if let Some(target) =
            self.navigation_class_target_in_body(search, &qualified_name, &class_item_key)
        {
            return Some(target);
        }
        for (nested_name, nested_class) in &class.classes {
            if let Some(target) = self.navigation_class_target_in_class(
                search,
                &qualified_name,
                nested_name,
                nested_class,
            ) {
                return Some(target);
            }
        }
        for component in class.components.values() {
            if let Some(target) = self.navigation_class_target_in_type_name(
                search.uri,
                &qualified_name,
                &component.type_name,
                search.line,
                search.character,
            ) {
                return Some(target);
            }
            if let Some(constrainedby) = &component.constrainedby
                && let Some(target) = self.navigation_class_target_in_type_name(
                    search.uri,
                    &qualified_name,
                    constrainedby,
                    search.line,
                    search.character,
                )
            {
                return Some(target);
            }
        }
        None
    }

    fn navigation_class_target_in_body(
        &mut self,
        search: &NavigationTargetSearch<'_>,
        enclosing_qualified_name: &str,
        class_item_key: &ItemKey,
    ) -> Option<QueryClassNavigationTarget> {
        let class_body = search.class_bodies.class_body(class_item_key)?;
        for target in class_body.modifier_class_targets() {
            if !navigation_location_contains_position(
                target.location(),
                search.line,
                search.character,
            ) {
                continue;
            }
            let qualified_name = self.resolve_navigation_class_name(
                search.uri,
                enclosing_qualified_name,
                target.raw_name(),
            )?;
            return Some(QueryClassNavigationTarget {
                qualified_name,
                token_text: target.token_text().to_string(),
                location: target.location().clone(),
            });
        }
        None
    }

    fn navigation_class_target_in_import(
        &mut self,
        import: &ast::Import,
        line: u32,
        character: u32,
    ) -> Option<QueryClassNavigationTarget> {
        match import {
            ast::Import::Qualified { path, .. } => {
                let token = path.name.last()?;
                if !navigation_location_contains_position(&token.location, line, character) {
                    return None;
                }
                let qualified_name = self.class_lookup_query(&path.to_string())?;
                Some(QueryClassNavigationTarget {
                    qualified_name,
                    token_text: token.text.to_string(),
                    location: token.location.clone(),
                })
            }
            ast::Import::Renamed { alias, path, .. } => {
                let path_token = path.name.last()?;
                let on_alias =
                    navigation_location_contains_position(&alias.location, line, character);
                let on_path =
                    navigation_location_contains_position(&path_token.location, line, character);
                if !on_alias && !on_path {
                    return None;
                }
                let token = if on_alias { alias } else { path_token };
                let qualified_name = self.class_lookup_query(&path.to_string())?;
                Some(QueryClassNavigationTarget {
                    qualified_name,
                    token_text: token.text.to_string(),
                    location: token.location.clone(),
                })
            }
            ast::Import::Selective { path, names, .. } => {
                let token = names.iter().find(|token| {
                    navigation_location_contains_position(&token.location, line, character)
                })?;
                let candidate = format!("{path}.{}", token.text);
                let qualified_name = self.class_lookup_query(&candidate)?;
                Some(QueryClassNavigationTarget {
                    qualified_name,
                    token_text: token.text.to_string(),
                    location: token.location.clone(),
                })
            }
            ast::Import::Unqualified { .. } => None,
        }
    }

    fn navigation_class_target_in_type_name(
        &mut self,
        uri: &str,
        enclosing_qualified_name: &str,
        type_name: &ast::Name,
        line: u32,
        character: u32,
    ) -> Option<QueryClassNavigationTarget> {
        let token = type_name.name.last()?;
        if !navigation_location_contains_position(&token.location, line, character) {
            return None;
        }
        let qualified_name = self.resolve_navigation_class_name(
            uri,
            enclosing_qualified_name,
            &type_name.to_string(),
        )?;
        Some(QueryClassNavigationTarget {
            qualified_name,
            token_text: token.text.to_string(),
            location: token.location.clone(),
        })
    }

    pub(crate) fn resolve_navigation_class_name(
        &mut self,
        uri: &str,
        enclosing_qualified_name: &str,
        raw_type_name: &str,
    ) -> Option<String> {
        for candidate in self.class_type_resolution_candidates_query(
            uri,
            enclosing_qualified_name,
            raw_type_name,
        ) {
            if let Some(qualified_name) = self.class_lookup_query(&candidate) {
                return Some(qualified_name);
            }
        }
        None
    }

    fn parsed_navigation_document_uris(&self) -> Vec<String> {
        self.documents
            .iter()
            .filter(|(_, doc)| doc.parsed().is_some())
            .map(|(uri, _)| uri.clone())
            .collect()
    }

    fn prewarm_navigation_read_queries(&mut self, document_uris: &[String]) {
        let _ = self.session_package_def_map_query();
        for document_uri in document_uris {
            let _ = self.class_interface_index_query(document_uri);
            let _ = self.class_body_semantics_query(document_uri);
        }
    }
}

fn qualified_class_name_at_position(source: &str, line: u32, character: u32) -> Option<String> {
    let dotted_token = dotted_token_at_position(source, line, character)?;
    rumoca_core::has_top_level_dot(&dotted_token)
        .then_some(dotted_token)
        .filter(|token| token.chars().next().is_some_and(|c| c.is_ascii_uppercase()))
}

fn dotted_token_at_position(source: &str, line: u32, character: u32) -> Option<String> {
    let line = source.lines().nth(line as usize)?;
    let col = character as usize;
    if col > line.len() {
        return None;
    }
    let start = line[..col]
        .rfind(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
        .map(|idx| idx + 1)
        .unwrap_or(0);
    let end = line[col..]
        .find(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
        .map(|idx| col + idx)
        .unwrap_or(line.len());
    (start < end).then(|| line[start..end].to_string())
}
