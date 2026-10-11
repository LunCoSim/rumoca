//! Structural queries over authored declarations (SPEC_0029 section 3).

use crate::{AstIndexMap, ClassDef, StoredDefinition};

impl StoredDefinition {
    /// Normalize only this definition's authored package prefix.
    pub fn relative_class_name<'a>(&self, qualified_name: &'a str) -> &'a str {
        let Some(within) = self.within.as_ref() else {
            return qualified_name;
        };
        let prefix = within.to_string();
        if prefix.is_empty() {
            return qualified_name;
        }
        qualified_name
            .strip_prefix(&prefix)
            .and_then(|tail| tail.strip_prefix('.'))
            .unwrap_or(qualified_name)
    }

    /// Borrow an exact qualified or document-relative class declaration.
    /// Quoted identifiers retain their dots; empty path segments do not resolve.
    pub fn get_class_by_qualified_name(&self, qualified_name: &str) -> Option<&ClassDef> {
        let mut parts =
            rumoca_core::qualified_name_segments(self.relative_class_name(qualified_name));
        let mut class = self.classes.get(parts.next()?)?;
        for part in parts {
            class = class.classes.get(part)?;
        }
        Some(class)
    }

    /// Mutably borrow an exact declaration using the read query's source paths.
    pub fn get_class_by_qualified_name_mut(&mut self, name: &str) -> Option<&mut ClassDef> {
        let mut segments = rumoca_core::qualified_name_segments(self.relative_class_name(name));
        let mut current = self.classes.get_mut(segments.next()?)?;
        for segment in segments {
            current = current.classes.get_mut(segment)?;
        }
        Some(current)
    }

    /// Return the authored qualified identity of an exact local declaration.
    /// Missing declarations and invalid paths have no identity. This observes
    /// source structure; it does not resolve imports or inherited members.
    pub fn qualified_class_name(&self, name: &str) -> Option<String> {
        self.get_class_by_qualified_name(name)?;
        Some(qualify(
            &self.package_prefix(),
            self.relative_class_name(name),
        ))
    }

    /// Enumerate every declared class in deterministic source-tree order,
    /// including nested classes and the authored `within` prefix.
    pub fn declared_class_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        collect_names(&self.classes, &self.package_prefix(), &mut names);
        names
    }

    fn package_prefix(&self) -> String {
        self.within
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_default()
    }
}

fn qualify(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_owned()
    } else {
        format!("{prefix}.{name}")
    }
}

fn collect_names(classes: &AstIndexMap<String, ClassDef>, prefix: &str, names: &mut Vec<String>) {
    for (name, class) in classes {
        let qualified = qualify(prefix, name);
        names.push(qualified.clone());
        collect_names(&class.classes, &qualified, names);
    }
}
