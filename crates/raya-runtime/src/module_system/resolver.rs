use crate::error::RuntimeError;
use raya_engine::compiler::module::{ModuleResolver as EngineModuleResolver, StdModuleRegistry};
use std::path::{Path, PathBuf};

use super::std_module_registry;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModuleKey {
    File(PathBuf),
    Std(String),
}

impl ModuleKey {
    pub fn display_name(&self) -> String {
        match self {
            ModuleKey::File(path) => path.display().to_string(),
            ModuleKey::Std(name) => format!("std:{}", name),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleSpecifierKind {
    File(PathBuf),
    Std(String),
}

#[derive(Debug, Clone)]
pub struct ImportResolution {
    pub importer: ModuleKey,
    pub raw_specifier: String,
    pub kind: ModuleSpecifierKind,
}

#[derive(Debug, Default, Clone)]
pub struct ModuleResolverV2;

impl ModuleResolverV2 {
    pub fn resolve(
        &self,
        importer: &ModuleKey,
        specifier: &str,
    ) -> Result<ImportResolution, RuntimeError> {
        let kind = if let Some((canonical, _)) = std_module_registry().resolve_specifier(specifier)
        {
            ModuleSpecifierKind::Std(canonical)
        } else if specifier.starts_with("node:") {
            let supported = StdModuleRegistry::supported_node_module_names()
                .map(|name| format!("node:{}", name))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(RuntimeError::Dependency(format!(
                "Unsupported node module import '{}'. Supported node modules: {}",
                specifier, supported
            )));
        } else {
            match importer {
                ModuleKey::Std(name) => {
                    return Err(RuntimeError::Dependency(format!(
                        "Module '{}' cannot resolve non-std import '{}'",
                        name, specifier
                    )));
                }
                ModuleKey::File(importer_path) => {
                    ModuleSpecifierKind::File(self.resolve_file_like(importer_path, specifier)?)
                }
            }
        };

        Ok(ImportResolution {
            importer: importer.clone(),
            raw_specifier: specifier.to_string(),
            kind,
        })
    }

    fn resolve_file_like(&self, importer: &Path, specifier: &str) -> Result<PathBuf, RuntimeError> {
        if is_local_like(specifier) {
            return self.resolve_local(importer, specifier);
        }

        let project_root = find_project_root(importer);
        let engine_resolver = EngineModuleResolver::new(project_root);
        let resolved = engine_resolver.resolve(specifier, importer).map_err(|e| {
            RuntimeError::Dependency(format!(
                "Failed to resolve '{}' imported from '{}': {}",
                specifier,
                importer.display(),
                e
            ))
        })?;

        let ext = resolved
            .path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        if ext == "ryb" {
            return Err(RuntimeError::Dependency(format!(
                "Import '{}' from '{}' resolved to bytecode '{}'. \
                 Source-program compilation requires .raya sources in lockfile/cache.",
                specifier,
                importer.display(),
                resolved.path.display()
            )));
        }
        if ext != "raya" {
            return Err(RuntimeError::Dependency(format!(
                "Import '{}' from '{}' resolved to unsupported file '{}'",
                specifier,
                importer.display(),
                resolved.path.display()
            )));
        }

        Ok(resolved.path)
    }

    fn resolve_local(&self, importer: &Path, specifier: &str) -> Result<PathBuf, RuntimeError> {
        let importer_dir = importer.parent().ok_or_else(|| {
            RuntimeError::Dependency(format!(
                "Cannot resolve '{}' from '{}': importer has no parent directory",
                specifier,
                importer.display()
            ))
        })?;

        let base = if Path::new(specifier).is_absolute() {
            PathBuf::from(specifier)
        } else {
            importer_dir.join(specifier)
        };

        let mut tried = Vec::new();

        if base.extension().is_some() {
            tried.push(base.clone());
            if base.is_file() {
                return base.canonicalize().map_err(RuntimeError::Io);
            }
            return Err(RuntimeError::Dependency(format!(
                "Module not found: '{}' imported from '{}'. Tried: {}",
                specifier,
                importer.display(),
                tried
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }

        let exact = base.clone();
        // C9: TypeScript entries participate in local resolution alongside
        // .raya. Order is deterministic; multiple existing candidates are
        // an ambiguity error rather than a silent pick.
        let ext_candidates = [
            base.with_extension("raya"),
            base.with_extension("ts"),
            base.with_extension("tsx"),
        ];
        let index_candidates = [
            base.join("index.raya"),
            base.join("index.ts"),
            base.join("index.tsx"),
        ];
        tried.push(exact.clone());
        for c in ext_candidates.iter().chain(index_candidates.iter()) {
            tried.push(c.clone());
        }

        let exact_exists = exact.is_file();
        let existing_exts: Vec<&PathBuf> =
            ext_candidates.iter().filter(|c| c.is_file()).collect();
        let existing_idxs: Vec<&PathBuf> =
            index_candidates.iter().filter(|c| c.is_file()).collect();

        if exact_exists {
            return exact.canonicalize().map_err(RuntimeError::Io);
        }

        if existing_exts.len() + existing_idxs.len() > 1 {
            let all: Vec<String> = existing_exts
                .iter()
                .chain(existing_idxs.iter())
                .map(|p| p.display().to_string())
                .collect();
            return Err(RuntimeError::Dependency(format!(
                "Ambiguous local import '{}' from '{}'. Candidates: {}",
                specifier,
                importer.display(),
                all.join(", ")
            )));
        }

        if let Some(&single) = existing_exts.first().or_else(|| existing_idxs.first()) {
            return single.canonicalize().map_err(RuntimeError::Io);
        }

        Err(RuntimeError::Dependency(format!(
            "Module not found: '{}' imported from '{}'. Tried: {}",
            specifier,
            importer.display(),
            tried
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )))
    }
}

fn is_local_like(specifier: &str) -> bool {
    specifier.starts_with("./") || specifier.starts_with("../") || specifier.starts_with('/')
}

fn find_project_root(importer: &Path) -> PathBuf {
    let mut dir = importer
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    loop {
        if dir.join("raya.toml").is_file() {
            return dir;
        }
        if !dir.pop() {
            break;
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn resolve_local_prefers_exact_then_ext_then_index() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("src");
        fs::create_dir_all(&root).unwrap();

        let importer = root.join("main.raya");
        fs::write(&importer, "").unwrap();

        let exact = root.join("mod");
        fs::write(&exact, "let x: number = 1;").unwrap();
        fs::write(root.join("mod.raya"), "let x: number = 2;").unwrap();

        let resolver = ModuleResolverV2;
        let resolved = resolver
            .resolve(&ModuleKey::File(importer), "./mod")
            .unwrap();
        let ModuleSpecifierKind::File(path) = resolved.kind else {
            panic!("expected local file resolution");
        };
        assert_eq!(path, exact.canonicalize().unwrap());
    }

    #[test]
    fn resolve_local_ambiguity_errors() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("src");
        fs::create_dir_all(&root).unwrap();

        let importer = root.join("main.raya");
        fs::write(&importer, "").unwrap();

        fs::write(root.join("mod.raya"), "let x: number = 2;").unwrap();
        fs::create_dir_all(root.join("mod")).unwrap();
        fs::write(root.join("mod/index.raya"), "let x: number = 3;").unwrap();

        let resolver = ModuleResolverV2;
        let err = resolver
            .resolve(&ModuleKey::File(importer), "./mod")
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("Ambiguous local import"),
            "expected ambiguity error, got: {msg}"
        );
    }
}
