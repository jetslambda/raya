//! Shared file collection utilities for CLI commands.

use std::path::{Path, PathBuf};

/// Collect all .raya source files from the given paths (files or directories).
pub fn collect_raya_files(paths: &[String]) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    for path_str in paths {
        let path = Path::new(path_str);
        if path.is_file() {
            if is_source_extension(path) {
                files.push(path.to_path_buf());
            }
        } else if path.is_dir() {
            collect_raya_in_dir(path, &mut files)?;
        } else if path_str == "." {
            // Current directory
            collect_raya_in_dir(Path::new("."), &mut files)?;
        }
    }

    Ok(files)
}

/// Recursively collect .raya files in a directory.
fn collect_raya_in_dir(dir: &Path, files: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        // Skip hidden dirs, raya_packages, dist, node_modules
        if path.is_dir() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with('.')
                || name_str == "raya_packages"
                || name_str == "dist"
                || name_str == "node_modules"
            {
                continue;
            }
            collect_raya_in_dir(&path, files)?;
        } else if is_source_extension(&path) {
            files.push(path);
        }
    }
    Ok(())
}

/// Source extensions the CLI treats as first-class (task C9).
pub fn is_source_extension(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("raya") | Some("ts") | Some("tsx")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c9_discovers_ts_and_raya_but_not_other_extensions() {
        let dir = std::env::temp_dir().join(format!("raya-c9-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("app.ts"), "const x: number = 1;").unwrap();
        std::fs::write(dir.join("lib.raya"), "export const y: number = 2;").unwrap();
        std::fs::write(dir.join("notes.txt"), "not source").unwrap();

        let found =
            collect_raya_files(&[dir.to_string_lossy().into_owned()]).expect("collect");
        let names: Vec<String> = found
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();

        assert!(names.contains(&"app.ts".to_string()), "{names:?}");
        assert!(names.contains(&"lib.raya".to_string()), "{names:?}");
        assert!(!names.iter().any(|n| n.ends_with(".txt")), "{names:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
