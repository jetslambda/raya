//! `raya readiness`: run the raya-ts-check analyzer over a TypeScript project.
//!
//! The checker is a standalone Node package (packages/raya-ts-check). This
//! command locates it and forwards arguments verbatim — no shell, no
//! interpolation. Exit codes pass through: 0 pass, 1 threshold met,
//! 2 configuration/analysis failure.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn execute(project: PathBuf, format: String, fail_on: String) -> Result<()> {
    let code = run_checker(&project, &format, &fail_on)?;
    std::process::exit(code);
}

/// Validate flags, locate the checker, run it, return its exit code.
fn run_checker(project: &Path, format: &str, fail_on: &str) -> Result<i32> {
    if !matches!(format, "pretty" | "json") {
        bail!("invalid --format '{format}' (expected pretty|json)");
    }
    if !matches!(fail_on, "info" | "warning" | "error") {
        bail!("invalid --fail-on '{fail_on}' (expected info|warning|error)");
    }

    let checker = find_checker().context(
        "raya-ts-check not found.\n\
         Install it with: cd packages/raya-ts-check && npm install && npm run build\n\
         or set RAYA_TS_CHECK to its cli.js path",
    )?;

    let mut cmd = match checker.extension().and_then(|e| e.to_str()) {
        Some("js") => {
            let mut c = Command::new("node");
            c.arg(&checker);
            c
        }
        _ => Command::new(&checker),
    };

    cmd.arg("--project").arg(project)
        .arg("--format").arg(format)
        .arg("--fail-on").arg(fail_on);

    let status = cmd.status().with_context(|| {
        format!("failed to launch {}", checker.display())
    })?;

    Ok(status.code().unwrap_or(2))
}

/// Resolution order: $RAYA_TS_CHECK, repo-local package build output, PATH.
fn find_checker() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("RAYA_TS_CHECK") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }

    let repo_local = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../packages/raya-ts-check/dist/cli.js");
    if repo_local.exists() {
        return Some(repo_local);
    }

    let path_var = std::env::var_os("PATH")?;
    let dirs = std::env::split_paths(&path_var);
    for dir in dirs {
        for candidate in ["raya-ts-check", "raya-ts-check.exe"] {
            let pb = dir.join(candidate);
            if pb.is_file() {
                return Some(pb);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_format_before_spawning() {
        let err = run_checker(Path::new("."), "yaml", "error").unwrap_err();
        assert!(err.to_string().contains("invalid --format"));
    }

    #[test]
    fn rejects_invalid_fail_on_before_spawning() {
        let err = run_checker(Path::new("."), "json", "fatal").unwrap_err();
        assert!(err.to_string().contains("invalid --fail-on"));
    }

    #[test]
    fn missing_project_returns_config_failure_code_2() {
        // Requires the repo-local checker build; skip when absent.
        if find_checker().is_none() {
            return;
        }
        let temp = std::env::temp_dir().join("raya-readiness-missing-project");
        let _ = std::fs::remove_dir_all(&temp);
        std::fs::create_dir_all(&temp).unwrap();

        let code = run_checker(&temp, "json", "error").unwrap();
        assert_eq!(code, 2, "missing tsconfig must surface as exit code 2");
    }
}
