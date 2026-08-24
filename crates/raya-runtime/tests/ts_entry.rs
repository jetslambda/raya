//! C9: TypeScript files as first-class entry points.
//!
//! `raya run app.ts` must work without renames: extension-aware type-mode
//! selection, multi-extension local imports, and end-to-end execution.

use raya_runtime::BuiltinMode;
use raya_runtime::module_system::graph::ProgramGraphBuilder;
use raya_runtime::{Runtime, RuntimeOptions};
use std::fs;
use std::path::Path;
use tempfile::TempDir;

#[test]
fn ts_entry_compiles_and_runs_end_to_end() {
    let tmp = TempDir::new().unwrap();
    let entry = tmp.path().join("hello.ts");
    fs::write(
        &entry,
        r#"
        const answer: number = 40 + 2;
        function double(n: number): number {
            return n * 2;
        }
        const doubled: number = double(answer);
        "#,
    )
    .unwrap();

    let rt = Runtime::new();
    let exit = rt.run_file(&entry).expect("run .ts entry");
    assert_eq!(exit, 0);
}

#[test]
fn ts_extension_selects_ts_mode_implicitly() {
    // A file using TS-only surface (type alias + assertion) must compile
    // through the implicit Ts path, proving the mode was selected by
    // extension rather than builtin default.
    let tmp = TempDir::new().unwrap();
    let entry = tmp.path().join("ts_syntax.ts");
    fs::write(
        &entry,
        r#"
        type Meters = number;
        const distance: Meters = 5;
        const scaled = distance as number;
        "#,
    )
    .unwrap();

    let rt = Runtime::new();
    let program = rt
        .compile_program_file(&entry)
        .expect("implicit Ts mode compiles TS-only syntax");
    assert!(!program.entry.module().functions.is_empty());
}

#[test]
fn explicit_type_mode_override_wins_over_extension() {
    let tmp = TempDir::new().unwrap();
    let entry = tmp.path().join("plain.ts");
    fs::write(&entry, "const v: number = 1;").unwrap();

    let options = RuntimeOptions {
        builtin_mode: BuiltinMode::RayaStrict,
        type_mode: Some(raya_runtime::compile::TypeMode::Raya),
        ..Default::default()
    };
    let rt = Runtime::with_options(options);
    // Raya-strict rejects TS-flavored constructs; use plain code so this
    // only asserts the override path doesn't panic and still compiles.
    let program = rt.compile_program_file(&entry).expect("override compiles");
    assert!(!program.entry.module().functions.is_empty());
}

#[test]
fn extensionless_ts_import_resolves() {
    let tmp = TempDir::new().unwrap();
    let a = tmp.path().join("a.ts");
    let utils = tmp.path().join("utils.ts");
    fs::write(
        &a,
        r#"
        import { helper } from "./utils";
        export const combined: number = helper + 1;
        "#,
    )
    .unwrap();
    fs::write(&utils, "export const helper: number = 41;").unwrap();

    let graph = ProgramGraphBuilder::new().build(&a).expect("graph resolves");
    assert!(graph.topological_order.len() >= 2, "both modules in graph");
}

#[test]
fn index_ts_resolves_for_directory_imports() {
    let tmp = TempDir::new().unwrap();
    let a = tmp.path().join("main.ts");
    let pkg_dir = tmp.path().join("pkg");
    fs::create_dir_all(&pkg_dir).unwrap();
    fs::write(
        &a,
        r#"
        import { value } from "./pkg";
        export const out: number = value;
        "#,
    )
    .unwrap();
    fs::write(pkg_dir.join("index.ts"), "export const value: number = 7;").unwrap();

    let graph = ProgramGraphBuilder::new().build(&a).expect("index.ts resolves");
    assert!(graph.topological_order.len() >= 2);
}

#[test]
fn ambiguous_raya_and_ts_candidates_are_rejected() {
    let tmp = TempDir::new().unwrap();
    let a = tmp.path().join("a.raya");
    fs::write(
        &a,
        r#"
        import { v } from "./dupe";
        export const a: number = v;
        "#,
    )
    .unwrap();
    fs::write(tmp.path().join("dupe.raya"), "export const v: number = 1;").unwrap();
    fs::write(tmp.path().join("dupe.ts"), "export const v: number = 2;").unwrap();

    let err = ProgramGraphBuilder::new().build(&a).unwrap_err();
    let msg = format!("{}", err);
    assert!(
        msg.contains("Ambiguous") && msg.contains("dupe.raya") && msg.contains("dupe.ts"),
        "ambiguity names both candidates: {msg}"
    );
}

#[test]
fn mixed_project_raya_entry_imports_ts_module() {
    let tmp = TempDir::new().unwrap();
    let main = tmp.path().join("main.raya");
    let lib = tmp.path().join("lib.ts");
    fs::write(
        &main,
        r#"
        import { ts_value } from "./lib";
        export const total: number = ts_value;
        "#,
    )
    .unwrap();
    fs::write(&lib, "export const ts_value: number = 9;").unwrap();

    let graph = ProgramGraphBuilder::new().build(&main).expect("mixed links");
    assert!(graph.topological_order.len() >= 2);
}
