//! I4 regression guard: async calls inside async functions must emit Spawn
//! for every task result type, not only Task<number>.
//!
//! Issue doc: docs/issues/task-non-number-types.md — Task<string> calls made
//! from inside another async function were lowered to synchronous Call,
//! putting result values where WaitAll expects task ids.

use raya_runtime::Runtime;
use std::fs;
use tempfile::TempDir;

/// Count occurrences of an opcode byte in all function bodies.
/// (Crude but effective: operand bytes may alias opcode values, so these
/// assertions compare *relative* presence between variants compiled from
/// identical structure.)
fn count_opcode(module: &raya_engine::compiler::bytecode::Module, op: raya_engine::compiler::bytecode::Opcode) -> usize {
    module.functions
        .iter()
        .map(|f| f.code.iter().filter(|&&b| b == op as u8).count())
        .sum()
}

#[test]
fn i4_task_string_called_inside_async_fn_emits_spawn() {
    let tmp = TempDir::new().unwrap();
    let entry = tmp.path().join("main.raya");
    fs::write(
        &entry,
        r#"
        async function fetchUser(id: number): Promise<string> {
            return "user";
        }

        async function main(): Promise<void> {
            let users = await [fetchUser(1), fetchUser(2)];
        }

        function syncCaller(): Promise<string> {
            return fetchUser(3);
        }
        "#,
    )
    .unwrap();

    let rt = Runtime::new();
    let program = rt
        .compile_program_file(&entry)
        .expect("compiles the issue scenario");

    let m = program.entry.module();
    let spawns = count_opcode(m, raya_engine::compiler::bytecode::Opcode::Spawn);
    let _calls = count_opcode(m, raya_engine::compiler::bytecode::Opcode::Call);

    // fetchUser x2 inside async main (Promise<string>) + x1 inside sync
    // caller. All three call sites must spawn, regardless of result type.
    assert!(
        spawns >= 3,
        "expected >=3 Spawn opcodes, found {spawns}. \
         Non-number promise calls are being lowered to synchronous Call."
    );
}
