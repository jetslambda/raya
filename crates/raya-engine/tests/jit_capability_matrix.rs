//! Phase 0 regression tests: JIT capability matrix, placeholder rejection,
//! and lifter hard-gate behavior (plan tasks S0, S1, S2).

use raya_engine::compiler::bytecode::{ConstantPool, Function, Metadata, Module, Opcode};
use raya_engine::jit::capability::{
    jit_support, opcode_supported_for_jit, produces_incorrect_native_results, JitSupport,
};
use raya_engine::jit::pipeline::lifter::{lift_function, LiftError};
use raya_engine::jit::analysis::heuristics::function_supported_for_jit;

fn make_module(functions: Vec<Function>) -> Module {
    Module {
        runtime_types: Vec::new(),
        function_signatures: Vec::new(),
        magic: *b"RAYA",
        version: 1,
        flags: 0,
        constants: ConstantPool::new(),
        functions,
        classes: vec![],
        metadata: Metadata {
            name: "capmatrix".to_string(),
            source_file: None,
            generic_templates: vec![],
            template_symbol_table: vec![],
            mono_debug_map: vec![],
            structural_shapes: vec![],
            structural_layouts: vec![],
        },
        exports: vec![],
        imports: vec![],
        checksum: [0; 32],
        reflection: None,
        debug_info: None,
        native_functions: vec![],
        jit_hints: vec![],
    }
}

fn func(name: &str, code: Vec<u8>, local_count: usize) -> Function {
    Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: name.to_string(),
        param_count: 0,
        local_count,
        code,
    }
}

fn emit_i32(code: &mut Vec<u8>, v: i32) {
    code.push(Opcode::ConstI32 as u8);
    code.extend_from_slice(&v.to_le_bytes());
}
fn emit_f64(code: &mut Vec<u8>, v: f64) {
    code.push(Opcode::ConstF64 as u8);
    code.extend_from_slice(&v.to_le_bytes());
}
fn emit(code: &mut Vec<u8>, op: Opcode) {
    code.push(op as u8);
}

#[test]
fn known_wrong_opcodes_are_rejected_by_the_lifter() {
    let cases: Vec<(&str, Vec<u8>, Opcode)> = vec![
        (
            "ipow",
            {
                let mut c = Vec::new();
                emit_i32(&mut c, 2);
                emit_i32(&mut c, 3);
                emit(&mut c, Opcode::Ipow);
                emit(&mut c, Opcode::Return);
                c
            },
            Opcode::Ipow,
        ),
        (
            "fpow",
            {
                let mut c = Vec::new();
                emit_f64(&mut c, 2.0);
                emit_f64(&mut c, 3.0);
                emit(&mut c, Opcode::Fpow);
                emit(&mut c, Opcode::Return);
                c
            },
            Opcode::Fpow,
        ),
        (
            "fmod",
            {
                let mut c = Vec::new();
                emit_f64(&mut c, 2.0);
                emit_f64(&mut c, 3.0);
                emit(&mut c, Opcode::Fmod);
                emit(&mut c, Opcode::Return);
                c
            },
            Opcode::Fmod,
        ),
        (
            "get_arg_count",
            {
                let mut c = Vec::new();
                emit(&mut c, Opcode::GetArgCount);
                emit(&mut c, Opcode::Return);
                c
            },
            Opcode::GetArgCount,
        ),
    ];

    for (name, code, expected_opcode) in cases {
        let module = make_module(vec![func(name, code, 0)]);
        let result = lift_function(&module.functions[0], &module, 0);

        match result {
            Err(LiftError::UnsupportedOpcode { opcode, .. }) => {
                assert_eq!(opcode, expected_opcode, "{name}: wrong opcode in error");
            }
            Ok(_) => panic!("{name}: lifter accepted a known-wrong opcode"),
            Err(other) => panic!("{name}: unexpected error: {other:?}"),
        }

        assert!(
            !function_supported_for_jit(&module.functions[0]),
            "{name} must not be selectable"
        );
    }
}

#[test]
fn simple_math_loop_is_still_selectable_and_liftable() {
    // i = 0; while i < 10 { i = i + 1 } return i
    let mut code = Vec::new();
    emit_i32(&mut code, 0); // offset 0
    code.push(Opcode::StoreLocal as u8); // offset 5
    code.extend_from_slice(&0u16.to_le_bytes());
    let loop_start = 8usize;
    code.push(Opcode::LoadLocal as u8); // offset 8 (header)
    code.extend_from_slice(&0u16.to_le_bytes());
    emit_i32(&mut code, 10); // offset 11
    emit(&mut code, Opcode::Ilt); // offset 16
    // JmpIfFalse to exit at offset 22+... computed below
    let jmp_pos = code.len(); // offset 17
    code.push(Opcode::JmpIfFalse as u8);
    code.extend_from_slice(&[0, 0, 0, 0]); // patched later
    code.push(Opcode::LoadLocal as u8); // offset 22
    code.extend_from_slice(&0u16.to_le_bytes());
    emit_i32(&mut code, 1); // offset 25
    emit(&mut code, Opcode::Iadd); // offset 30
    code.push(Opcode::StoreLocal as u8); // offset 31
    code.extend_from_slice(&0u16.to_le_bytes());
    // backward jump to loop_start; encoded relative to ip after operand
    let back = (loop_start as i32) - (code.len() as i32) - 5;
    code.push(Opcode::Jmp as u8);
    code.extend_from_slice(&back.to_le_bytes());
    let exit_pos = code.len();
    let fwd = (exit_pos as i32) - (jmp_pos as i32) - 5;
    code[jmp_pos + 1..jmp_pos + 5].copy_from_slice(&fwd.to_le_bytes());
    code.push(Opcode::LoadLocal as u8);
    code.extend_from_slice(&0u16.to_le_bytes());
    emit(&mut code, Opcode::Return);

    let module = make_module(vec![func("loop", code, 1)]);
    assert!(function_supported_for_jit(&module.functions[0]));
    let lifted = lift_function(&module.functions[0], &module, 0);
    assert!(lifted.is_ok(), "loop must still lift: {:?}", lifted.err());
}

#[test]
fn capability_table_is_internally_consistent() {
    // This test used to assert one flat invariant over nine opcodes: "if it is on the
    // known-wrong list then it is never selectable and never Native/Helper exact". That
    // conflated three genuinely different categories and went stale twice, so CI was red
    // for much of the milestone.
    //
    //   * `Try`, `Throw`, `Rethrow`, `EndTry` were NEVER on the known-wrong list. They
    //     are deliberately interpreted (D4.5): `JitTerminator::Throw` stays fail-closed
    //     so a compiled trap cannot SIGTRAP the process. Asserting they were
    //     "known-wrong" was simply wrong.
    //   * `BindMethod` WAS on that list, and D4.4 removed it when it was promoted on
    //     evidence. The old invariant cannot hold for a promoted opcode.
    //   * The rest are still known-wrong and still unselectable.
    //
    // The invariant that actually survives: an opcode is either on the known-wrong list
    // and unselectable, or it was deliberately re-classified with the reasoning recorded
    // next to it. Each group below says which.

    // (a) Still known-wrong: on the list, unselectable, never natively exact.
    for op in [
        Opcode::Ipow,
        Opcode::Fpow,
        Opcode::Fmod,
        Opcode::GetArgCount,
        Opcode::LoadArgLocal,
    ] {
        assert!(produces_incorrect_native_results(op), "{op:?}");
        assert!(!opcode_supported_for_jit(op), "{op:?}");
        assert_ne!(jit_support(op), JitSupport::NativeExact, "{op:?}");
        assert_ne!(jit_support(op), JitSupport::HelperExact, "{op:?}");
    }

    // (b) Deliberately interpreted, NOT known-wrong: unselectable because we chose to
    // keep them out of the JIT, not because the JIT got them wrong.
    for op in [Opcode::Try, Opcode::Throw, Opcode::Rethrow, Opcode::EndTry] {
        assert!(
            !produces_incorrect_native_results(op),
            "{op:?} is deliberately interpreted, not known-wrong"
        );
        assert_eq!(jit_support(op), JitSupport::Rejected, "{op:?}");
        assert!(!opcode_supported_for_jit(op), "{op:?}");
    }

    // (c) Re-promoted on evidence: D4.4 promoted `BindMethod` to `HelperExact` and
    // REMOVED it from the known-wrong list in the same change. Selecting it is the point;
    // the evidence lives in the differential, not in this table.
    assert!(
        !produces_incorrect_native_results(Opcode::BindMethod),
        "D4.4 removed BindMethod from the known-wrong list when it was promoted"
    );
    assert_eq!(
        jit_support(Opcode::BindMethod),
        JitSupport::HelperExact,
        "D4.4 promoted BindMethod to HelperExact on evidence"
    );
    assert!(
        opcode_supported_for_jit(Opcode::BindMethod),
        "a promoted opcode must be selectable"
    );
}

#[test]
fn string_and_equality_family_is_selectable() {
    for op in [
        Opcode::Slen,
        Opcode::ToString,
        Opcode::Sconcat,
        Opcode::Seq,
        Opcode::Sne,
        Opcode::Slt,
        Opcode::Sle,
        Opcode::Sgt,
        Opcode::Sge,
        Opcode::Eq,
        Opcode::Ne,
        Opcode::StrictEq,
        Opcode::StrictNe,
    ] {
        assert_eq!(jit_support(op), JitSupport::HelperExact, "{op:?}");
        assert!(opcode_supported_for_jit(op), "{op:?}");
        assert!(!produces_incorrect_native_results(op), "{op:?}");
    }
}

#[test]
fn spawn_stays_out_of_selection() {
    let mut code = Vec::new();
    emit_i32(&mut code, 0);
    code.push(Opcode::Spawn as u8);
    code.extend_from_slice(&1u32.to_le_bytes());
    code.extend_from_slice(&0u16.to_le_bytes());
    emit(&mut code, Opcode::Return);

    let module = make_module(vec![func("spawner", code, 0)]);
    assert!(!function_supported_for_jit(&module.functions[0]));
}
