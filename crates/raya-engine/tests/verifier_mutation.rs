//! Verifier mutation tests (plan task B6).
//!
//! Treat artifacts as untrusted: every test builds a valid typed module,
//! corrupts it, and requires `verify_module` to reject the result with the
//! specific error — never panic, never accept.

use raya_engine::compiler::bytecode::{
    attach_function_signature_hashes, verify_module, Export, Function, FunctionSignature,
    RuntimeTypeDescriptor, SymbolScope, SymbolType, UNTYPED_SIGNATURE_ID,
};
use raya_engine::compiler::bytecode::{ConstantPool, Metadata, Module, Opcode};

fn base_module(functions: Vec<Function>) -> Module {
    Module {
        magic: *b"RAYA",
        version: raya_engine::compiler::bytecode::VERSION,
        flags: 0,
        constants: ConstantPool::new(),
        functions,
        classes: vec![],
        metadata: Metadata {
            name: "mutation".to_string(),
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
        runtime_types: vec![],
        function_signatures: vec![],
    }
}

fn func(name: &str, code: Vec<u8>, param_count: usize, local_count: usize) -> Function {
    Function {
        name: name.to_string(),
        param_count,
        local_count,
        code,
        signature_id: UNTYPED_SIGNATURE_ID,
        local_types: Vec::new(),
        abi_version: 1,
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
fn emit_local(code: &mut Vec<u8>, op: Opcode, idx: u16) {
    code.push(op as u8);
    code.extend_from_slice(&idx.to_le_bytes());
}
fn emit_jmp(code: &mut Vec<u8>, op: Opcode, target: usize, from_offset: usize) {
    code.push(op as u8);
    let rel = (target as i32) - (from_offset as i32) - 5;
    code.extend_from_slice(&rel.to_le_bytes());
}

/// add(a: int, b: int): int — the canonical typed module.
fn typed_add_module() -> Module {
    let mut code = Vec::new();
    emit_local(&mut code, Opcode::LoadLocal, 0);
    emit_local(&mut code, Opcode::LoadLocal, 1);
    code.push(Opcode::Iadd as u8);
    code.push(Opcode::Return as u8);

    let mut m = base_module(vec![Function {
        name: "add".to_string(),
        param_count: 2,
        local_count: 2,
        code,
        signature_id: 1,
        local_types: vec![0, 0], // i32, i32
        abi_version: 1,
    }]);
    m.function_signatures.push(FunctionSignature {
        params: vec![RuntimeTypeDescriptor::I32, RuntimeTypeDescriptor::I32],
        return_type: RuntimeTypeDescriptor::I32,
        rest_element: None,
        flags: Default::default(),
    });
    m.exports.push(Export {
        name: "add".into(),
        symbol_type: SymbolType::Function,
        index: 0,
        symbol_id: 0,
        scope: SymbolScope::Module,
        signature_hash: 0,
        type_signature: None,
        nominal_type: None,
    });
    attach_function_signature_hashes(&mut m);
    m
}

#[test]
fn baseline_typed_module_verifies_and_round_trips() {
    let m = typed_add_module();
    verify_module(&m).expect("valid typed module must verify");

    let bytes = m.encode();
    let decoded = Module::decode(&bytes).expect("round-trip decode");
    assert_eq!(decoded.functions[0].signature_id, 1);
    assert_eq!(decoded.function_signatures.len(), 1);
    verify_module(&decoded).expect("decoded module must verify");
}

#[test]
fn float_operand_into_int_add_is_rejected() {
    let mut m = typed_add_module();
    // Replace body with: ConstF64 1.0, LoadLocal 0, Iadd, Return
    let mut code = Vec::new();
    emit_f64(&mut code, 1.0);
    emit_local(&mut code, Opcode::LoadLocal, 0);
    code.push(Opcode::Iadd as u8);
    code.push(Opcode::Return as u8);
    m.functions[0].code = code;

    match verify_module(&m) {
        Err(raya_engine::compiler::bytecode::VerifyError::TypeMismatch {
            expected, actual, ..
        }) => {
            assert_eq!(expected, "Known(0)", "expected i32 descriptor");
            assert_eq!(actual, "Known(1)", "actual f64 descriptor");
        }
        other => panic!("expected TypeMismatch, got {other:?}"),
    }
}

#[test]
fn wrong_return_type_is_rejected() {
    let mut m = typed_add_module();
    // () -> void style abuse: return a value from a void-declared signature.
    m.function_signatures[0].return_type = RuntimeTypeDescriptor::Void;
    let mut code = Vec::new();
    emit_i32(&mut code, 1);
    code.push(Opcode::Return as u8);
    m.functions[0].code = code;

    match verify_module(&m) {
        Err(raya_engine::compiler::bytecode::VerifyError::TypeMismatch { .. }) => {}
        other => panic!("expected TypeMismatch, got {other:?}"),
    }
}

#[test]
fn uninitialized_local_load_is_rejected() {
    let mut m = base_module(vec![func(
        "uninit",
        {
            let mut c = Vec::new();
            emit_local(&mut c, Opcode::LoadLocal, 1); // slot 1 never stored
            c.push(Opcode::Pop as u8);
            c.push(Opcode::ReturnVoid as u8);
            c
        },
        0,
        2,
    )]);

    match verify_module(&m) {
        Err(raya_engine::compiler::bytecode::VerifyError::UninitializedLocal { index, .. }) => {
            assert_eq!(index, 1);
        }
        other => panic!("expected UninitializedLocal, got {other:?}"),
    }
}

#[test]
fn out_of_range_local_index_is_rejected() {
    let mut m = base_module(vec![func(
        "oob",
        {
            let mut c = Vec::new();
            emit_local(&mut c, Opcode::StoreLocal, 9);
            c.push(Opcode::ReturnVoid as u8);
            c
        },
        0,
        1,
    )]);

    match verify_module(&m) {
        Err(raya_engine::compiler::bytecode::VerifyError::InvalidLocalRef { index, .. }) => {
            assert_eq!(index, 9);
        }
        other => panic!("expected InvalidLocalRef, got {other:?}"),
    }
}

#[test]
fn join_with_mismatched_depths_is_rejected() {
    // Two paths reach offset 22: one pushed an i32, the other pushed nothing.
    let mut code = Vec::new();
    code.push(Opcode::ConstTrue as u8); // 0
    emit_jmp(&mut code, Opcode::JmpIfFalse, 16, 1); // 1..6
    emit_i32(&mut code, 1); // 6..11, stack [i32]
    emit_jmp(&mut code, Opcode::Jmp, 22, 11); // 11..16
    code.push(Opcode::Nop as u8); // 16, stack []
    emit_jmp(&mut code, Opcode::Jmp, 22, 17); // 17..22
    code.push(Opcode::ReturnVoid as u8); // 22

    let m = base_module(vec![func("join", code, 0, 0)]);

    match verify_module(&m) {
        Err(raya_engine::compiler::bytecode::VerifyError::StackDepthMismatch { .. }) => {}
        other => panic!("expected StackDepthMismatch, got {other:?}"),
    }
}

#[test]
fn encoded_artifact_byte_flip_is_rejected() {
    // Full artifact treatment: encode, corrupt one opcode byte in the
    // payload, decode, and require rejection.
    let m = typed_add_module();
    let mut bytes = m.encode();

    // LoadLocal 0 (3B), LoadLocal 1 (3B), Iadd (1B) — operands included.
    let code_pattern: Vec<u8> = vec![
        Opcode::LoadLocal as u8,
        0x00,
        0x00,
        Opcode::LoadLocal as u8,
        0x01,
        0x00,
        Opcode::Iadd as u8,
    ];
    let pos = bytes
        .windows(code_pattern.len())
        .position(|w| w == code_pattern.as_slice())
        .expect("body pattern present in encoded module");
    bytes[pos + 3] = Opcode::ConstF64 as u8; // second LoadLocal -> ConstF64

    // Layer 1: artifact integrity. Tampered bytes must fail decoding.
    assert!(
        matches!(
            Module::decode(&bytes),
            Err(raya_engine::compiler::bytecode::ModuleError::ChecksumMismatch { .. })
        ),
        "integrity checksums must reject raw corruption"
    );

    // Layer 2: typed verification. Re-stamp checksums by round-tripping a
    // mutated module through encode, so the verifier sees the bad body.
    let mut mutated = m;
    let mut code = Vec::new();
    code.push(Opcode::LoadLocal as u8);
    code.extend_from_slice(&0u16.to_le_bytes());
    code.push(Opcode::ConstF64 as u8);
    code.extend_from_slice(&1.0f64.to_le_bytes());
    code.push(Opcode::Iadd as u8);
    code.push(Opcode::Return as u8);
    mutated.functions[0].code = code;
    let reencoded = mutated.encode();
    let decoded = Module::decode(&reencoded).expect("re-encoded module decodes");
    let result = verify_module(&decoded);
    assert!(
        matches!(
            result,
            Err(raya_engine::compiler::bytecode::VerifyError::TypeMismatch { .. })
        ),
        "flipped constant must fail typed verification, got {result:?}"
    );
}
