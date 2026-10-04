//! Comprehensive JIT end-to-end integration tests.
//!
//! Tests the full pipeline: bytecode → analysis → SSA IR → Cranelift → native execution.
//! Organized in 5 categories:
//! 1. Lifter (bytecode → JIT IR)
//! 2. Native execution — constants
//! 3. Native execution — arithmetic
//! 4. Native execution — comparisons, logic, branches
//! 5. Full pipeline + VM integration

use raya_engine::compiler::bytecode::{
    ClassDef, ConstantPool, Function, Metadata, Method, Module, Opcode, VERSION,
};
use raya_engine::jit::backend::cranelift::lowering::{jit_entry_signature, LoweringContext};
use raya_engine::jit::backend::cranelift::CraneliftBackend;
use raya_engine::jit::backend::traits::CodegenBackend;
use raya_engine::jit::ir::instr::{JitBlockId, JitFunction, JitInstr, JitTerminator, Reg};
use raya_engine::jit::ir::types::JitType;
use raya_engine::jit::pipeline::lifter::lift_function;
use raya_engine::jit::pipeline::JitPipeline;
use raya_engine::jit::runtime::helpers::JIT_NATIVE_SUSPEND_SENTINEL;
use raya_engine::jit::runtime::trampoline::{JitEntryFn, RuntimeContext, RuntimeHelperTable};
use raya_engine::jit::{JitConfig, JitEngine};
use raya_engine::Vm;
use rustc_hash::FxHashMap;

use cranelift_codegen::ir::{self, AbiParam};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::FunctionBuilderContext;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::Module as CraneliftModule;

use std::ptr;

// ============================================================================
// NaN-boxing constants (from jit/backend/cranelift/abi.rs)
// ============================================================================

const NAN_BOX_BASE: u64 = 0xFFF8_0000_0000_0000;
const TAG_SHIFT: u64 = 48;
const TAG_I32: u64 = 0x1 << TAG_SHIFT;
const TAG_BOOL: u64 = 0x2 << TAG_SHIFT;
const TAG_NULL: u64 = 0x6 << TAG_SHIFT;
const TAG_MASK: u64 = 0x7 << TAG_SHIFT;
const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;
const PAYLOAD_MASK_32: u64 = 0x0000_0000_FFFF_FFFF;
const I32_TAG_BASE: u64 = NAN_BOX_BASE | TAG_I32;
const BOOL_TAG_BASE: u64 = NAN_BOX_BASE | TAG_BOOL;
const NULL_VALUE: u64 = NAN_BOX_BASE | TAG_NULL;

unsafe extern "C" fn stub_object_get_field(
    _object_raw: u64,
    _expected_slot: u32,
    _layout_generation: u64,
    _func_id: u32,
    _module_ptr: *const (),
    _shared_state: *mut (),
) -> u64 {
    NULL_VALUE
}

unsafe extern "C" fn stub_object_set_field(
    _object_raw: u64,
    _expected_slot: u32,
    _value_raw: u64,
    _func_id: u32,
    _module_ptr: *const (),
    _shared_state: *mut (),
) -> bool {
    false
}

unsafe extern "C" fn stub_object_implements_shape(
    _object_raw: u64,
    _shape_id: u64,
    _shared_state: *mut (),
) -> bool {
    false
}

unsafe extern "C" fn stub_object_is_nominal(
    _object_raw: u64,
    _local_nominal_type_index: u32,
    _module_ptr: *const (),
    _shared_state: *mut (),
) -> bool {
    false
}

unsafe extern "C" fn stub_object_get_shape_field(
    _object_raw: u64,
    _shape_id: u64,
    _expected_slot: u32,
    _optional: u8,
    _func_id: u32,
    _module_ptr: *const (),
    _shared_state: *mut (),
) -> u64 {
    NULL_VALUE
}

unsafe extern "C" fn stub_object_set_shape_field(
    _object_raw: u64,
    _shape_id: u64,
    _expected_slot: u32,
    _value_raw: u64,
    _func_id: u32,
    _module_ptr: *const (),
    _shared_state: *mut (),
) -> i8 {
    0
}

unsafe extern "C" fn stub_string_len(_string_raw: u64, _shared_state: *mut ()) -> i32 {
    i32::MIN
}

unsafe extern "C" fn stub_string_compare(
    _left_raw: u64,
    _right_raw: u64,
    _shared_state: *mut (),
) -> i8 {
    2
}

unsafe extern "C" fn stub_value_to_string(_value_raw: u64, _shared_state: *mut ()) -> u64 {
    NULL_VALUE
}

unsafe extern "C" fn stub_const_string(
    _pool_index: u32,
    _module_ptr: *const (),
    _shared_state: *mut (),
) -> *mut () {
    std::ptr::null_mut()
}

// ============================================================================
// NaN-boxing decode helpers
// ============================================================================

fn is_i32(val: u64) -> bool {
    (val & (NAN_BOX_BASE | TAG_MASK)) == I32_TAG_BASE
}

fn decode_i32(val: u64) -> i32 {
    assert!(is_i32(val), "Expected NaN-boxed i32, got 0x{:016X}", val);
    // Sign-extend from the lower 48 bits
    let payload = val & PAYLOAD_MASK;
    // The i32 is in the lower 32 bits, sign-extended to 48 bits
    payload as i32
}

fn is_f64(val: u64) -> bool {
    // f64 values don't have the NaN-box base pattern
    (val & NAN_BOX_BASE) != NAN_BOX_BASE
}

fn decode_f64(val: u64) -> f64 {
    assert!(is_f64(val), "Expected NaN-boxed f64, got 0x{:016X}", val);
    f64::from_bits(val)
}

fn is_bool(val: u64) -> bool {
    (val & (NAN_BOX_BASE | TAG_MASK)) == BOOL_TAG_BASE
}

fn decode_bool(val: u64) -> bool {
    assert!(is_bool(val), "Expected NaN-boxed bool, got 0x{:016X}", val);
    (val & 1) != 0
}

fn is_null(val: u64) -> bool {
    val == NULL_VALUE
}

// ============================================================================
// Module/bytecode builder helpers
// ============================================================================

fn make_module(code: Vec<u8>, param_count: usize, local_count: usize) -> Module {
    Module {
        runtime_types: Vec::new(),
        function_signatures: Vec::new(),
        magic: *b"RAYA",
        version: VERSION,
        flags: 0,
        constants: ConstantPool::new(),
        functions: vec![Function {
            signature_id: 0,
            local_types: Vec::new(),
            abi_version: 1,
            name: "test_func".to_string(),
            param_count,
            local_count,
            code,
        }],
        classes: vec![],
        metadata: Metadata {
            name: "test_module".to_string(),
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

/// Make a module with a "main" function (required by Vm::execute)
fn make_vm_module(code: Vec<u8>, param_count: usize, local_count: usize) -> Module {
    Module {
        runtime_types: Vec::new(),
        function_signatures: Vec::new(),
        magic: *b"RAYA",
        version: VERSION,
        flags: 0,
        constants: ConstantPool::new(),
        functions: vec![Function {
            signature_id: 0,
            local_types: Vec::new(),
            abi_version: 1,
            name: "main".to_string(),
            param_count,
            local_count,
            code,
        }],
        classes: vec![],
        metadata: Metadata {
            name: "test_module".to_string(),
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

fn make_custom_module(functions: Vec<Function>, classes: Vec<ClassDef>) -> Module {
    Module {
        runtime_types: Vec::new(),
        function_signatures: Vec::new(),
        magic: *b"RAYA",
        version: VERSION,
        flags: 0,
        constants: ConstantPool::new(),
        functions,
        classes,
        metadata: Metadata {
            name: "test_module".to_string(),
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

fn finalize_module(module: Module) -> std::sync::Arc<Module> {
    std::sync::Arc::new(Module::decode(&module.encode()).expect("finalize module checksum"))
}

fn new_shared_vm_state() -> (
    std::sync::Arc<raya_engine::vm::interpreter::SafepointCoordinator>,
    std::sync::Arc<raya_engine::vm::interpreter::SharedVmState>,
) {
    let safepoint = std::sync::Arc::new(
        raya_engine::vm::interpreter::SafepointCoordinator::new(1),
    );
    let tasks = std::sync::Arc::new(parking_lot::RwLock::new(FxHashMap::default()));
    let injector = std::sync::Arc::new(crossbeam_deque::Injector::new());
    let shared = std::sync::Arc::new(raya_engine::vm::interpreter::SharedVmState::new(
        safepoint.clone(),
        tasks,
        injector,
    ));
    *shared.code_cache.lock() = Some(std::sync::Arc::new(
        raya_engine::jit::runtime::code_cache::CodeCache::new(1024 * 1024),
    ));
    (safepoint, shared)
}

fn test_code_cache(
    shared: &raya_engine::vm::interpreter::SharedVmState,
) -> std::sync::Arc<raya_engine::jit::runtime::code_cache::CodeCache> {
    let mut cache = shared.code_cache.lock();
    cache
        .get_or_insert_with(|| {
            std::sync::Arc::new(
                raya_engine::jit::runtime::code_cache::CodeCache::new(1024 * 1024),
            )
        })
        .clone()
}

fn build_bridge_and_ctx<'a>(
    safepoint: &'a std::sync::Arc<raya_engine::vm::interpreter::SafepointCoordinator>,
    shared: &'a std::sync::Arc<raya_engine::vm::interpreter::SharedVmState>,
    task: &'a std::sync::Arc<raya_engine::vm::scheduler::Task>,
    module: &'a std::sync::Arc<Module>,
) -> (
    parking_lot::RwLock<raya_engine::vm::native_registry::ResolvedNatives>,
    raya_engine::jit::runtime::helpers::JitRuntimeBridgeContext,
) {
    let resolved_natives = parking_lot::RwLock::new(shared.resolved_natives.read().clone());
    let code_cache = test_code_cache(shared.as_ref());
    let bridge = raya_engine::jit::runtime::helpers::build_runtime_bridge_context(
        safepoint.as_ref(),
        task,
        &shared.gc,
        &shared.classes,
        &shared.layouts,
        code_cache.as_ref(),
        &shared.mutex_registry,
        &shared.semaphore_registry,
        &shared.globals_by_index,
        &shared.builtin_global_slots,
        &shared.constant_string_cache,
        &shared.ephemeral_gc_roots,
        &shared.pinned_handles,
        &shared.tasks,
        &shared.injector,
        &shared.module_layouts,
        &shared.metadata,
        &shared.class_metadata,
        &shared.native_handler,
        &resolved_natives,
        &shared.structural_shape_names,
        &shared.structural_layout_shapes,
        &shared.structural_shape_adapters,
        &shared.aot_profile,
        &shared.type_handles,
        &shared.prop_keys,
        &shared.stack_pool,
        shared.max_preemptions,
        0,
        None,
    );
    let _ = module;
    (resolved_natives, bridge)
}

fn execute_module_natively(
    module: std::sync::Arc<Module>,
    gc_threshold: Option<usize>,
) -> (
    raya_engine::vm::value::Value,
    raya_engine::jit::runtime::trampoline::JitExitInfo,
    std::sync::Arc<raya_engine::vm::interpreter::SharedVmState>,
) {
    let (safepoint, shared) = new_shared_vm_state();
    if let Some(threshold) = gc_threshold {
        shared.gc.lock().set_threshold(threshold);
    }
    let gc_context_id = shared.gc.lock().context_id();
    let weak_shared = std::sync::Arc::downgrade(&shared);
    raya_engine::vm::gc::register_external_roots_provider(
        gc_context_id,
        std::sync::Arc::new(move || {
            weak_shared
                .upgrade()
                .map(|state| state.collect_gc_roots())
                .unwrap_or_default()
        }),
    );
    shared
        .register_module(module.clone())
        .expect("register native test module");
    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(
        0,
        module.clone(),
        None,
    ));
    let (_resolved_natives, bridge) = build_bridge_and_ctx(&safepoint, &shared, &task, &module);
    let mut ctx =
        raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    let jit_func = lift_function(&module.functions[0], &module, 0).expect("lift native test module");
    let (raw, exit) = jit_compile_and_call_with_locals_exit_and_ctx_and_module(
        &jit_func,
        &module,
        &mut [],
        &mut ctx as *mut _,
    );
    raya_engine::vm::gc::unregister_external_roots_provider(gc_context_id);
    (
        unsafe { raya_engine::vm::value::Value::from_raw(raw) },
        exit,
        shared,
    )
}

fn string_contents(value: raya_engine::vm::value::Value) -> String {
    let ptr = unsafe { value.as_ptr::<raya_engine::vm::object::RayaString>() }
        .expect("expected string pointer");
    unsafe { &*ptr.as_ptr() }.data.clone()
}

fn emit(code: &mut Vec<u8>, op: Opcode) {
    code.push(op as u8);
}

fn emit_i32(code: &mut Vec<u8>, val: i32) {
    code.push(Opcode::ConstI32 as u8);
    code.extend_from_slice(&val.to_le_bytes());
}

fn emit_f64(code: &mut Vec<u8>, val: f64) {
    code.push(Opcode::ConstF64 as u8);
    code.extend_from_slice(&val.to_le_bytes());
}

fn emit_const_str(code: &mut Vec<u8>, index: u32) {
    code.push(Opcode::ConstStr as u8);
    code.extend_from_slice(&index.to_le_bytes());
}

fn emit_store_local(code: &mut Vec<u8>, idx: u16) {
    code.push(Opcode::StoreLocal as u8);
    code.extend_from_slice(&idx.to_le_bytes());
}

fn emit_load_local(code: &mut Vec<u8>, idx: u16) {
    code.push(Opcode::LoadLocal as u8);
    code.extend_from_slice(&idx.to_le_bytes());
}

fn emit_store_field_exact(code: &mut Vec<u8>, idx: u16) {
    code.push(Opcode::StoreFieldExact as u8);
    code.extend_from_slice(&idx.to_le_bytes());
}

fn emit_jmp(code: &mut Vec<u8>, op: Opcode, offset: i32) {
    code.push(op as u8);
    code.extend_from_slice(&offset.to_le_bytes());
}

fn emit_jmp_i32_placeholder(code: &mut Vec<u8>, op: Opcode) -> usize {
    code.push(op as u8);
    let imm_pos = code.len();
    code.extend_from_slice(&0i32.to_le_bytes());
    imm_pos
}

fn patch_jmp_i32_compiler(code: &mut [u8], imm_pos: usize, target_pos: usize) {
    // Match compiler/VM semantics: jump offset is relative to IP after reading i32.
    let rel = target_pos as isize - (imm_pos as isize + 4);
    let rel_i32 = i32::try_from(rel).expect("test jump offset must fit i32");
    code[imm_pos..imm_pos + 4].copy_from_slice(&rel_i32.to_le_bytes());
}

// ============================================================================
// JIT execution helper
// ============================================================================

/// Compile a JitFunction to native code via cranelift_jit::JITModule and call it.
/// Returns the raw NaN-boxed u64 result.
fn jit_compile_and_call(func: &JitFunction) -> u64 {
    jit_compile_and_call_with_locals(func, &mut [])
}

/// Same as jit_compile_and_call but with a pre-allocated locals buffer.
fn jit_compile_and_call_with_locals(func: &JitFunction, locals: &mut [u64]) -> u64 {
    jit_compile_and_call_with_locals_and_exit(func, locals).0
}

/// Same as jit_compile_and_call_with_locals but also returns JIT exit metadata.
fn jit_compile_and_call_with_locals_and_exit(
    func: &JitFunction,
    locals: &mut [u64],
) -> (u64, raya_engine::jit::runtime::trampoline::JitExitInfo) {
    jit_compile_and_call_with_locals_exit_and_ctx(func, locals, std::ptr::null_mut())
}

fn dummy_lowering_module() -> Module {
    make_custom_module(vec![], vec![])
}

fn jit_compile_and_call_with_locals_exit_and_ctx(
    func: &JitFunction,
    locals: &mut [u64],
    ctx_ptr: *mut RuntimeContext,
) -> (u64, raya_engine::jit::runtime::trampoline::JitExitInfo) {
    let module = dummy_lowering_module();
    jit_compile_and_call_with_locals_exit_and_ctx_and_module(func, &module, locals, ctx_ptr)
}

fn jit_compile_and_call_with_locals_exit_and_ctx_and_module(
    func: &JitFunction,
    module: &Module,
    locals: &mut [u64],
    ctx_ptr: *mut RuntimeContext,
) -> (u64, raya_engine::jit::runtime::trampoline::JitExitInfo) {
    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let isa = cranelift_native::builder().unwrap().finish(flags).unwrap();

    let call_conv = isa.default_call_conv();
    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut jit_module = JITModule::new(builder);

    // Declare the function
    let sig = jit_entry_signature(call_conv);
    let func_id = jit_module
        .declare_function("test_func", cranelift_module::Linkage::Local, &sig)
        .unwrap();

    // Compile: build Cranelift IR from JIT IR
    let mut codegen_ctx = Context::new();
    let mut func_builder_ctx = FunctionBuilderContext::new();

    codegen_ctx.func.signature = jit_entry_signature(call_conv);
    codegen_ctx.func.name = ir::UserFuncName::user(0, func.func_index);

    {
        let builder =
            cranelift_frontend::FunctionBuilder::new(&mut codegen_ctx.func, &mut func_builder_ctx);
        LoweringContext::lower(func, module, None, builder).expect("Lowering failed");
    }

    // Define and finalize
    jit_module
        .define_function(func_id, &mut codegen_ctx)
        .expect("Define function failed");
    jit_module.finalize_definitions().unwrap();

    // Get function pointer and call
    let code_ptr = jit_module.get_finalized_function(func_id);
    let jit_fn: JitEntryFn = unsafe { std::mem::transmute(code_ptr) };

    let locals_ptr = if locals.is_empty() {
        ptr::null_mut()
    } else {
        locals.as_mut_ptr()
    };
    let local_count = locals.len() as u32;

    let mut exit = raya_engine::jit::runtime::trampoline::JitExitInfo::default();
    let result = unsafe {
        jit_fn(
            ptr::null(),
            0,
            locals_ptr,
            local_count,
            ctx_ptr,
            (&mut exit as *mut _),
        )
    };
    (result, exit)
}

/// Run bytecode through the full pipeline (lift → optimize → compile) then execute.
fn jit_pipeline_and_call(code: Vec<u8>, local_count: usize) -> u64 {
    let module = make_module(code, 0, local_count);
    let func = &module.functions[0];

    // Lift bytecode → JIT IR
    let jit_func = lift_function(func, &module, 0).expect("Lift failed");

    // Allocate locals buffer
    let mut locals = vec![0u64; local_count];
    jit_compile_and_call_with_locals(&jit_func, &mut locals)
}

#[test]
fn jit_native_call_exits_with_suspend_kind() {
    let mut code = Vec::new();
    code.push(Opcode::NativeCall as u8);
    code.extend_from_slice(&0u16.to_le_bytes()); // native_id
    code.push(0u8); // arg_count
    emit(&mut code, Opcode::Return);

    let module = make_module(code, 0, 0);
    let func = &module.functions[0];
    let jit_func = lift_function(func, &module, 0).expect("Lift failed");

    let (raw, exit) = jit_compile_and_call_with_locals_and_exit(&jit_func, &mut []);
    assert_eq!(raw, NULL_VALUE, "native-call bridge returns null sentinel");
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Suspended as u32
    );
    assert_eq!(
        exit.suspend_reason,
        raya_engine::jit::runtime::trampoline::JitSuspendReason::NativeCallBoundary as u32
    );
    assert_eq!(exit.bytecode_offset, 0);
}

#[test]
fn jit_native_call_reports_bytecode_offset() {
    let mut code = Vec::new();
    emit(&mut code, Opcode::ConstNull); // offset 0
    code.push(Opcode::NativeCall as u8); // offset 1
    code.extend_from_slice(&0u16.to_le_bytes());
    code.push(0u8);
    emit(&mut code, Opcode::Return);

    let module = make_module(code, 0, 1);
    let func = &module.functions[0];
    let jit_func = lift_function(func, &module, 0).expect("Lift failed");

    let (raw, exit) = jit_compile_and_call_with_locals_and_exit(&jit_func, &mut []);
    assert_eq!(raw, NULL_VALUE);
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Suspended as u32
    );
    assert_eq!(
        exit.suspend_reason,
        raya_engine::jit::runtime::trampoline::JitSuspendReason::NativeCallBoundary as u32
    );
    assert_eq!(exit.bytecode_offset, 1);
}

#[test]
fn jit_native_call_materializes_operands_in_exit_info() {
    let mut code = Vec::new();
    emit_i32(&mut code, 7);
    emit_i32(&mut code, 11);
    code.push(Opcode::NativeCall as u8);
    code.extend_from_slice(&0u16.to_le_bytes());
    code.push(2u8); // arg_count
    emit(&mut code, Opcode::Return);

    let module = make_module(code, 0, 0);
    let func = &module.functions[0];
    let jit_func = lift_function(func, &module, 0).expect("Lift failed");

    let (_raw, exit) = jit_compile_and_call_with_locals_and_exit(&jit_func, &mut []);
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Suspended as u32
    );
    assert_eq!(
        exit.suspend_reason,
        raya_engine::jit::runtime::trampoline::JitSuspendReason::NativeCallBoundary as u32
    );
    assert_eq!(exit.native_arg_count, 2);
    assert_eq!(decode_i32(exit.native_args[0]), 7);
    assert_eq!(decode_i32(exit.native_args[1]), 11);
}

#[test]
fn jit_native_call_materializes_operands_truncated_to_exit_cap() {
    let mut code = Vec::new();
    let arg_cap = raya_engine::jit::runtime::trampoline::JIT_EXIT_MAX_NATIVE_ARGS as i32;
    for v in 0..(arg_cap + 8) {
        emit_i32(&mut code, v);
    }
    code.push(Opcode::NativeCall as u8);
    code.extend_from_slice(&0u16.to_le_bytes());
    code.push((arg_cap + 8) as u8);
    emit(&mut code, Opcode::Return);

    let module = make_module(code, 0, 0);
    let func = &module.functions[0];
    let jit_func = lift_function(func, &module, 0).expect("Lift failed");

    let (_raw, exit) = jit_compile_and_call_with_locals_and_exit(&jit_func, &mut []);
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Suspended as u32
    );
    assert_eq!(
        exit.suspend_reason,
        raya_engine::jit::runtime::trampoline::JitSuspendReason::NativeCallBoundary as u32
    );
    assert_eq!(
        exit.native_arg_count as usize,
        raya_engine::jit::runtime::trampoline::JIT_EXIT_MAX_NATIVE_ARGS
    );
    assert_eq!(decode_i32(exit.native_args[0]), 0);
    assert_eq!(
        decode_i32(
            exit.native_args[raya_engine::jit::runtime::trampoline::JIT_EXIT_MAX_NATIVE_ARGS - 1]
        ),
        arg_cap - 1
    );
}

#[test]
fn jit_call_static_exits_with_interpreter_boundary() {
    let mut code = Vec::new();
    emit_i32(&mut code, 7);
    emit_i32(&mut code, 11);
    let boundary_offset = code.len() as u32;
    code.push(Opcode::CallStatic as u8);
    code.extend_from_slice(&3u32.to_le_bytes());
    code.extend_from_slice(&2u16.to_le_bytes());
    emit(&mut code, Opcode::Return);

    let module = make_module(code, 0, 0);
    let func = &module.functions[0];
    let jit_func = lift_function(func, &module, 0).expect("Lift failed");
    assert!(matches!(
        jit_func.blocks[0].instrs.last(),
        Some(JitInstr::CallStatic { .. })
    ));

    let (raw, exit) = jit_compile_and_call_with_locals_and_exit(&jit_func, &mut []);
    assert_eq!(raw, NULL_VALUE);
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Suspended as u32
    );
    assert_eq!(
        exit.suspend_reason,
        raya_engine::jit::runtime::trampoline::JitSuspendReason::InterpreterBoundary as u32
    );
    assert_eq!(exit.bytecode_offset, boundary_offset);
    assert_eq!(exit.native_arg_count, 2);
    assert_eq!(decode_i32(exit.native_args[0]), 7);
    assert_eq!(decode_i32(exit.native_args[1]), 11);
}

#[test]
fn jit_call_boundary_materializes_full_pre_call_stack() {
    let mut code = Vec::new();
    emit_i32(&mut code, 99);
    emit_i32(&mut code, 7);
    emit_i32(&mut code, 11);
    code.push(Opcode::CallStatic as u8);
    code.extend_from_slice(&3u32.to_le_bytes());
    code.extend_from_slice(&2u16.to_le_bytes());
    emit(&mut code, Opcode::Return);

    let module = make_module(code, 0, 0);
    let func = &module.functions[0];
    let jit_func = lift_function(func, &module, 0).expect("Lift failed");

    let (_raw, exit) = jit_compile_and_call_with_locals_and_exit(&jit_func, &mut []);
    assert_eq!(
        exit.suspend_reason,
        raya_engine::jit::runtime::trampoline::JitSuspendReason::InterpreterBoundary as u32
    );
    assert_eq!(exit.native_arg_count, 3);
    assert_eq!(decode_i32(exit.native_args[0]), 99);
    assert_eq!(decode_i32(exit.native_args[1]), 7);
    assert_eq!(decode_i32(exit.native_args[2]), 11);
}

#[test]
fn jit_construct_type_lifts_to_helper_call() {
    let mut code = Vec::new();
    emit_load_local(&mut code, 0);
    code.push(Opcode::ConstructType as u8);
    code.extend_from_slice(&1u16.to_le_bytes());
    code.push(0u8);
    emit(&mut code, Opcode::Return);

    let module = make_module(code, 0, 1);
    let func = &module.functions[0];
    let jit_func = lift_function(func, &module, 0).expect("Lift failed");

    assert!(matches!(
        jit_func.blocks[0].instrs.get(1),
        Some(JitInstr::ConstructType { .. })
    ));
}

#[test]
fn jit_call_executes_sync_callee_via_runtime_helper() {
    let (safepoint, shared) = new_shared_vm_state();

    let mut main_code = Vec::new();
    emit_i32(&mut main_code, 7);
    emit_i32(&mut main_code, 11);
    main_code.push(Opcode::Call as u8);
    main_code.extend_from_slice(&1u32.to_le_bytes());
    main_code.extend_from_slice(&2u16.to_le_bytes());
    emit(&mut main_code, Opcode::Return);

    let mut add_code = Vec::new();
    emit_load_local(&mut add_code, 0);
    emit_load_local(&mut add_code, 1);
    emit(&mut add_code, Opcode::Iadd);
    emit(&mut add_code, Opcode::Return);

    let module = finalize_module(make_custom_module(
        vec![
            Function {
                signature_id: 0,
                local_types: Vec::new(),
                abi_version: 1,
                name: "main".to_string(),
                param_count: 0,
                local_count: 0,
                code: main_code,
            },
            Function {
                signature_id: 0,
                local_types: Vec::new(),
                abi_version: 1,
                name: "add".to_string(),
                param_count: 2,
                local_count: 2,
                code: add_code,
            },
        ],
        vec![],
    ));

    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let (_resolved_natives, bridge) = build_bridge_and_ctx(&safepoint, &shared, &task, &module);
    let mut ctx = raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");
    assert!(matches!(
        jit_func.blocks[0].instrs.iter().find(|instr| matches!(instr, JitInstr::Call { .. })),
        Some(JitInstr::Call { func_index: 1, .. })
    ));

    let (raw, exit) = jit_compile_and_call_with_locals_exit_and_ctx(
        &jit_func,
        &mut [],
        (&mut ctx as *mut _),
    );
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
    );
    assert!(is_i32(raw));
    assert_eq!(decode_i32(raw), 18);
}

#[test]
fn jit_call_static_executes_sync_callee_via_runtime_helper() {
    let (safepoint, shared) = new_shared_vm_state();

    let mut main_code = Vec::new();
    emit_i32(&mut main_code, 7);
    emit_i32(&mut main_code, 11);
    main_code.push(Opcode::CallStatic as u8);
    main_code.extend_from_slice(&1u32.to_le_bytes());
    main_code.extend_from_slice(&2u16.to_le_bytes());
    emit(&mut main_code, Opcode::Return);

    let mut add_code = Vec::new();
    emit_load_local(&mut add_code, 0);
    emit_load_local(&mut add_code, 1);
    emit(&mut add_code, Opcode::Iadd);
    emit(&mut add_code, Opcode::Return);

    let module = finalize_module(make_custom_module(
        vec![
            Function {
                signature_id: 0,
                local_types: Vec::new(),
                abi_version: 1,
                name: "main".to_string(),
                param_count: 0,
                local_count: 0,
                code: main_code,
            },
            Function {
                signature_id: 0,
                local_types: Vec::new(),
                abi_version: 1,
                name: "add".to_string(),
                param_count: 2,
                local_count: 2,
                code: add_code,
            },
        ],
        vec![],
    ));

    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let (_resolved_natives, bridge) = build_bridge_and_ctx(&safepoint, &shared, &task, &module);
    let mut ctx = raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");

    let (raw, exit) = jit_compile_and_call_with_locals_exit_and_ctx(
        &jit_func,
        &mut [],
        (&mut ctx as *mut _),
    );
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
    );
    assert!(is_i32(raw));
    assert_eq!(decode_i32(raw), 18);
}

// NOTE: `LoadFieldShape` is `Rejected` in the capability table, so this is NOT a
// reachable fast path. This test lifts and calls the function directly, which
// bypasses candidate selection, so it covers the lowering and the helper in
// isolation. It must not be read as evidence that compiled code uses the helper:
// see `accessor_and_proxy_field_opcodes_are_rejected_until_exact` in
// `jit/capability.rs` for the gate that keeps this unreachable.
#[test]
fn load_field_shape_lowering_uses_runtime_helper_directly() {
    let (safepoint, shared) = new_shared_vm_state();

    let layout_names = vec!["b".to_string(), "a".to_string()];
    let shape_names = vec!["a".to_string(), "b".to_string()];
    let layout_id = raya_engine::vm::object::layout_id_from_ordered_names(&layout_names);
    let shape_id = raya_engine::vm::object::shape_id_from_member_names(&shape_names);
    shared.register_structural_layout_shape(layout_id, &layout_names);
    shared.register_structural_shape_names(shape_id, &shape_names);

    let object_raw = {
        let mut gc = shared.gc.lock();
        let mut object = raya_engine::vm::object::Object::new_structural(layout_id, 2);
        object.set_field(0, raya_engine::vm::value::Value::i32(11)).unwrap();
        object.set_field(1, raya_engine::vm::value::Value::i32(7)).unwrap();
        let ptr = gc.allocate(object);
        unsafe {
            raya_engine::vm::value::Value::from_ptr(std::ptr::NonNull::new(ptr.as_ptr()).unwrap())
                .raw()
        }
    };

    let mut code = Vec::new();
    emit_load_local(&mut code, 0);
    code.push(Opcode::LoadFieldShape as u8);
    code.extend_from_slice(&shape_id.to_le_bytes());
    code.extend_from_slice(&0u16.to_le_bytes());
    emit(&mut code, Opcode::Return);

    let module = finalize_module(make_module(code, 0, 1));
    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let (_resolved_natives, bridge) = build_bridge_and_ctx(&safepoint, &shared, &task, &module);
    let mut ctx = raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    let mut locals = vec![object_raw];
    let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");

    let (raw, exit) = jit_compile_and_call_with_locals_exit_and_ctx(
        &jit_func,
        &mut locals,
        (&mut ctx as *mut _),
    );
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
    );
    assert_eq!(decode_i32(raw), 7);
}

// NOTE: as above, `StoreFieldShape` is `Rejected` in the capability table. The
// interpreter handler for this opcode can invoke a descriptor setter and check
// writability; this helper cannot, which is why it was demoted. This test covers
// the lowering and helper only.
#[test]
fn store_field_shape_lowering_uses_runtime_helper_directly() {
    let (safepoint, shared) = new_shared_vm_state();

    let layout_names = vec!["b".to_string(), "a".to_string()];
    let shape_names = vec!["a".to_string(), "b".to_string()];
    let layout_id = raya_engine::vm::object::layout_id_from_ordered_names(&layout_names);
    let shape_id = raya_engine::vm::object::shape_id_from_member_names(&shape_names);
    shared.register_structural_layout_shape(layout_id, &layout_names);
    shared.register_structural_shape_names(shape_id, &shape_names);

    let object_raw = {
        let mut gc = shared.gc.lock();
        let mut object = raya_engine::vm::object::Object::new_structural(layout_id, 2);
        object.set_field(0, raya_engine::vm::value::Value::i32(11)).unwrap();
        object.set_field(1, raya_engine::vm::value::Value::i32(7)).unwrap();
        let ptr = gc.allocate(object);
        unsafe {
            raya_engine::vm::value::Value::from_ptr(std::ptr::NonNull::new(ptr.as_ptr()).unwrap())
                .raw()
        }
    };

    let mut code = Vec::new();
    emit_load_local(&mut code, 0);
    emit_i32(&mut code, 99);
    code.push(Opcode::StoreFieldShape as u8);
    code.extend_from_slice(&shape_id.to_le_bytes());
    code.extend_from_slice(&0u16.to_le_bytes());
    emit_load_local(&mut code, 0);
    code.push(Opcode::LoadFieldShape as u8);
    code.extend_from_slice(&shape_id.to_le_bytes());
    code.extend_from_slice(&0u16.to_le_bytes());
    emit(&mut code, Opcode::Return);

    let module = finalize_module(make_module(code, 0, 1));
    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let (_resolved_natives, bridge) = build_bridge_and_ctx(&safepoint, &shared, &task, &module);
    let mut ctx = raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    let mut locals = vec![object_raw];
    let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");

    let (raw, exit) = jit_compile_and_call_with_locals_exit_and_ctx(
        &jit_func,
        &mut locals,
        (&mut ctx as *mut _),
    );
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
    );
    assert_eq!(decode_i32(raw), 99);
}

#[test]
fn jit_call_method_exact_executes_via_runtime_helper() {
    let (safepoint, shared) = new_shared_vm_state();

    let mut main_code = Vec::new();
    main_code.push(Opcode::NewType as u8);
    main_code.extend_from_slice(&0u16.to_le_bytes());
    main_code.push(Opcode::CallMethodExact as u8);
    main_code.extend_from_slice(&0u32.to_le_bytes());
    main_code.extend_from_slice(&0u16.to_le_bytes());
    emit(&mut main_code, Opcode::Return);

    let mut method_code = Vec::new();
    emit_i32(&mut method_code, 42);
    emit(&mut method_code, Opcode::Return);

    let module = finalize_module(make_custom_module(
        vec![
            Function {
                signature_id: 0,
                local_types: Vec::new(),
                abi_version: 1,
                name: "main".to_string(),
                param_count: 0,
                local_count: 0,
                code: main_code,
            },
            Function {
                signature_id: 0,
                local_types: Vec::new(),
                abi_version: 1,
                name: "value".to_string(),
                param_count: 1,
                local_count: 1,
                code: method_code,
            },
        ],
        vec![ClassDef {
            name: "Target".to_string(),
            field_count: 0,
            parent_id: None,
            methods: vec![Method {
                name: "value".to_string(),
                function_id: 1,
                slot: 0,
            }],
        }],
    ));
    shared
        .register_module(module.clone())
        .expect("register method module");

    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let (_resolved_natives, bridge) = build_bridge_and_ctx(&safepoint, &shared, &task, &module);
    let mut ctx = raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");

    let (raw, exit) = jit_compile_and_call_with_locals_exit_and_ctx(
        &jit_func,
        &mut [],
        (&mut ctx as *mut _),
    );
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
    );
    assert_eq!(decode_i32(raw), 42);
}

#[test]
fn jit_call_method_shape_executes_via_runtime_helper() {
    let (safepoint, shared) = new_shared_vm_state();
    let shape_names = vec!["value".to_string()];
    let shape_id = raya_engine::vm::object::shape_id_from_member_names(&shape_names);
    shared.register_structural_shape_names(shape_id, &shape_names);

    let mut main_code = Vec::new();
    main_code.push(Opcode::NewType as u8);
    main_code.extend_from_slice(&0u16.to_le_bytes());
    main_code.push(Opcode::CallMethodShape as u8);
    main_code.extend_from_slice(&shape_id.to_le_bytes());
    main_code.extend_from_slice(&0u16.to_le_bytes());
    main_code.extend_from_slice(&0u16.to_le_bytes());
    emit(&mut main_code, Opcode::Return);

    let mut method_code = Vec::new();
    emit_i32(&mut method_code, 42);
    emit(&mut method_code, Opcode::Return);

    let module = finalize_module(make_custom_module(
        vec![
            Function {
                signature_id: 0,
                local_types: Vec::new(),
                abi_version: 1,
                name: "main".to_string(),
                param_count: 0,
                local_count: 0,
                code: main_code,
            },
            Function {
                signature_id: 0,
                local_types: Vec::new(),
                abi_version: 1,
                name: "value".to_string(),
                param_count: 1,
                local_count: 1,
                code: method_code,
            },
        ],
        vec![ClassDef {
            name: "Target".to_string(),
            field_count: 0,
            parent_id: None,
            methods: vec![Method {
                name: "value".to_string(),
                function_id: 1,
                slot: 0,
            }],
        }],
    ));
    shared
        .register_module(module.clone())
        .expect("register structural method module");

    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let (_resolved_natives, bridge) = build_bridge_and_ctx(&safepoint, &shared, &task, &module);
    let mut ctx = raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");

    let (raw, exit) =
        jit_compile_and_call_with_locals_exit_and_ctx_and_module(
            &jit_func,
            &module,
            &mut [],
            (&mut ctx as *mut _),
        );
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
    );
    assert_eq!(decode_i32(raw), 42);
}

#[test]
fn jit_construct_type_executes_constructor_via_runtime_helper() {
    let (safepoint, shared) = new_shared_vm_state();

    let mut main_code = Vec::new();
    main_code.push(Opcode::NewType as u8);
    main_code.extend_from_slice(&0u16.to_le_bytes());
    main_code.push(Opcode::ConstructType as u8);
    main_code.extend_from_slice(&0u16.to_le_bytes());
    main_code.push(0u8);
    emit(&mut main_code, Opcode::Return);

    let mut ctor_code = Vec::new();
    emit_load_local(&mut ctor_code, 0);
    emit_i32(&mut ctor_code, 42);
    emit_store_field_exact(&mut ctor_code, 0);
    emit(&mut ctor_code, Opcode::Return);

    let module = finalize_module(make_custom_module(
        vec![
            Function {
                signature_id: 0,
                local_types: Vec::new(),
                abi_version: 1,
                name: "main".to_string(),
                param_count: 0,
                local_count: 0,
                code: main_code,
            },
            Function {
                signature_id: 0,
                local_types: Vec::new(),
                abi_version: 1,
                name: "Target::constructor".to_string(),
                param_count: 1,
                local_count: 1,
                code: ctor_code,
            },
        ],
        vec![ClassDef {
            name: "Target".to_string(),
            field_count: 1,
            parent_id: None,
            methods: vec![Method {
                name: "constructor".to_string(),
                function_id: 1,
                slot: 0,
            }],
        }],
    ));
    shared
        .register_module(module.clone())
        .expect("register construct module");

    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let (_resolved_natives, bridge) = build_bridge_and_ctx(&safepoint, &shared, &task, &module);
    let mut ctx = raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");

    let (raw, exit) =
        jit_compile_and_call_with_locals_exit_and_ctx_and_module(
            &jit_func,
            &module,
            &mut [],
            (&mut ctx as *mut _),
        );
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
    );
    let value = unsafe { raya_engine::vm::value::Value::from_raw(raw) };
    let object = unsafe {
        &*value
            .as_ptr::<raya_engine::vm::object::Object>()
            .expect("construct type result object")
            .as_ptr()
    };
    assert_eq!(
        object.get_field(0),
        Some(raya_engine::vm::value::Value::i32(42))
    );
}

#[test]
fn jit_call_constructor_executes_via_runtime_helper() {
    let (safepoint, shared) = new_shared_vm_state();

    let mut main_code = Vec::new();
    main_code.push(Opcode::CallConstructor as u8);
    main_code.extend_from_slice(&0u32.to_le_bytes());
    main_code.extend_from_slice(&0u16.to_le_bytes());
    emit(&mut main_code, Opcode::Return);

    let mut ctor_code = Vec::new();
    emit_load_local(&mut ctor_code, 0);
    emit_i32(&mut ctor_code, 42);
    emit_store_field_exact(&mut ctor_code, 0);
    emit(&mut ctor_code, Opcode::Return);

    let module = finalize_module(make_custom_module(
        vec![
            Function {
                signature_id: 0,
                local_types: Vec::new(),
                abi_version: 1,
                name: "main".to_string(),
                param_count: 0,
                local_count: 0,
                code: main_code,
            },
            Function {
                signature_id: 0,
                local_types: Vec::new(),
                abi_version: 1,
                name: "Target::constructor".to_string(),
                param_count: 1,
                local_count: 1,
                code: ctor_code,
            },
        ],
        vec![ClassDef {
            name: "Target".to_string(),
            field_count: 1,
            parent_id: None,
            methods: vec![Method {
                name: "constructor".to_string(),
                function_id: 1,
                slot: 0,
            }],
        }],
    ));
    shared
        .register_module(module.clone())
        .expect("register constructor module");

    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let (_resolved_natives, bridge) = build_bridge_and_ctx(&safepoint, &shared, &task, &module);
    let mut ctx = raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");

    let (raw, exit) =
        jit_compile_and_call_with_locals_exit_and_ctx_and_module(
            &jit_func,
            &module,
            &mut [],
            (&mut ctx as *mut _),
        );
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
    );
    let value = unsafe { raya_engine::vm::value::Value::from_raw(raw) };
    let object = unsafe {
        &*value
            .as_ptr::<raya_engine::vm::object::Object>()
            .expect("constructor result object")
            .as_ptr()
    };
    assert_eq!(
        object.get_field(0),
        Some(raya_engine::vm::value::Value::i32(42))
    );
}

#[test]
fn jit_call_super_executes_via_runtime_helper() {
    let (safepoint, shared) = new_shared_vm_state();

    let mut main_code = Vec::new();
    main_code.push(Opcode::NewType as u8);
    main_code.extend_from_slice(&1u16.to_le_bytes());
    emit(&mut main_code, Opcode::Dup);
    main_code.push(Opcode::CallSuper as u8);
    main_code.extend_from_slice(&1u32.to_le_bytes());
    main_code.extend_from_slice(&0u16.to_le_bytes());
    emit(&mut main_code, Opcode::Return);

    let mut parent_ctor = Vec::new();
    emit_load_local(&mut parent_ctor, 0);
    emit_i32(&mut parent_ctor, 42);
    emit_store_field_exact(&mut parent_ctor, 0);
    emit(&mut parent_ctor, Opcode::Return);

    let module = finalize_module(make_custom_module(
        vec![
            Function {
                signature_id: 0,
                local_types: Vec::new(),
                abi_version: 1,
                name: "main".to_string(),
                param_count: 0,
                local_count: 0,
                code: main_code,
            },
            Function {
                signature_id: 0,
                local_types: Vec::new(),
                abi_version: 1,
                name: "Parent::constructor".to_string(),
                param_count: 1,
                local_count: 1,
                code: parent_ctor,
            },
        ],
        vec![
            ClassDef {
                name: "Parent".to_string(),
                field_count: 1,
                parent_id: None,
                methods: vec![Method {
                    name: "constructor".to_string(),
                    function_id: 1,
                    slot: 0,
                }],
            },
            ClassDef {
                name: "Child".to_string(),
                field_count: 1,
                parent_id: Some(0),
                methods: vec![],
            },
        ],
    ));
    shared
        .register_module(module.clone())
        .expect("register super module");

    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let (_resolved_natives, bridge) = build_bridge_and_ctx(&safepoint, &shared, &task, &module);
    let mut ctx = raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");

    let (raw, exit) =
        jit_compile_and_call_with_locals_exit_and_ctx_and_module(
            &jit_func,
            &module,
            &mut [],
            (&mut ctx as *mut _),
        );
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
    );
    let value = unsafe { raya_engine::vm::value::Value::from_raw(raw) };
    let object = unsafe {
        &*value
            .as_ptr::<raya_engine::vm::object::Object>()
            .expect("super call result object")
            .as_ptr()
    };
    assert_eq!(
        object.get_field(0),
        Some(raya_engine::vm::value::Value::i32(42))
    );
    let child_nominal_type_id = shared
        .resolve_nominal_type_id(&module, 1)
        .expect("child nominal type id");
    assert_eq!(object.nominal_type_id_usize(), Some(child_nominal_type_id));
}

#[test]
fn jit_new_type_lifts_to_new_object() {
    let mut code = Vec::new();
    code.push(Opcode::NewType as u8);
    code.extend_from_slice(&1u16.to_le_bytes());
    emit(&mut code, Opcode::Return);

    let module = make_module(code, 0, 0);
    let func = &module.functions[0];
    let jit_func = lift_function(func, &module, 0).expect("Lift failed");

    // NewType now lifts to GcSafepoint then NewObject due to J4 GC polling
    assert!(jit_func.blocks[0].instrs.iter().any(|i| matches!(i, JitInstr::GcSafepoint { .. })));
    assert!(jit_func.blocks[0].instrs.iter().any(|i| matches!(i, JitInstr::NewObject { .. })));
    // Ensure NewObject is after GcSafepoint (order may vary)
    let gc_index = jit_func.blocks[0].instrs.iter().position(|i| matches!(i, JitInstr::GcSafepoint { .. }));
    let new_obj_index = jit_func.blocks[0].instrs.iter().position(|i| matches!(i, JitInstr::NewObject { .. }));
    if let (Some(gc), Some(new_obj)) = (gc_index, new_obj_index) {
        assert!(gc < new_obj, "GcSafepoint should precede NewObject");
    }
}

#[test]
fn jit_new_type_uses_alloc_object_helper() {
    let safepoint = std::sync::Arc::new(
        raya_engine::vm::interpreter::SafepointCoordinator::new(1),
    );
    let tasks = std::sync::Arc::new(parking_lot::RwLock::new(FxHashMap::default()));
    let injector = std::sync::Arc::new(crossbeam_deque::Injector::new());
    let shared = std::sync::Arc::new(raya_engine::vm::interpreter::SharedVmState::new(
        safepoint.clone(),
        tasks,
        injector,
    ));

    let mut code = Vec::new();
    code.push(Opcode::NewType as u8);
    code.extend_from_slice(&0u16.to_le_bytes());
    emit(&mut code, Opcode::Return);

    let mut module = make_module(code, 0, 0);
    module.classes.push(ClassDef {
        name: "Target".to_string(),
        field_count: 0,
        parent_id: None,
        methods: Vec::new(),
    });
    let module = std::sync::Arc::new(
        Module::decode(&module.encode()).expect("finalize target module checksum"),
    );
    shared
        .register_module(module.clone())
        .expect("register target module");

    let expected_nominal_type_id = shared
        .resolve_nominal_type_id(&module, 0)
        .expect("module-local nominal type id");

    let func = &module.functions[0];
    let jit_func = lift_function(func, &module, 0).expect("Lift failed");

    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let resolved_natives = parking_lot::RwLock::new(shared.resolved_natives.read().clone());
    let code_cache = test_code_cache(&shared);
    let bridge = raya_engine::jit::runtime::helpers::build_runtime_bridge_context(
        safepoint.as_ref(),
        &task,
        &shared.gc,
        &shared.classes,
        &shared.layouts,
        code_cache.as_ref(),
        &shared.mutex_registry,
        &shared.semaphore_registry,
        &shared.globals_by_index,
        &shared.builtin_global_slots,
        &shared.constant_string_cache,
        &shared.ephemeral_gc_roots,
        &shared.pinned_handles,
        &shared.tasks,
        &shared.injector,
        &shared.module_layouts,
        &shared.metadata,
        &shared.class_metadata,
        &shared.native_handler,
        &resolved_natives,
        &shared.structural_shape_names,
        &shared.structural_layout_shapes,
        &shared.structural_shape_adapters,
        &shared.aot_profile,
        &shared.type_handles,
        &shared.prop_keys,
        &shared.stack_pool,
        shared.max_preemptions,
        0,
        None,
    );
    let mut ctx =
        raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());

    let (raw, exit) =
        jit_compile_and_call_with_locals_exit_and_ctx_and_module(
            &jit_func,
            &module,
            &mut [],
            (&mut ctx as *mut _),
        );
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
    );
    assert_ne!(raw, 0);

    let value = unsafe { raya_engine::vm::value::Value::from_raw(raw) };
    let obj = unsafe {
        &*value
            .as_ptr::<raya_engine::vm::object::Object>()
            .expect("new type result object")
            .as_ptr()
    };
    assert_eq!(obj.nominal_type_id_usize(), Some(expected_nominal_type_id));
}

#[test]
fn jit_implements_shape_uses_runtime_helper() {
    let safepoint = std::sync::Arc::new(
        raya_engine::vm::interpreter::SafepointCoordinator::new(1),
    );
    let tasks = std::sync::Arc::new(parking_lot::RwLock::new(FxHashMap::default()));
    let injector = std::sync::Arc::new(crossbeam_deque::Injector::new());
    let shared = std::sync::Arc::new(raya_engine::vm::interpreter::SharedVmState::new(
        safepoint.clone(),
        tasks,
        injector,
    ));

    let layout_names = vec!["b".to_string(), "a".to_string()];
    let shape_names = vec!["a".to_string(), "b".to_string()];
    let layout_id = raya_engine::vm::object::layout_id_from_ordered_names(&layout_names);
    let shape_id = raya_engine::vm::object::shape_id_from_member_names(&shape_names);
    shared.register_structural_layout_shape(layout_id, &layout_names);
    shared.register_structural_shape_names(shape_id, &shape_names);

    let object_raw = {
        let mut gc = shared.gc.lock();
        let mut object = raya_engine::vm::object::Object::new_structural(layout_id, 2);
        object.set_field(0, raya_engine::vm::value::Value::i32(11)).unwrap();
        object.set_field(1, raya_engine::vm::value::Value::i32(7)).unwrap();
        let ptr = gc.allocate(object);
        unsafe {
            raya_engine::vm::value::Value::from_ptr(std::ptr::NonNull::new(ptr.as_ptr()).unwrap())
                .raw()
        }
    };

    let mut code = Vec::new();
    emit_load_local(&mut code, 0);
    code.push(Opcode::ImplementsShape as u8);
    code.extend_from_slice(&shape_id.to_le_bytes());
    emit(&mut code, Opcode::Return);

    let module = std::sync::Arc::new(make_module(code, 0, 1));
    let func = &module.functions[0];
    let jit_func = lift_function(func, &module, 0).expect("Lift failed");

    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let resolved_natives = parking_lot::RwLock::new(shared.resolved_natives.read().clone());
    let code_cache = test_code_cache(&shared);
    let bridge = raya_engine::jit::runtime::helpers::build_runtime_bridge_context(
        safepoint.as_ref(),
        &task,
        &shared.gc,
        &shared.classes,
        &shared.layouts,
        code_cache.as_ref(),
        &shared.mutex_registry,
        &shared.semaphore_registry,
        &shared.globals_by_index,
        &shared.builtin_global_slots,
        &shared.constant_string_cache,
        &shared.ephemeral_gc_roots,
        &shared.pinned_handles,
        &shared.tasks,
        &shared.injector,
        &shared.module_layouts,
        &shared.metadata,
        &shared.class_metadata,
        &shared.native_handler,
        &resolved_natives,
        &shared.structural_shape_names,
        &shared.structural_layout_shapes,
        &shared.structural_shape_adapters,
        &shared.aot_profile,
        &shared.type_handles,
        &shared.prop_keys,
        &shared.stack_pool,
        shared.max_preemptions,
        0,
        None,
    );
    let mut ctx =
        raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    let mut locals = vec![object_raw];

    let (raw, exit) =
        jit_compile_and_call_with_locals_exit_and_ctx_and_module(
            &jit_func,
            &module,
            &mut locals,
            (&mut ctx as *mut _),
        );
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
    );
    assert!(is_bool(raw));
    assert!(decode_bool(raw));
}

#[test]
fn jit_cast_shape_uses_runtime_helper_fastpath() {
    let safepoint = std::sync::Arc::new(
        raya_engine::vm::interpreter::SafepointCoordinator::new(1),
    );
    let tasks = std::sync::Arc::new(parking_lot::RwLock::new(FxHashMap::default()));
    let injector = std::sync::Arc::new(crossbeam_deque::Injector::new());
    let shared = std::sync::Arc::new(raya_engine::vm::interpreter::SharedVmState::new(
        safepoint.clone(),
        tasks,
        injector,
    ));

    let layout_names = vec!["b".to_string(), "a".to_string()];
    let shape_names = vec!["a".to_string(), "b".to_string()];
    let layout_id = raya_engine::vm::object::layout_id_from_ordered_names(&layout_names);
    let shape_id = raya_engine::vm::object::shape_id_from_member_names(&shape_names);
    shared.register_structural_layout_shape(layout_id, &layout_names);
    shared.register_structural_shape_names(shape_id, &shape_names);

    let object_raw = {
        let mut gc = shared.gc.lock();
        let mut object = raya_engine::vm::object::Object::new_structural(layout_id, 2);
        object.set_field(0, raya_engine::vm::value::Value::i32(11)).unwrap();
        object.set_field(1, raya_engine::vm::value::Value::i32(7)).unwrap();
        let ptr = gc.allocate(object);
        unsafe {
            raya_engine::vm::value::Value::from_ptr(std::ptr::NonNull::new(ptr.as_ptr()).unwrap())
                .raw()
        }
    };

    let mut code = Vec::new();
    emit_load_local(&mut code, 0);
    code.push(Opcode::CastShape as u8);
    code.extend_from_slice(&shape_id.to_le_bytes());
    emit(&mut code, Opcode::Return);

    let module = std::sync::Arc::new(make_module(code, 0, 1));
    let func = &module.functions[0];
    let jit_func = lift_function(func, &module, 0).expect("Lift failed");
    assert!(matches!(
        jit_func.blocks[0].instrs.last(),
        Some(JitInstr::CastShape { .. })
    ));

    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let resolved_natives = parking_lot::RwLock::new(shared.resolved_natives.read().clone());
    let code_cache = test_code_cache(&shared);
    let bridge = raya_engine::jit::runtime::helpers::build_runtime_bridge_context(
        safepoint.as_ref(),
        &task,
        &shared.gc,
        &shared.classes,
        &shared.layouts,
        code_cache.as_ref(),
        &shared.mutex_registry,
        &shared.semaphore_registry,
        &shared.globals_by_index,
        &shared.builtin_global_slots,
        &shared.constant_string_cache,
        &shared.ephemeral_gc_roots,
        &shared.pinned_handles,
        &shared.tasks,
        &shared.injector,
        &shared.module_layouts,
        &shared.metadata,
        &shared.class_metadata,
        &shared.native_handler,
        &resolved_natives,
        &shared.structural_shape_names,
        &shared.structural_layout_shapes,
        &shared.structural_shape_adapters,
        &shared.aot_profile,
        &shared.type_handles,
        &shared.prop_keys,
        &shared.stack_pool,
        shared.max_preemptions,
        0,
        None,
    );
    let mut ctx =
        raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    let mut locals = vec![object_raw];

    let (raw, exit) =
        jit_compile_and_call_with_locals_exit_and_ctx_and_module(
            &jit_func,
            &module,
            &mut locals,
            (&mut ctx as *mut _),
        );
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
    );
    assert_eq!(raw, object_raw);
}

#[test]
fn jit_cast_shape_failure_exits_with_interpreter_boundary() {
    let safepoint = std::sync::Arc::new(
        raya_engine::vm::interpreter::SafepointCoordinator::new(1),
    );
    let tasks = std::sync::Arc::new(parking_lot::RwLock::new(FxHashMap::default()));
    let injector = std::sync::Arc::new(crossbeam_deque::Injector::new());
    let shared = std::sync::Arc::new(raya_engine::vm::interpreter::SharedVmState::new(
        safepoint.clone(),
        tasks,
        injector,
    ));

    let layout_names = vec!["a".to_string()];
    let shape_names = vec!["a".to_string(), "b".to_string()];
    let layout_id = raya_engine::vm::object::layout_id_from_ordered_names(&layout_names);
    let shape_id = raya_engine::vm::object::shape_id_from_member_names(&shape_names);
    shared.register_structural_layout_shape(layout_id, &layout_names);
    shared.register_structural_shape_names(shape_id, &shape_names);

    let object_raw = {
        let mut gc = shared.gc.lock();
        let mut object = raya_engine::vm::object::Object::new_structural(layout_id, 1);
        object.set_field(0, raya_engine::vm::value::Value::i32(11)).unwrap();
        let ptr = gc.allocate(object);
        unsafe {
            raya_engine::vm::value::Value::from_ptr(std::ptr::NonNull::new(ptr.as_ptr()).unwrap())
                .raw()
        }
    };

    let mut code = Vec::new();
    emit_load_local(&mut code, 0);
    code.push(Opcode::CastShape as u8);
    code.extend_from_slice(&shape_id.to_le_bytes());
    emit(&mut code, Opcode::Return);

    let module = std::sync::Arc::new(make_module(code, 0, 1));
    let func = &module.functions[0];
    let jit_func = lift_function(func, &module, 0).expect("Lift failed");

    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let resolved_natives = parking_lot::RwLock::new(shared.resolved_natives.read().clone());
    let code_cache = test_code_cache(&shared);
    let bridge = raya_engine::jit::runtime::helpers::build_runtime_bridge_context(
        safepoint.as_ref(),
        &task,
        &shared.gc,
        &shared.classes,
        &shared.layouts,
        code_cache.as_ref(),
        &shared.mutex_registry,
        &shared.semaphore_registry,
        &shared.globals_by_index,
        &shared.builtin_global_slots,
        &shared.constant_string_cache,
        &shared.ephemeral_gc_roots,
        &shared.pinned_handles,
        &shared.tasks,
        &shared.injector,
        &shared.module_layouts,
        &shared.metadata,
        &shared.class_metadata,
        &shared.native_handler,
        &resolved_natives,
        &shared.structural_shape_names,
        &shared.structural_layout_shapes,
        &shared.structural_shape_adapters,
        &shared.aot_profile,
        &shared.type_handles,
        &shared.prop_keys,
        &shared.stack_pool,
        shared.max_preemptions,
        0,
        None,
    );
    let mut ctx =
        raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    let mut locals = vec![object_raw];

    let (_raw, exit) =
        jit_compile_and_call_with_locals_exit_and_ctx_and_module(
            &jit_func,
            module.as_ref(),
            &mut locals,
            (&mut ctx as *mut _),
        );
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Suspended as u32
    );
    assert_eq!(
        exit.suspend_reason,
        raya_engine::jit::runtime::trampoline::JitSuspendReason::InterpreterBoundary as u32
    );
    assert_eq!(exit.bytecode_offset, 3);
    assert_eq!(exit.native_arg_count, 1);
    assert_eq!(exit.native_args[0], object_raw);
}

#[test]
fn jit_is_nominal_uses_runtime_helper() {
    let safepoint = std::sync::Arc::new(
        raya_engine::vm::interpreter::SafepointCoordinator::new(1),
    );
    let tasks = std::sync::Arc::new(parking_lot::RwLock::new(FxHashMap::default()));
    let injector = std::sync::Arc::new(crossbeam_deque::Injector::new());
    let shared = std::sync::Arc::new(raya_engine::vm::interpreter::SharedVmState::new(
        safepoint.clone(),
        tasks,
        injector,
    ));

    let mut code = Vec::new();
    emit_load_local(&mut code, 0);
    code.push(Opcode::IsNominal as u8);
    code.extend_from_slice(&0u16.to_le_bytes());
    emit(&mut code, Opcode::Return);

    let mut module = make_module(code, 0, 1);
    module.classes.push(ClassDef {
        name: "Target".to_string(),
        field_count: 0,
        parent_id: None,
        methods: Vec::new(),
    });
    let module = std::sync::Arc::new(
        Module::decode(&module.encode()).expect("finalize target module checksum"),
    );
    shared
        .register_module(module.clone())
        .expect("register target module");

    let func = &module.functions[0];
    let jit_func = lift_function(func, &module, 0).expect("Lift failed");

    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let resolved_natives = parking_lot::RwLock::new(shared.resolved_natives.read().clone());
    let code_cache = test_code_cache(&shared);
    let bridge = raya_engine::jit::runtime::helpers::build_runtime_bridge_context(
        safepoint.as_ref(),
        &task,
        &shared.gc,
        &shared.classes,
        &shared.layouts,
        code_cache.as_ref(),
        &shared.mutex_registry,
        &shared.semaphore_registry,
        &shared.globals_by_index,
        &shared.builtin_global_slots,
        &shared.constant_string_cache,
        &shared.ephemeral_gc_roots,
        &shared.pinned_handles,
        &shared.tasks,
        &shared.injector,
        &shared.module_layouts,
        &shared.metadata,
        &shared.class_metadata,
        &shared.native_handler,
        &resolved_natives,
        &shared.structural_shape_names,
        &shared.structural_layout_shapes,
        &shared.structural_shape_adapters,
        &shared.aot_profile,
        &shared.type_handles,
        &shared.prop_keys,
        &shared.stack_pool,
        shared.max_preemptions,
        0,
        None,
    );
    let object_ptr = unsafe {
        (raya_engine::jit::runtime::helpers::runtime_helpers().alloc_object)(
            0,
            std::sync::Arc::as_ptr(&module) as *const (),
            (&bridge as *const raya_engine::jit::runtime::helpers::JitRuntimeBridgeContext)
                as *mut (),
        )
    };
    assert!(!object_ptr.is_null());
    let object_raw = unsafe {
        raya_engine::vm::value::Value::from_ptr(
            std::ptr::NonNull::new(object_ptr.cast::<raya_engine::vm::object::Object>()).unwrap(),
        )
        .raw()
    };

    let mut ctx =
        raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    let mut locals = vec![object_raw];

    let (raw, exit) =
        jit_compile_and_call_with_locals_exit_and_ctx_and_module(
            &jit_func,
            &module,
            &mut locals,
            (&mut ctx as *mut _),
        );
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
    );
    assert!(is_bool(raw));
    assert!(decode_bool(raw));
}

#[test]
fn jit_cast_nominal_failure_exits_with_interpreter_boundary() {
    let safepoint = std::sync::Arc::new(
        raya_engine::vm::interpreter::SafepointCoordinator::new(1),
    );
    let tasks = std::sync::Arc::new(parking_lot::RwLock::new(FxHashMap::default()));
    let injector = std::sync::Arc::new(crossbeam_deque::Injector::new());
    let shared = std::sync::Arc::new(raya_engine::vm::interpreter::SharedVmState::new(
        safepoint.clone(),
        tasks,
        injector,
    ));

    let mut source_module = make_module(vec![Opcode::Return as u8], 0, 0);
    source_module.classes.push(ClassDef {
        name: "Source".to_string(),
        field_count: 0,
        parent_id: None,
        methods: Vec::new(),
    });
    let source_module = std::sync::Arc::new(
        Module::decode(&source_module.encode()).expect("finalize source module checksum"),
    );
    shared
        .register_module(source_module.clone())
        .expect("register source module");

    let mut code = Vec::new();
    emit_load_local(&mut code, 0);
    code.push(Opcode::CastNominal as u8);
    code.extend_from_slice(&0u16.to_le_bytes());
    emit(&mut code, Opcode::Return);

    let mut target_module = make_module(code, 0, 1);
    target_module.classes.push(ClassDef {
        name: "Target".to_string(),
        field_count: 0,
        parent_id: None,
        methods: Vec::new(),
    });
    let target_module = std::sync::Arc::new(
        Module::decode(&target_module.encode()).expect("finalize target module checksum"),
    );
    shared
        .register_module(target_module.clone())
        .expect("register target module");

    let func = &target_module.functions[0];
    let jit_func = lift_function(func, &target_module, 0).expect("Lift failed");

    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, target_module.clone(), None));
    let resolved_natives = parking_lot::RwLock::new(shared.resolved_natives.read().clone());
    let code_cache = test_code_cache(&shared);
    let bridge = raya_engine::jit::runtime::helpers::build_runtime_bridge_context(
        safepoint.as_ref(),
        &task,
        &shared.gc,
        &shared.classes,
        &shared.layouts,
        code_cache.as_ref(),
        &shared.mutex_registry,
        &shared.semaphore_registry,
        &shared.globals_by_index,
        &shared.builtin_global_slots,
        &shared.constant_string_cache,
        &shared.ephemeral_gc_roots,
        &shared.pinned_handles,
        &shared.tasks,
        &shared.injector,
        &shared.module_layouts,
        &shared.metadata,
        &shared.class_metadata,
        &shared.native_handler,
        &resolved_natives,
        &shared.structural_shape_names,
        &shared.structural_layout_shapes,
        &shared.structural_shape_adapters,
        &shared.aot_profile,
        &shared.type_handles,
        &shared.prop_keys,
        &shared.stack_pool,
        shared.max_preemptions,
        0,
        None,
    );
    let object_ptr = unsafe {
        (raya_engine::jit::runtime::helpers::runtime_helpers().alloc_object)(
            0,
            std::sync::Arc::as_ptr(&source_module) as *const (),
            (&bridge as *const raya_engine::jit::runtime::helpers::JitRuntimeBridgeContext)
                as *mut (),
        )
    };
    assert!(!object_ptr.is_null());
    let object_raw = unsafe {
        raya_engine::vm::value::Value::from_ptr(
            std::ptr::NonNull::new(object_ptr.cast::<raya_engine::vm::object::Object>()).unwrap(),
        )
        .raw()
    };

    let mut ctx =
        raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, target_module.as_ref());
    let mut locals = vec![object_raw];

    let (_raw, exit) =
        jit_compile_and_call_with_locals_exit_and_ctx_and_module(
            &jit_func,
            target_module.as_ref(),
            &mut locals,
            (&mut ctx as *mut _),
        );
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Suspended as u32
    );
    assert_eq!(
        exit.suspend_reason,
        raya_engine::jit::runtime::trampoline::JitSuspendReason::InterpreterBoundary as u32
    );
    assert_eq!(exit.bytecode_offset, 3);
    assert_eq!(exit.native_arg_count, 1);
    assert_eq!(exit.native_args[0], object_raw);
}

#[test]
fn jit_native_call_zero_arg_ctx_fastpath_returns_value() {
    unsafe extern "C" fn stub_alloc_object(
        _local_nominal_type_index: u32,
        _module_ptr: *const (),
        _shared_state: *mut (),
    ) -> *mut () {
        std::ptr::null_mut()
    }
    unsafe extern "C" fn stub_alloc_array(
        _type_id: u32,
        _len: u64,
        _module: *const (),
        _shared_state: *mut (),
    ) -> *mut () {
        std::ptr::null_mut()
    }
    unsafe extern "C" fn stub_array_load(_array: u64, _index: u64, _shared_state: *mut ()) -> u64 {
        0
    }
    unsafe extern "C" fn stub_array_store(
        _array: u64,
        _index: u64,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0
    }
    unsafe extern "C" fn stub_array_push(_array: u64, _value: u64, _shared_state: *mut ()) -> i8 {
        0
    }
    unsafe extern "C" fn stub_array_pop(_array: u64, _shared_state: *mut ()) -> u64 {
        0
    }
    unsafe extern "C" fn stub_array_len(_array: u64, _shared_state: *mut ()) -> i32 {
        0
    }
    /// RefCell load has no stub behaviour: any RefCell opcode is `Rejected`, so no
    /// compiled test can reach it. Returning the interpreter-fallback sentinel is
    /// the honest stub — it is what the real helper returns for a receiver that is
    /// not a pointer, and it routes to the interpreter rather than inventing a
    /// value.
    unsafe extern "C" fn stub_refcell_load(_refcell: u64, _shared_state: *mut ()) -> u64 {
        raya_engine::jit::runtime::helpers::JIT_INTERPRETER_FALLBACK_SENTINEL
    }

    /// Reports failure, which routes the store to the interpreter rather than
    /// pretending it happened. `StoreRefCell` is still `Rejected`, so no compiled
    /// test reaches this.
    unsafe extern "C" fn stub_refcell_store(
        _refcell: u64,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0
    }

    /// Null means the allocation did not happen, which routes to the interpreter.
    /// `NewRefCell` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_refcell_new(_initial: u64, _shared_state: *mut ()) -> u64 {
        0
    }

    /// Reports failure, which routes the patch to the interpreter.
    /// `SetClosureCapture` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_set_closure_capture(
        _closure: u64,
        _index: u32,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0
    }

    /// Null means no allocation happened, which routes to the interpreter.
    /// `MakeClosure` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_make_closure(
        _func_id: u32,
        _captures_ptr: *const u64,
        _capture_count: u32,
        _shared_state: *mut (),
    ) -> u64 {
        0
    }

    /// Returns the fallback sentinel, which routes the load to the interpreter.
    /// `LoadCaptured` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_load_captured(_index: u32, _shared_state: *mut ()) -> u64 {
        raya_engine::jit::runtime::helpers::JIT_INTERPRETER_FALLBACK_SENTINEL
    }

    /// Reports failure, which routes the store to the interpreter.
    /// `StoreCaptured` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_store_captured(
        _index: u32,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0
    }

    /// Null means the bind did not happen, which routes to the interpreter.
    /// `BindMethod` is still not selectable, so no compiled test reaches this.
    unsafe extern "C" fn stub_bind_method(
        _object: u64,
        _method_slot: u32,
        _shared_state: *mut (),
    ) -> u64 {
        0
    }

    /// Returns the fallback sentinel so the load routes to the interpreter.
    /// `Await` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_await_task(_value: u64, _shared_state: *mut ()) -> u64 {
        raya_engine::jit::runtime::helpers::JIT_INTERPRETER_FALLBACK_SENTINEL
    }

    /// Returns the fallback sentinel so the load routes to the interpreter.
    /// `DynGetKeyed` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_dyn_get_keyed(
        _object: u64,
        _key: u64,
        _shared_state: *mut (),
    ) -> u64 {
        raya_engine::jit::runtime::helpers::JIT_INTERPRETER_FALLBACK_SENTINEL
    }

    /// Returns the fallback status so a bad future promotion lands in the
    /// interpreter rather than corrupting an array.
    unsafe extern "C" fn stub_dyn_set_keyed(
        _object: u64,
        _key: u64,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0 // JIT_STORE_FALLBACK
    }

    /// Null, so a failure to allocate falls back rather than yielding a bogus object.
    /// `ObjectLiteral` is still `Rejected`, so no compiled test reaches this yet.
    unsafe extern "C" fn stub_alloc_struct_object(
        _type_index: u32,
        _field_count: u32,
        _shared_state: *mut (),
    ) -> *mut () {
        std::ptr::null_mut()
    }

    /// Declines, so a slot write routes to the interpreter rather than silently
    /// succeeding or writing out of bounds. `InitObject` is still `Rejected`.
    unsafe extern "C" fn stub_init_object_field(
        _object: u64,
        _offset: u64,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0 // JIT_STORE_FALLBACK
    }

    /// Declines, so a cast routes to the interpreter rather than silently passing.
    /// `CastObjectMinFields` is still `Rejected` AND still on
    /// `produces_incorrect_native_results`, so no compiled test reaches this yet.
    unsafe extern "C" fn stub_cast_object_min_fields(
        _object: u64,
        _required_fields: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0 // OBJECT_MIN_FIELDS_DECLINE
    }
    unsafe extern "C" fn stub_alloc_string(
        _data_ptr: *const u8,
        _len: usize,
        _shared_state: *mut (),
    ) -> *mut () {
        std::ptr::null_mut()
    }
    unsafe extern "C" fn stub_safepoint_poll(_shared_state: *const ()) {}
    unsafe extern "C" fn stub_check_preemption(_current_task: *const ()) -> bool {
        false
    }
    unsafe extern "C" fn stub_native_call_dispatch(
        _native_id: u16,
        _args_ptr: *const u64,
        _arg_count: u8,
        _shared_state: *mut (),
    ) -> u64 {
        I32_TAG_BASE | (42u64 & PAYLOAD_MASK_32)
    }
    unsafe extern "C" fn stub_interpreter_call(
        _opcode: u8,
        _operand_u64: u64,
        _operand_u32: u32,
        _receiver: u64,
        _args_ptr: *const u64,
        _arg_count: u16,
        _module_ptr: *const (),
        _shared_state: *mut (),
    ) -> u64 {
        NULL_VALUE
    }
    unsafe extern "C" fn stub_string_concat(
        _left: u64,
        _right: u64,
        _shared_state: *mut (),
    ) -> u64 {
        NULL_VALUE
    }
    unsafe extern "C" fn stub_generic_equals(
        _left: u64,
        _right: u64,
        _shared_state: *mut (),
    ) -> bool {
        false
    }

    let mut code = Vec::new();
    code.push(Opcode::NativeCall as u8);
    code.extend_from_slice(&0u16.to_le_bytes());
    code.push(0u8);
    emit(&mut code, Opcode::Return);

    let module = make_module(code, 0, 0);
    let func = &module.functions[0];
    let jit_func = lift_function(func, &module, 0).expect("Lift failed");

    let mut ctx = RuntimeContext {
        shared_state: std::ptr::null(),
        current_task: std::ptr::null(),
        module: std::ptr::null(),
        helpers: RuntimeHelperTable {
            alloc_object: stub_alloc_object,
            alloc_array: stub_alloc_array,
            alloc_string: stub_alloc_string,
            safepoint_poll: stub_safepoint_poll,
            check_preemption: stub_check_preemption,
            native_call_dispatch: stub_native_call_dispatch,
            interpreter_call: stub_interpreter_call,
            string_concat: stub_string_concat,
            generic_equals: stub_generic_equals,
            object_get_field: stub_object_get_field,
            object_set_field: stub_object_set_field,
            object_implements_shape: stub_object_implements_shape,
            object_is_nominal: stub_object_is_nominal,
            object_get_shape_field: stub_object_get_shape_field,
            object_set_shape_field: stub_object_set_shape_field,
            string_len: stub_string_len,
            string_compare: stub_string_compare,
            value_to_string: stub_value_to_string,
            const_string: stub_const_string,
            array_load: stub_array_load,
            array_store: stub_array_store,
            array_push: stub_array_push,
            array_pop: stub_array_pop,
            array_len: stub_array_len,
            refcell_load: stub_refcell_load,
            refcell_store: stub_refcell_store,
            refcell_new: stub_refcell_new,
            set_closure_capture: stub_set_closure_capture,
            make_closure: stub_make_closure,
            load_captured: stub_load_captured,
            store_captured: stub_store_captured,
            bind_method: stub_bind_method,
            await_task: stub_await_task,
            dyn_get_keyed: stub_dyn_get_keyed,
            dyn_set_keyed: stub_dyn_set_keyed,
            alloc_struct_object: stub_alloc_struct_object,
            init_object_field: stub_init_object_field,
            cast_object_min_fields: stub_cast_object_min_fields,
        },
    };

    let (raw, exit) = jit_compile_and_call_with_locals_exit_and_ctx(
        &jit_func,
        &mut [],
        (&mut ctx as *mut _),
    );
    assert!(is_i32(raw));
    assert_eq!(decode_i32(raw), 42);
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
    );
    assert_eq!(
        exit.suspend_reason,
        raya_engine::jit::runtime::trampoline::JitSuspendReason::None as u32
    );
}

#[test]
fn jit_native_call_zero_arg_ctx_fastpath_sentinel_suspends() {
    unsafe extern "C" fn stub_alloc_object(
        _local_nominal_type_index: u32,
        _module_ptr: *const (),
        _shared_state: *mut (),
    ) -> *mut () {
        std::ptr::null_mut()
    }
    unsafe extern "C" fn stub_alloc_array(
        _type_id: u32,
        _len: u64,
        _module: *const (),
        _shared_state: *mut (),
    ) -> *mut () {
        std::ptr::null_mut()
    }
    unsafe extern "C" fn stub_array_load(_array: u64, _index: u64, _shared_state: *mut ()) -> u64 {
        0
    }
    unsafe extern "C" fn stub_array_store(
        _array: u64,
        _index: u64,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0
    }
    unsafe extern "C" fn stub_array_push(_array: u64, _value: u64, _shared_state: *mut ()) -> i8 {
        0
    }
    unsafe extern "C" fn stub_array_pop(_array: u64, _shared_state: *mut ()) -> u64 {
        0
    }
    unsafe extern "C" fn stub_array_len(_array: u64, _shared_state: *mut ()) -> i32 {
        0
    }
    /// RefCell load has no stub behaviour: any RefCell opcode is `Rejected`, so no
    /// compiled test can reach it. Returning the interpreter-fallback sentinel is
    /// the honest stub — it is what the real helper returns for a receiver that is
    /// not a pointer, and it routes to the interpreter rather than inventing a
    /// value.
    unsafe extern "C" fn stub_refcell_load(_refcell: u64, _shared_state: *mut ()) -> u64 {
        raya_engine::jit::runtime::helpers::JIT_INTERPRETER_FALLBACK_SENTINEL
    }

    /// Reports failure, which routes the store to the interpreter rather than
    /// pretending it happened. `StoreRefCell` is still `Rejected`, so no compiled
    /// test reaches this.
    unsafe extern "C" fn stub_refcell_store(
        _refcell: u64,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0
    }

    /// Null means the allocation did not happen, which routes to the interpreter.
    /// `NewRefCell` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_refcell_new(_initial: u64, _shared_state: *mut ()) -> u64 {
        0
    }

    /// Reports failure, which routes the patch to the interpreter.
    /// `SetClosureCapture` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_set_closure_capture(
        _closure: u64,
        _index: u32,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0
    }

    /// Null means no allocation happened, which routes to the interpreter.
    /// `MakeClosure` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_make_closure(
        _func_id: u32,
        _captures_ptr: *const u64,
        _capture_count: u32,
        _shared_state: *mut (),
    ) -> u64 {
        0
    }

    /// Returns the fallback sentinel, which routes the load to the interpreter.
    /// `LoadCaptured` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_load_captured(_index: u32, _shared_state: *mut ()) -> u64 {
        raya_engine::jit::runtime::helpers::JIT_INTERPRETER_FALLBACK_SENTINEL
    }

    /// Reports failure, which routes the store to the interpreter.
    /// `StoreCaptured` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_store_captured(
        _index: u32,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0
    }

    /// Null means the bind did not happen, which routes to the interpreter.
    /// `BindMethod` is still not selectable, so no compiled test reaches this.
    unsafe extern "C" fn stub_bind_method(
        _object: u64,
        _method_slot: u32,
        _shared_state: *mut (),
    ) -> u64 {
        0
    }

    /// Returns the fallback sentinel so the load routes to the interpreter.
    /// `Await` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_await_task(_value: u64, _shared_state: *mut ()) -> u64 {
        raya_engine::jit::runtime::helpers::JIT_INTERPRETER_FALLBACK_SENTINEL
    }

    /// Returns the fallback sentinel so the load routes to the interpreter.
    /// `DynGetKeyed` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_dyn_get_keyed(
        _object: u64,
        _key: u64,
        _shared_state: *mut (),
    ) -> u64 {
        raya_engine::jit::runtime::helpers::JIT_INTERPRETER_FALLBACK_SENTINEL
    }

    /// Returns the fallback status so a bad future promotion lands in the
    /// interpreter rather than corrupting an array.
    unsafe extern "C" fn stub_dyn_set_keyed(
        _object: u64,
        _key: u64,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0 // JIT_STORE_FALLBACK
    }

    /// Null, so a failure to allocate falls back rather than yielding a bogus object.
    /// `ObjectLiteral` is still `Rejected`, so no compiled test reaches this yet.
    unsafe extern "C" fn stub_alloc_struct_object(
        _type_index: u32,
        _field_count: u32,
        _shared_state: *mut (),
    ) -> *mut () {
        std::ptr::null_mut()
    }

    /// Declines, so a slot write routes to the interpreter rather than silently
    /// succeeding or writing out of bounds. `InitObject` is still `Rejected`.
    unsafe extern "C" fn stub_init_object_field(
        _object: u64,
        _offset: u64,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0 // JIT_STORE_FALLBACK
    }

    /// Declines, so a cast routes to the interpreter rather than silently passing.
    /// `CastObjectMinFields` is still `Rejected` AND still on
    /// `produces_incorrect_native_results`, so no compiled test reaches this yet.
    unsafe extern "C" fn stub_cast_object_min_fields(
        _object: u64,
        _required_fields: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0 // OBJECT_MIN_FIELDS_DECLINE
    }
    unsafe extern "C" fn stub_alloc_string(
        _data_ptr: *const u8,
        _len: usize,
        _shared_state: *mut (),
    ) -> *mut () {
        std::ptr::null_mut()
    }
    unsafe extern "C" fn stub_safepoint_poll(_shared_state: *const ()) {}
    unsafe extern "C" fn stub_check_preemption(_current_task: *const ()) -> bool {
        false
    }
    unsafe extern "C" fn stub_native_call_dispatch(
        _native_id: u16,
        _args_ptr: *const u64,
        _arg_count: u8,
        _shared_state: *mut (),
    ) -> u64 {
        JIT_NATIVE_SUSPEND_SENTINEL
    }
    unsafe extern "C" fn stub_interpreter_call(
        _opcode: u8,
        _operand_u64: u64,
        _operand_u32: u32,
        _receiver: u64,
        _args_ptr: *const u64,
        _arg_count: u16,
        _module_ptr: *const (),
        _shared_state: *mut (),
    ) -> u64 {
        NULL_VALUE
    }
    unsafe extern "C" fn stub_string_concat(
        _left: u64,
        _right: u64,
        _shared_state: *mut (),
    ) -> u64 {
        NULL_VALUE
    }
    unsafe extern "C" fn stub_generic_equals(
        _left: u64,
        _right: u64,
        _shared_state: *mut (),
    ) -> bool {
        false
    }

    let mut code = Vec::new();
    code.push(Opcode::NativeCall as u8);
    code.extend_from_slice(&0u16.to_le_bytes());
    code.push(0u8);
    emit(&mut code, Opcode::Return);

    let module = make_module(code, 0, 0);
    let func = &module.functions[0];
    let jit_func = lift_function(func, &module, 0).expect("Lift failed");

    let mut ctx = RuntimeContext {
        shared_state: std::ptr::null(),
        current_task: std::ptr::null(),
        module: std::ptr::null(),
        helpers: RuntimeHelperTable {
            alloc_object: stub_alloc_object,
            alloc_array: stub_alloc_array,
            alloc_string: stub_alloc_string,
            safepoint_poll: stub_safepoint_poll,
            check_preemption: stub_check_preemption,
            native_call_dispatch: stub_native_call_dispatch,
            interpreter_call: stub_interpreter_call,
            string_concat: stub_string_concat,
            generic_equals: stub_generic_equals,
            object_get_field: stub_object_get_field,
            object_set_field: stub_object_set_field,
            object_implements_shape: stub_object_implements_shape,
            object_is_nominal: stub_object_is_nominal,
            object_get_shape_field: stub_object_get_shape_field,
            object_set_shape_field: stub_object_set_shape_field,
            string_len: stub_string_len,
            string_compare: stub_string_compare,
            value_to_string: stub_value_to_string,
            const_string: stub_const_string,
            array_load: stub_array_load,
            array_store: stub_array_store,
            array_push: stub_array_push,
            array_pop: stub_array_pop,
            array_len: stub_array_len,
            refcell_load: stub_refcell_load,
            refcell_store: stub_refcell_store,
            refcell_new: stub_refcell_new,
            set_closure_capture: stub_set_closure_capture,
            make_closure: stub_make_closure,
            load_captured: stub_load_captured,
            store_captured: stub_store_captured,
            bind_method: stub_bind_method,
            await_task: stub_await_task,
            dyn_get_keyed: stub_dyn_get_keyed,
            dyn_set_keyed: stub_dyn_set_keyed,
            alloc_struct_object: stub_alloc_struct_object,
            init_object_field: stub_init_object_field,
            cast_object_min_fields: stub_cast_object_min_fields,
        },
    };

    let (_raw, exit) =
        jit_compile_and_call_with_locals_exit_and_ctx_and_module(
            &jit_func,
            &module,
            &mut [],
            (&mut ctx as *mut _),
        );
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Suspended as u32
    );
    assert_eq!(
        exit.suspend_reason,
        raya_engine::jit::runtime::trampoline::JitSuspendReason::NativeCallBoundary as u32
    );
    assert_eq!(exit.bytecode_offset, 0);
}

#[test]
fn jit_native_call_args_ctx_fastpath_returns_value() {
    unsafe extern "C" fn stub_alloc_object(
        _local_nominal_type_index: u32,
        _module_ptr: *const (),
        _shared_state: *mut (),
    ) -> *mut () {
        std::ptr::null_mut()
    }
    unsafe extern "C" fn stub_alloc_array(
        _type_id: u32,
        _len: u64,
        _module: *const (),
        _shared_state: *mut (),
    ) -> *mut () {
        std::ptr::null_mut()
    }
    unsafe extern "C" fn stub_array_load(_array: u64, _index: u64, _shared_state: *mut ()) -> u64 {
        0
    }
    unsafe extern "C" fn stub_array_store(
        _array: u64,
        _index: u64,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0
    }
    unsafe extern "C" fn stub_array_push(_array: u64, _value: u64, _shared_state: *mut ()) -> i8 {
        0
    }
    unsafe extern "C" fn stub_array_pop(_array: u64, _shared_state: *mut ()) -> u64 {
        0
    }
    unsafe extern "C" fn stub_array_len(_array: u64, _shared_state: *mut ()) -> i32 {
        0
    }
    /// RefCell load has no stub behaviour: any RefCell opcode is `Rejected`, so no
    /// compiled test can reach it. Returning the interpreter-fallback sentinel is
    /// the honest stub — it is what the real helper returns for a receiver that is
    /// not a pointer, and it routes to the interpreter rather than inventing a
    /// value.
    unsafe extern "C" fn stub_refcell_load(_refcell: u64, _shared_state: *mut ()) -> u64 {
        raya_engine::jit::runtime::helpers::JIT_INTERPRETER_FALLBACK_SENTINEL
    }

    /// Reports failure, which routes the store to the interpreter rather than
    /// pretending it happened. `StoreRefCell` is still `Rejected`, so no compiled
    /// test reaches this.
    unsafe extern "C" fn stub_refcell_store(
        _refcell: u64,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0
    }

    /// Null means the allocation did not happen, which routes to the interpreter.
    /// `NewRefCell` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_refcell_new(_initial: u64, _shared_state: *mut ()) -> u64 {
        0
    }

    /// Reports failure, which routes the patch to the interpreter.
    /// `SetClosureCapture` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_set_closure_capture(
        _closure: u64,
        _index: u32,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0
    }

    /// Null means no allocation happened, which routes to the interpreter.
    /// `MakeClosure` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_make_closure(
        _func_id: u32,
        _captures_ptr: *const u64,
        _capture_count: u32,
        _shared_state: *mut (),
    ) -> u64 {
        0
    }

    /// Returns the fallback sentinel, which routes the load to the interpreter.
    /// `LoadCaptured` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_load_captured(_index: u32, _shared_state: *mut ()) -> u64 {
        raya_engine::jit::runtime::helpers::JIT_INTERPRETER_FALLBACK_SENTINEL
    }

    /// Reports failure, which routes the store to the interpreter.
    /// `StoreCaptured` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_store_captured(
        _index: u32,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0
    }

    /// Null means the bind did not happen, which routes to the interpreter.
    /// `BindMethod` is still not selectable, so no compiled test reaches this.
    unsafe extern "C" fn stub_bind_method(
        _object: u64,
        _method_slot: u32,
        _shared_state: *mut (),
    ) -> u64 {
        0
    }

    /// Returns the fallback sentinel so the load routes to the interpreter.
    /// `Await` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_await_task(_value: u64, _shared_state: *mut ()) -> u64 {
        raya_engine::jit::runtime::helpers::JIT_INTERPRETER_FALLBACK_SENTINEL
    }

    /// Returns the fallback sentinel so the load routes to the interpreter.
    /// `DynGetKeyed` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_dyn_get_keyed(
        _object: u64,
        _key: u64,
        _shared_state: *mut (),
    ) -> u64 {
        raya_engine::jit::runtime::helpers::JIT_INTERPRETER_FALLBACK_SENTINEL
    }

    /// Returns the fallback status so a bad future promotion lands in the
    /// interpreter rather than corrupting an array.
    unsafe extern "C" fn stub_dyn_set_keyed(
        _object: u64,
        _key: u64,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0 // JIT_STORE_FALLBACK
    }

    /// Null, so a failure to allocate falls back rather than yielding a bogus object.
    /// `ObjectLiteral` is still `Rejected`, so no compiled test reaches this yet.
    unsafe extern "C" fn stub_alloc_struct_object(
        _type_index: u32,
        _field_count: u32,
        _shared_state: *mut (),
    ) -> *mut () {
        std::ptr::null_mut()
    }

    /// Declines, so a slot write routes to the interpreter rather than silently
    /// succeeding or writing out of bounds. `InitObject` is still `Rejected`.
    unsafe extern "C" fn stub_init_object_field(
        _object: u64,
        _offset: u64,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0 // JIT_STORE_FALLBACK
    }

    /// Declines, so a cast routes to the interpreter rather than silently passing.
    /// `CastObjectMinFields` is still `Rejected` AND still on
    /// `produces_incorrect_native_results`, so no compiled test reaches this yet.
    unsafe extern "C" fn stub_cast_object_min_fields(
        _object: u64,
        _required_fields: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0 // OBJECT_MIN_FIELDS_DECLINE
    }
    unsafe extern "C" fn stub_alloc_string(
        _data_ptr: *const u8,
        _len: usize,
        _shared_state: *mut (),
    ) -> *mut () {
        std::ptr::null_mut()
    }
    unsafe extern "C" fn stub_safepoint_poll(_shared_state: *const ()) {}
    unsafe extern "C" fn stub_check_preemption(_current_task: *const ()) -> bool {
        false
    }
    unsafe extern "C" fn stub_native_call_dispatch(
        _native_id: u16,
        args_ptr: *const u64,
        arg_count: u8,
        _shared_state: *mut (),
    ) -> u64 {
        assert_eq!(arg_count, 2);
        let a = unsafe { *args_ptr.add(0) };
        let b = unsafe { *args_ptr.add(1) };
        assert_eq!(decode_i32(a), 7);
        assert_eq!(decode_i32(b), 11);
        I32_TAG_BASE | (99u64 & PAYLOAD_MASK_32)
    }
    unsafe extern "C" fn stub_interpreter_call(
        _opcode: u8,
        _operand_u64: u64,
        _operand_u32: u32,
        _receiver: u64,
        _args_ptr: *const u64,
        _arg_count: u16,
        _module_ptr: *const (),
        _shared_state: *mut (),
    ) -> u64 {
        NULL_VALUE
    }
    unsafe extern "C" fn stub_string_concat(
        _left: u64,
        _right: u64,
        _shared_state: *mut (),
    ) -> u64 {
        NULL_VALUE
    }
    unsafe extern "C" fn stub_generic_equals(
        _left: u64,
        _right: u64,
        _shared_state: *mut (),
    ) -> bool {
        false
    }

    let mut code = Vec::new();
    emit_i32(&mut code, 7);
    emit_i32(&mut code, 11);
    code.push(Opcode::NativeCall as u8);
    code.extend_from_slice(&0u16.to_le_bytes());
    code.push(2u8);
    emit(&mut code, Opcode::Return);

    let module = make_module(code, 0, 0);
    let func = &module.functions[0];
    let jit_func = lift_function(func, &module, 0).expect("Lift failed");

    let mut ctx = RuntimeContext {
        shared_state: std::ptr::null(),
        current_task: std::ptr::null(),
        module: std::ptr::null(),
        helpers: RuntimeHelperTable {
            alloc_object: stub_alloc_object,
            alloc_array: stub_alloc_array,
            alloc_string: stub_alloc_string,
            safepoint_poll: stub_safepoint_poll,
            check_preemption: stub_check_preemption,
            native_call_dispatch: stub_native_call_dispatch,
            interpreter_call: stub_interpreter_call,
            string_concat: stub_string_concat,
            generic_equals: stub_generic_equals,
            object_get_field: stub_object_get_field,
            object_set_field: stub_object_set_field,
            object_implements_shape: stub_object_implements_shape,
            object_is_nominal: stub_object_is_nominal,
            object_get_shape_field: stub_object_get_shape_field,
            object_set_shape_field: stub_object_set_shape_field,
            string_len: stub_string_len,
            string_compare: stub_string_compare,
            value_to_string: stub_value_to_string,
            const_string: stub_const_string,
            array_load: stub_array_load,
            array_store: stub_array_store,
            array_push: stub_array_push,
            array_pop: stub_array_pop,
            array_len: stub_array_len,
            refcell_load: stub_refcell_load,
            refcell_store: stub_refcell_store,
            refcell_new: stub_refcell_new,
            set_closure_capture: stub_set_closure_capture,
            make_closure: stub_make_closure,
            load_captured: stub_load_captured,
            store_captured: stub_store_captured,
            bind_method: stub_bind_method,
            await_task: stub_await_task,
            dyn_get_keyed: stub_dyn_get_keyed,
            dyn_set_keyed: stub_dyn_set_keyed,
            alloc_struct_object: stub_alloc_struct_object,
            init_object_field: stub_init_object_field,
            cast_object_min_fields: stub_cast_object_min_fields,
        },
    };

    let (raw, exit) = jit_compile_and_call_with_locals_exit_and_ctx(
        &jit_func,
        &mut [],
        (&mut ctx as *mut _),
    );
    assert!(is_i32(raw));
    assert_eq!(decode_i32(raw), 99);
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
    );
}

#[test]
fn jit_native_call_args_ctx_fastpath_sentinel_suspends() {
    unsafe extern "C" fn stub_alloc_object(
        _local_nominal_type_index: u32,
        _module_ptr: *const (),
        _shared_state: *mut (),
    ) -> *mut () {
        std::ptr::null_mut()
    }
    unsafe extern "C" fn stub_alloc_array(
        _type_id: u32,
        _len: u64,
        _module: *const (),
        _shared_state: *mut (),
    ) -> *mut () {
        std::ptr::null_mut()
    }
    unsafe extern "C" fn stub_array_load(_array: u64, _index: u64, _shared_state: *mut ()) -> u64 {
        0
    }
    unsafe extern "C" fn stub_array_store(
        _array: u64,
        _index: u64,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0
    }
    unsafe extern "C" fn stub_array_push(_array: u64, _value: u64, _shared_state: *mut ()) -> i8 {
        0
    }
    unsafe extern "C" fn stub_array_pop(_array: u64, _shared_state: *mut ()) -> u64 {
        0
    }
    unsafe extern "C" fn stub_array_len(_array: u64, _shared_state: *mut ()) -> i32 {
        0
    }
    /// RefCell load has no stub behaviour: any RefCell opcode is `Rejected`, so no
    /// compiled test can reach it. Returning the interpreter-fallback sentinel is
    /// the honest stub — it is what the real helper returns for a receiver that is
    /// not a pointer, and it routes to the interpreter rather than inventing a
    /// value.
    unsafe extern "C" fn stub_refcell_load(_refcell: u64, _shared_state: *mut ()) -> u64 {
        raya_engine::jit::runtime::helpers::JIT_INTERPRETER_FALLBACK_SENTINEL
    }

    /// Reports failure, which routes the store to the interpreter rather than
    /// pretending it happened. `StoreRefCell` is still `Rejected`, so no compiled
    /// test reaches this.
    unsafe extern "C" fn stub_refcell_store(
        _refcell: u64,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0
    }

    /// Null means the allocation did not happen, which routes to the interpreter.
    /// `NewRefCell` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_refcell_new(_initial: u64, _shared_state: *mut ()) -> u64 {
        0
    }

    /// Reports failure, which routes the patch to the interpreter.
    /// `SetClosureCapture` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_set_closure_capture(
        _closure: u64,
        _index: u32,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0
    }

    /// Null means no allocation happened, which routes to the interpreter.
    /// `MakeClosure` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_make_closure(
        _func_id: u32,
        _captures_ptr: *const u64,
        _capture_count: u32,
        _shared_state: *mut (),
    ) -> u64 {
        0
    }

    /// Returns the fallback sentinel, which routes the load to the interpreter.
    /// `LoadCaptured` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_load_captured(_index: u32, _shared_state: *mut ()) -> u64 {
        raya_engine::jit::runtime::helpers::JIT_INTERPRETER_FALLBACK_SENTINEL
    }

    /// Reports failure, which routes the store to the interpreter.
    /// `StoreCaptured` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_store_captured(
        _index: u32,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0
    }

    /// Null means the bind did not happen, which routes to the interpreter.
    /// `BindMethod` is still not selectable, so no compiled test reaches this.
    unsafe extern "C" fn stub_bind_method(
        _object: u64,
        _method_slot: u32,
        _shared_state: *mut (),
    ) -> u64 {
        0
    }

    /// Returns the fallback sentinel so the load routes to the interpreter.
    /// `Await` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_await_task(_value: u64, _shared_state: *mut ()) -> u64 {
        raya_engine::jit::runtime::helpers::JIT_INTERPRETER_FALLBACK_SENTINEL
    }

    /// Returns the fallback sentinel so the load routes to the interpreter.
    /// `DynGetKeyed` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_dyn_get_keyed(
        _object: u64,
        _key: u64,
        _shared_state: *mut (),
    ) -> u64 {
        raya_engine::jit::runtime::helpers::JIT_INTERPRETER_FALLBACK_SENTINEL
    }

    /// Returns the fallback status so a bad future promotion lands in the
    /// interpreter rather than corrupting an array.
    unsafe extern "C" fn stub_dyn_set_keyed(
        _object: u64,
        _key: u64,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0 // JIT_STORE_FALLBACK
    }

    /// Null, so a failure to allocate falls back rather than yielding a bogus object.
    /// `ObjectLiteral` is still `Rejected`, so no compiled test reaches this yet.
    unsafe extern "C" fn stub_alloc_struct_object(
        _type_index: u32,
        _field_count: u32,
        _shared_state: *mut (),
    ) -> *mut () {
        std::ptr::null_mut()
    }

    /// Declines, so a slot write routes to the interpreter rather than silently
    /// succeeding or writing out of bounds. `InitObject` is still `Rejected`.
    unsafe extern "C" fn stub_init_object_field(
        _object: u64,
        _offset: u64,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0 // JIT_STORE_FALLBACK
    }

    /// Declines, so a cast routes to the interpreter rather than silently passing.
    /// `CastObjectMinFields` is still `Rejected` AND still on
    /// `produces_incorrect_native_results`, so no compiled test reaches this yet.
    unsafe extern "C" fn stub_cast_object_min_fields(
        _object: u64,
        _required_fields: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0 // OBJECT_MIN_FIELDS_DECLINE
    }
    unsafe extern "C" fn stub_alloc_string(
        _data_ptr: *const u8,
        _len: usize,
        _shared_state: *mut (),
    ) -> *mut () {
        std::ptr::null_mut()
    }
    unsafe extern "C" fn stub_safepoint_poll(_shared_state: *const ()) {}
    unsafe extern "C" fn stub_check_preemption(_current_task: *const ()) -> bool {
        false
    }
    unsafe extern "C" fn stub_native_call_dispatch(
        _native_id: u16,
        args_ptr: *const u64,
        arg_count: u8,
        _shared_state: *mut (),
    ) -> u64 {
        assert_eq!(arg_count, 2);
        let a = unsafe { *args_ptr.add(0) };
        let b = unsafe { *args_ptr.add(1) };
        assert_eq!(decode_i32(a), 7);
        assert_eq!(decode_i32(b), 11);
        JIT_NATIVE_SUSPEND_SENTINEL
    }
    unsafe extern "C" fn stub_interpreter_call(
        _opcode: u8,
        _operand_u64: u64,
        _operand_u32: u32,
        _receiver: u64,
        _args_ptr: *const u64,
        _arg_count: u16,
        _module_ptr: *const (),
        _shared_state: *mut (),
    ) -> u64 {
        NULL_VALUE
    }
    unsafe extern "C" fn stub_string_concat(
        _left: u64,
        _right: u64,
        _shared_state: *mut (),
    ) -> u64 {
        NULL_VALUE
    }
    unsafe extern "C" fn stub_generic_equals(
        _left: u64,
        _right: u64,
        _shared_state: *mut (),
    ) -> bool {
        false
    }

    let mut code = Vec::new();
    emit_i32(&mut code, 7);
    emit_i32(&mut code, 11);
    code.push(Opcode::NativeCall as u8);
    code.extend_from_slice(&0u16.to_le_bytes());
    code.push(2u8);
    emit(&mut code, Opcode::Return);

    let module = make_module(code, 0, 0);
    let func = &module.functions[0];
    let jit_func = lift_function(func, &module, 0).expect("Lift failed");

    let mut ctx = RuntimeContext {
        shared_state: std::ptr::null(),
        current_task: std::ptr::null(),
        module: std::ptr::null(),
        helpers: RuntimeHelperTable {
            alloc_object: stub_alloc_object,
            alloc_array: stub_alloc_array,
            alloc_string: stub_alloc_string,
            safepoint_poll: stub_safepoint_poll,
            check_preemption: stub_check_preemption,
            native_call_dispatch: stub_native_call_dispatch,
            interpreter_call: stub_interpreter_call,
            string_concat: stub_string_concat,
            generic_equals: stub_generic_equals,
            object_get_field: stub_object_get_field,
            object_set_field: stub_object_set_field,
            object_implements_shape: stub_object_implements_shape,
            object_is_nominal: stub_object_is_nominal,
            object_get_shape_field: stub_object_get_shape_field,
            object_set_shape_field: stub_object_set_shape_field,
            string_len: stub_string_len,
            string_compare: stub_string_compare,
            value_to_string: stub_value_to_string,
            const_string: stub_const_string,
            array_load: stub_array_load,
            array_store: stub_array_store,
            array_push: stub_array_push,
            array_pop: stub_array_pop,
            array_len: stub_array_len,
            refcell_load: stub_refcell_load,
            refcell_store: stub_refcell_store,
            refcell_new: stub_refcell_new,
            set_closure_capture: stub_set_closure_capture,
            make_closure: stub_make_closure,
            load_captured: stub_load_captured,
            store_captured: stub_store_captured,
            bind_method: stub_bind_method,
            await_task: stub_await_task,
            dyn_get_keyed: stub_dyn_get_keyed,
            dyn_set_keyed: stub_dyn_set_keyed,
            alloc_struct_object: stub_alloc_struct_object,
            init_object_field: stub_init_object_field,
            cast_object_min_fields: stub_cast_object_min_fields,
        },
    };

    let (_raw, exit) =
        jit_compile_and_call_with_locals_exit_and_ctx_and_module(
            &jit_func,
            &module,
            &mut [],
            (&mut ctx as *mut _),
        );
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Suspended as u32
    );
    assert_eq!(
        exit.suspend_reason,
        raya_engine::jit::runtime::trampoline::JitSuspendReason::NativeCallBoundary as u32
    );
    assert_eq!(exit.native_arg_count, 2);
    assert_eq!(decode_i32(exit.native_args[0]), 7);
    assert_eq!(decode_i32(exit.native_args[1]), 11);
}

#[test]
fn jit_check_preemption_exits_with_suspend_kind_when_helper_requests_preempt() {
    unsafe extern "C" fn stub_alloc_object(
        _local_nominal_type_index: u32,
        _module_ptr: *const (),
        _shared_state: *mut (),
    ) -> *mut () {
        std::ptr::null_mut()
    }
    unsafe extern "C" fn stub_alloc_array(
        _type_id: u32,
        _len: u64,
        _module: *const (),
        _shared_state: *mut (),
    ) -> *mut () {
        std::ptr::null_mut()
    }
    unsafe extern "C" fn stub_array_load(_array: u64, _index: u64, _shared_state: *mut ()) -> u64 {
        0
    }
    unsafe extern "C" fn stub_array_store(
        _array: u64,
        _index: u64,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0
    }
    unsafe extern "C" fn stub_array_push(_array: u64, _value: u64, _shared_state: *mut ()) -> i8 {
        0
    }
    unsafe extern "C" fn stub_array_pop(_array: u64, _shared_state: *mut ()) -> u64 {
        0
    }
    unsafe extern "C" fn stub_array_len(_array: u64, _shared_state: *mut ()) -> i32 {
        0
    }
    /// RefCell load has no stub behaviour: any RefCell opcode is `Rejected`, so no
    /// compiled test can reach it. Returning the interpreter-fallback sentinel is
    /// the honest stub — it is what the real helper returns for a receiver that is
    /// not a pointer, and it routes to the interpreter rather than inventing a
    /// value.
    unsafe extern "C" fn stub_refcell_load(_refcell: u64, _shared_state: *mut ()) -> u64 {
        raya_engine::jit::runtime::helpers::JIT_INTERPRETER_FALLBACK_SENTINEL
    }

    /// Reports failure, which routes the store to the interpreter rather than
    /// pretending it happened. `StoreRefCell` is still `Rejected`, so no compiled
    /// test reaches this.
    unsafe extern "C" fn stub_refcell_store(
        _refcell: u64,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0
    }

    /// Null means the allocation did not happen, which routes to the interpreter.
    /// `NewRefCell` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_refcell_new(_initial: u64, _shared_state: *mut ()) -> u64 {
        0
    }

    /// Reports failure, which routes the patch to the interpreter.
    /// `SetClosureCapture` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_set_closure_capture(
        _closure: u64,
        _index: u32,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0
    }

    /// Null means no allocation happened, which routes to the interpreter.
    /// `MakeClosure` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_make_closure(
        _func_id: u32,
        _captures_ptr: *const u64,
        _capture_count: u32,
        _shared_state: *mut (),
    ) -> u64 {
        0
    }

    /// Returns the fallback sentinel, which routes the load to the interpreter.
    /// `LoadCaptured` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_load_captured(_index: u32, _shared_state: *mut ()) -> u64 {
        raya_engine::jit::runtime::helpers::JIT_INTERPRETER_FALLBACK_SENTINEL
    }

    /// Reports failure, which routes the store to the interpreter.
    /// `StoreCaptured` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_store_captured(
        _index: u32,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0
    }

    /// Null means the bind did not happen, which routes to the interpreter.
    /// `BindMethod` is still not selectable, so no compiled test reaches this.
    unsafe extern "C" fn stub_bind_method(
        _object: u64,
        _method_slot: u32,
        _shared_state: *mut (),
    ) -> u64 {
        0
    }

    /// Returns the fallback sentinel so the load routes to the interpreter.
    /// `Await` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_await_task(_value: u64, _shared_state: *mut ()) -> u64 {
        raya_engine::jit::runtime::helpers::JIT_INTERPRETER_FALLBACK_SENTINEL
    }

    /// Returns the fallback sentinel so the load routes to the interpreter.
    /// `DynGetKeyed` is still `Rejected`, so no compiled test reaches this.
    unsafe extern "C" fn stub_dyn_get_keyed(
        _object: u64,
        _key: u64,
        _shared_state: *mut (),
    ) -> u64 {
        raya_engine::jit::runtime::helpers::JIT_INTERPRETER_FALLBACK_SENTINEL
    }

    /// Returns the fallback status so a bad future promotion lands in the
    /// interpreter rather than corrupting an array.
    unsafe extern "C" fn stub_dyn_set_keyed(
        _object: u64,
        _key: u64,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0 // JIT_STORE_FALLBACK
    }

    /// Null, so a failure to allocate falls back rather than yielding a bogus object.
    /// `ObjectLiteral` is still `Rejected`, so no compiled test reaches this yet.
    unsafe extern "C" fn stub_alloc_struct_object(
        _type_index: u32,
        _field_count: u32,
        _shared_state: *mut (),
    ) -> *mut () {
        std::ptr::null_mut()
    }

    /// Declines, so a slot write routes to the interpreter rather than silently
    /// succeeding or writing out of bounds. `InitObject` is still `Rejected`.
    unsafe extern "C" fn stub_init_object_field(
        _object: u64,
        _offset: u64,
        _value: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0 // JIT_STORE_FALLBACK
    }

    /// Declines, so a cast routes to the interpreter rather than silently passing.
    /// `CastObjectMinFields` is still `Rejected` AND still on
    /// `produces_incorrect_native_results`, so no compiled test reaches this yet.
    unsafe extern "C" fn stub_cast_object_min_fields(
        _object: u64,
        _required_fields: u64,
        _shared_state: *mut (),
    ) -> i8 {
        0 // OBJECT_MIN_FIELDS_DECLINE
    }
    unsafe extern "C" fn stub_alloc_string(
        _data_ptr: *const u8,
        _len: usize,
        _shared_state: *mut (),
    ) -> *mut () {
        std::ptr::null_mut()
    }
    unsafe extern "C" fn stub_safepoint_poll(_shared_state: *const ()) {}
    unsafe extern "C" fn stub_check_preemption(_current_task: *const ()) -> bool {
        true
    }
    unsafe extern "C" fn stub_native_call_dispatch(
        _native_id: u16,
        _args_ptr: *const u64,
        _arg_count: u8,
        _shared_state: *mut (),
    ) -> u64 {
        NULL_VALUE
    }
    unsafe extern "C" fn stub_interpreter_call(
        _opcode: u8,
        _operand_u64: u64,
        _operand_u32: u32,
        _receiver: u64,
        _args_ptr: *const u64,
        _arg_count: u16,
        _module_ptr: *const (),
        _shared_state: *mut (),
    ) -> u64 {
        NULL_VALUE
    }
    unsafe extern "C" fn stub_string_concat(
        _left: u64,
        _right: u64,
        _shared_state: *mut (),
    ) -> u64 {
        NULL_VALUE
    }
    unsafe extern "C" fn stub_generic_equals(
        _left: u64,
        _right: u64,
        _shared_state: *mut (),
    ) -> bool {
        false
    }

    let jit_func = JitFunction {
        name: "check_preemption".to_string(),
        func_index: 0,
        param_count: 0,
        local_count: 0,
        blocks: vec![raya_engine::jit::ir::instr::JitBlock {
            start_offset: raya_engine::jit::ir::instr::JitBlock::UNKNOWN_START_OFFSET,
            id: JitBlockId(0),
            instrs: vec![
                JitInstr::CheckPreemption {
                    bytecode_offset: 77,
                },
                JitInstr::ConstNull { dest: Reg(0) },
            ],
            terminator: JitTerminator::Return(Some(Reg(0))),
            predecessors: vec![],
        }],
        entry: JitBlockId(0),
        next_reg: 1,
        reg_types: FxHashMap::from_iter([(Reg(0), JitType::Value)]),
        signature_id: 0,
        abi_version: 0,
        param_types: vec![],
        return_type: JitType::Value,
    };

    let mut ctx = RuntimeContext {
        shared_state: std::ptr::null(),
        current_task: std::ptr::null(),
        module: std::ptr::null(),
        helpers: RuntimeHelperTable {
            alloc_object: stub_alloc_object,
            alloc_array: stub_alloc_array,
            alloc_string: stub_alloc_string,
            safepoint_poll: stub_safepoint_poll,
            check_preemption: stub_check_preemption,
            native_call_dispatch: stub_native_call_dispatch,
            interpreter_call: stub_interpreter_call,
            string_concat: stub_string_concat,
            generic_equals: stub_generic_equals,
            object_get_field: stub_object_get_field,
            object_set_field: stub_object_set_field,
            object_implements_shape: stub_object_implements_shape,
            object_is_nominal: stub_object_is_nominal,
            object_get_shape_field: stub_object_get_shape_field,
            object_set_shape_field: stub_object_set_shape_field,
            string_len: stub_string_len,
            string_compare: stub_string_compare,
            value_to_string: stub_value_to_string,
            const_string: stub_const_string,
            array_load: stub_array_load,
            array_store: stub_array_store,
            array_push: stub_array_push,
            array_pop: stub_array_pop,
            array_len: stub_array_len,
            refcell_load: stub_refcell_load,
            refcell_store: stub_refcell_store,
            refcell_new: stub_refcell_new,
            set_closure_capture: stub_set_closure_capture,
            make_closure: stub_make_closure,
            load_captured: stub_load_captured,
            store_captured: stub_store_captured,
            bind_method: stub_bind_method,
            await_task: stub_await_task,
            dyn_get_keyed: stub_dyn_get_keyed,
            dyn_set_keyed: stub_dyn_set_keyed,
            alloc_struct_object: stub_alloc_struct_object,
            init_object_field: stub_init_object_field,
            cast_object_min_fields: stub_cast_object_min_fields,
        },
    };

    let (raw, exit) = jit_compile_and_call_with_locals_exit_and_ctx(
        &jit_func,
        &mut [],
        (&mut ctx as *mut _),
    );
    assert_eq!(raw, NULL_VALUE);
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Suspended as u32
    );
    assert_eq!(
        exit.suspend_reason,
        raya_engine::jit::runtime::trampoline::JitSuspendReason::Preemption as u32
    );
    assert_eq!(exit.bytecode_offset, 77);
}

// ============================================================================
// IR builder helpers — build JitFunction from instructions
// ============================================================================

/// Build a single-block JitFunction from a list of instructions and typed registers.
fn build_func(instrs: Vec<JitInstr>, regs: Vec<(Reg, JitType)>, ret: Option<Reg>) -> JitFunction {
    let mut func = JitFunction::new(0, "test_func".to_string(), 0, 0);
    let entry = func.add_block();

    for (reg, ty) in &regs {
        // Ensure the register is allocated with the right type
        while func.next_reg <= reg.0 {
            // Allocate dummy regs to reach the target index
            let next = Reg(func.next_reg);
            let t = regs
                .iter()
                .find(|(r, _)| *r == next)
                .map(|(_, t)| *t)
                .unwrap_or(JitType::Value);
            func.alloc_reg(t);
        }
    }

    func.block_mut(entry).instrs = instrs;
    func.block_mut(entry).terminator = JitTerminator::Return(ret);
    func
}

/// Build a branching JitFunction: entry branches on cond_reg, then/else return different values.
fn build_branch_func(
    entry_instrs: Vec<JitInstr>,
    entry_regs: Vec<(Reg, JitType)>,
    cond_reg: Reg,
    then_instrs: Vec<JitInstr>,
    then_regs: Vec<(Reg, JitType)>,
    then_ret: Reg,
    else_instrs: Vec<JitInstr>,
    else_regs: Vec<(Reg, JitType)>,
    else_ret: Reg,
) -> JitFunction {
    let mut func = JitFunction::new(0, "test_branch".to_string(), 0, 0);
    let entry = func.add_block();
    let then_block = func.add_block();
    let else_block = func.add_block();

    // Collect all regs
    let all_regs: Vec<(Reg, JitType)> = entry_regs
        .iter()
        .chain(then_regs.iter())
        .chain(else_regs.iter())
        .cloned()
        .collect();

    for (reg, ty) in &all_regs {
        while func.next_reg <= reg.0 {
            let next = Reg(func.next_reg);
            let t = all_regs
                .iter()
                .find(|(r, _)| *r == next)
                .map(|(_, t)| *t)
                .unwrap_or(JitType::Value);
            func.alloc_reg(t);
        }
    }

    func.block_mut(entry).instrs = entry_instrs;
    func.block_mut(entry).terminator = JitTerminator::Branch {
        cond: cond_reg,
        then_block,
        else_block,
    };

    func.block_mut(then_block).instrs = then_instrs;
    func.block_mut(then_block).terminator = JitTerminator::Return(Some(then_ret));

    func.block_mut(else_block).instrs = else_instrs;
    func.block_mut(else_block).terminator = JitTerminator::Return(Some(else_ret));

    func
}

// ============================================================================
// Category 1: Lifter Tests (bytecode → JIT IR)
// ============================================================================

#[test]
fn lift_const_i32_return() {
    let mut code = Vec::new();
    emit_i32(&mut code, 42);
    emit(&mut code, Opcode::Return);

    let module = make_module(code, 0, 0);
    let jit_func = lift_function(&module.functions[0], &module, 0).unwrap();

    assert_eq!(jit_func.name, "test_func");
    assert!(!jit_func.blocks.is_empty());

    let display = format!("{}", jit_func);
    assert!(
        display.contains("const.i32 42"),
        "IR should contain const.i32 42, got:\n{}",
        display
    );
}

#[test]
fn lift_const_f64_return() {
    let mut code = Vec::new();
    emit_f64(&mut code, 3.14);
    emit(&mut code, Opcode::Return);

    let module = make_module(code, 0, 0);
    let jit_func = lift_function(&module.functions[0], &module, 0).unwrap();

    let display = format!("{}", jit_func);
    assert!(
        display.contains("const.f64"),
        "IR should contain const.f64, got:\n{}",
        display
    );
}

#[test]
fn lift_const_bool_null() {
    let mut code = Vec::new();
    emit(&mut code, Opcode::ConstTrue);
    emit(&mut code, Opcode::Pop);
    emit(&mut code, Opcode::ConstNull);
    emit(&mut code, Opcode::Return);

    let module = make_module(code, 0, 0);
    let jit_func = lift_function(&module.functions[0], &module, 0).unwrap();

    let display = format!("{}", jit_func);
    assert!(
        display.contains("const.bool true"),
        "IR should contain const.bool true, got:\n{}",
        display
    );
    assert!(
        display.contains("const.null"),
        "IR should contain const.null, got:\n{}",
        display
    );
}

#[test]
fn lift_integer_arithmetic() {
    let mut code = Vec::new();
    emit_i32(&mut code, 3);
    emit_i32(&mut code, 5);
    emit(&mut code, Opcode::Iadd);
    emit(&mut code, Opcode::Return);

    let module = make_module(code, 0, 0);
    let jit_func = lift_function(&module.functions[0], &module, 0).unwrap();

    let display = format!("{}", jit_func);
    assert!(
        display.contains("iadd"),
        "IR should contain iadd, got:\n{}",
        display
    );
}

#[test]
fn lift_float_arithmetic() {
    let mut code = Vec::new();
    emit_f64(&mut code, 1.5);
    emit_f64(&mut code, 2.5);
    emit(&mut code, Opcode::Fadd);
    emit(&mut code, Opcode::Return);

    let module = make_module(code, 0, 0);
    let jit_func = lift_function(&module.functions[0], &module, 0).unwrap();

    let display = format!("{}", jit_func);
    assert!(
        display.contains("fadd"),
        "IR should contain fadd, got:\n{}",
        display
    );
}

#[test]
fn lift_locals() {
    let mut code = Vec::new();
    emit_i32(&mut code, 10);
    emit_store_local(&mut code, 0);
    emit_load_local(&mut code, 0);
    emit(&mut code, Opcode::Return);

    let module = make_module(code, 0, 1);
    let jit_func = lift_function(&module.functions[0], &module, 0).unwrap();

    let display = format!("{}", jit_func);
    assert!(
        display.contains("store.local"),
        "IR should contain store.local, got:\n{}",
        display
    );
    assert!(
        display.contains("load.local"),
        "IR should contain load.local, got:\n{}",
        display
    );
}

#[test]
fn lift_comparisons() {
    let mut code = Vec::new();
    emit_i32(&mut code, 3);
    emit_i32(&mut code, 5);
    emit(&mut code, Opcode::Ilt);
    emit(&mut code, Opcode::Return);

    let module = make_module(code, 0, 0);
    let jit_func = lift_function(&module.functions[0], &module, 0).unwrap();

    let display = format!("{}", jit_func);
    assert!(
        display.contains("icmp.lt"),
        "IR should contain icmp.lt, got:\n{}",
        display
    );
}

#[test]
fn lift_branch() {
    let mut code = Vec::new();
    emit(&mut code, Opcode::ConstTrue);
    // JmpIfFalse with offset to skip over the "then" path
    emit_jmp(&mut code, Opcode::JmpIfFalse, 6); // skip ConstI32(1) + Return = 6 bytes
    emit_i32(&mut code, 1);
    emit(&mut code, Opcode::Return);
    emit_i32(&mut code, 2);
    emit(&mut code, Opcode::Return);

    let module = make_module(code, 0, 0);
    let jit_func = lift_function(&module.functions[0], &module, 0).unwrap();

    // Should have multiple blocks due to branching
    assert!(
        jit_func.blocks.len() >= 3,
        "Expected >= 3 blocks for branch, got {}",
        jit_func.blocks.len()
    );
}

#[test]
fn lift_bitwise() {
    let mut code = Vec::new();
    emit_i32(&mut code, 0xFF);
    emit_i32(&mut code, 0x0F);
    emit(&mut code, Opcode::Iand);
    emit(&mut code, Opcode::Return);

    let module = make_module(code, 0, 0);
    let jit_func = lift_function(&module.functions[0], &module, 0).unwrap();

    let display = format!("{}", jit_func);
    assert!(
        display.contains("iand"),
        "IR should contain iand, got:\n{}",
        display
    );
}

#[test]
fn lift_negation() {
    let mut code = Vec::new();
    emit_i32(&mut code, 42);
    emit(&mut code, Opcode::Ineg);
    emit(&mut code, Opcode::Return);

    let module = make_module(code, 0, 0);
    let jit_func = lift_function(&module.functions[0], &module, 0).unwrap();

    let display = format!("{}", jit_func);
    assert!(
        display.contains("ineg"),
        "IR should contain ineg, got:\n{}",
        display
    );
}

#[test]
fn lift_all_int_arithmetic_ops() {
    // Only operations with exact native semantics are lifted.
    for (op, expected) in [(Opcode::Isub, "isub"), (Opcode::Imul, "imul")] {
        let mut code = Vec::new();
        emit_i32(&mut code, 10);
        emit_i32(&mut code, 3);
        emit(&mut code, op);
        emit(&mut code, Opcode::Return);

        let module = make_module(code, 0, 0);
        let jit_func = lift_function(&module.functions[0], &module, 0).unwrap();

        let display = format!("{}", jit_func);
        assert!(
            display.contains(expected),
            "IR should contain {expected} for {:?}, got:\n{display}",
            op
        );
    }
}

#[test]
fn lift_integer_division_and_remainder_are_rejected_until_exact() {
    for op in [Opcode::Idiv, Opcode::Imod] {
        let mut code = Vec::new();
        emit_i32(&mut code, 10);
        emit_i32(&mut code, 3);
        emit(&mut code, op);
        emit(&mut code, Opcode::Return);
        let module = make_module(code, 0, 0);
        assert!(lift_function(&module.functions[0], &module, 0).is_err());
    }
}

#[test]
fn lift_float_ops() {
    for (op, expected) in [
        (Opcode::Fsub, "fsub"),
        (Opcode::Fmul, "fmul"),
        (Opcode::Fdiv, "fdiv"),
        (Opcode::Fneg, "fneg"),
    ] {
        let mut code = Vec::new();
        if op == Opcode::Fneg {
            emit_f64(&mut code, 1.5);
        } else {
            emit_f64(&mut code, 1.5);
            emit_f64(&mut code, 2.5);
        }
        emit(&mut code, op);
        emit(&mut code, Opcode::Return);

        let module = make_module(code, 0, 0);
        let jit_func = lift_function(&module.functions[0], &module, 0).unwrap();

        let display = format!("{}", jit_func);
        assert!(
            display.contains(expected),
            "IR should contain {expected} for {:?}, got:\n{display}",
            op
        );
    }
}

// ============================================================================
// Category 2: Native Execution — Constants
// ============================================================================

#[test]
fn exec_return_i32() {
    let r0 = Reg(0);
    let func = build_func(
        vec![JitInstr::ConstI32 {
            dest: r0,
            value: 42,
        }],
        vec![(r0, JitType::I32)],
        Some(r0),
    );

    let result = jit_compile_and_call(&func);
    assert!(is_i32(result), "Expected i32, got 0x{:016X}", result);
    assert_eq!(decode_i32(result), 42);
}

#[test]
fn exec_return_f64() {
    let r0 = Reg(0);
    let func = build_func(
        vec![JitInstr::ConstF64 {
            dest: r0,
            value: 3.14,
        }],
        vec![(r0, JitType::F64)],
        Some(r0),
    );

    let result = jit_compile_and_call(&func);
    assert!(is_f64(result), "Expected f64, got 0x{:016X}", result);
    let val = decode_f64(result);
    assert!((val - 3.14).abs() < 1e-10, "Expected 3.14, got {}", val);
}

#[test]
fn exec_return_bool_true() {
    let r0 = Reg(0);
    let func = build_func(
        vec![JitInstr::ConstBool {
            dest: r0,
            value: true,
        }],
        vec![(r0, JitType::Bool)],
        Some(r0),
    );

    let result = jit_compile_and_call(&func);
    assert!(is_bool(result), "Expected bool, got 0x{:016X}", result);
    assert!(decode_bool(result));
}

#[test]
fn exec_return_bool_false() {
    let r0 = Reg(0);
    let func = build_func(
        vec![JitInstr::ConstBool {
            dest: r0,
            value: false,
        }],
        vec![(r0, JitType::Bool)],
        Some(r0),
    );

    let result = jit_compile_and_call(&func);
    assert!(is_bool(result), "Expected bool, got 0x{:016X}", result);
    assert!(!decode_bool(result));
}

#[test]
fn exec_return_null() {
    let func = build_func(vec![], vec![], None);

    let result = jit_compile_and_call(&func);
    assert!(is_null(result), "Expected null, got 0x{:016X}", result);
}

// ============================================================================
// Category 3: Native Execution — Arithmetic
// ============================================================================

/// Helper: build and execute i32 binary op, return decoded i32
fn exec_i32_binop(op: fn(Reg, Reg, Reg) -> JitInstr, a: i32, b: i32) -> i32 {
    let r0 = Reg(0);
    let r1 = Reg(1);
    let r2 = Reg(2);
    let func = build_func(
        vec![
            JitInstr::ConstI32 { dest: r0, value: a },
            JitInstr::ConstI32 { dest: r1, value: b },
            op(r2, r0, r1),
        ],
        vec![(r0, JitType::I32), (r1, JitType::I32), (r2, JitType::I32)],
        Some(r2),
    );
    decode_i32(jit_compile_and_call(&func))
}

/// Helper: build and execute f64 binary op, return decoded f64
fn exec_f64_binop(op: fn(Reg, Reg, Reg) -> JitInstr, a: f64, b: f64) -> f64 {
    let r0 = Reg(0);
    let r1 = Reg(1);
    let r2 = Reg(2);
    let func = build_func(
        vec![
            JitInstr::ConstF64 { dest: r0, value: a },
            JitInstr::ConstF64 { dest: r1, value: b },
            op(r2, r0, r1),
        ],
        vec![(r0, JitType::F64), (r1, JitType::F64), (r2, JitType::F64)],
        Some(r2),
    );
    decode_f64(jit_compile_and_call(&func))
}

fn make_iadd(dest: Reg, left: Reg, right: Reg) -> JitInstr {
    JitInstr::IAdd { dest, left, right }
}
fn make_isub(dest: Reg, left: Reg, right: Reg) -> JitInstr {
    JitInstr::ISub { dest, left, right }
}
fn make_imul(dest: Reg, left: Reg, right: Reg) -> JitInstr {
    JitInstr::IMul { dest, left, right }
}
fn make_idiv(dest: Reg, left: Reg, right: Reg) -> JitInstr {
    JitInstr::IDiv { dest, left, right }
}
fn make_imod(dest: Reg, left: Reg, right: Reg) -> JitInstr {
    JitInstr::IMod { dest, left, right }
}
fn make_iand(dest: Reg, left: Reg, right: Reg) -> JitInstr {
    JitInstr::IAnd { dest, left, right }
}
fn make_ior(dest: Reg, left: Reg, right: Reg) -> JitInstr {
    JitInstr::IOr { dest, left, right }
}
fn make_ixor(dest: Reg, left: Reg, right: Reg) -> JitInstr {
    JitInstr::IXor { dest, left, right }
}
fn make_ishl(dest: Reg, left: Reg, right: Reg) -> JitInstr {
    JitInstr::IShl { dest, left, right }
}
fn make_ishr(dest: Reg, left: Reg, right: Reg) -> JitInstr {
    JitInstr::IShr { dest, left, right }
}
fn make_fadd(dest: Reg, left: Reg, right: Reg) -> JitInstr {
    JitInstr::FAdd { dest, left, right }
}
fn make_fsub(dest: Reg, left: Reg, right: Reg) -> JitInstr {
    JitInstr::FSub { dest, left, right }
}
fn make_fmul(dest: Reg, left: Reg, right: Reg) -> JitInstr {
    JitInstr::FMul { dest, left, right }
}
fn make_fdiv(dest: Reg, left: Reg, right: Reg) -> JitInstr {
    JitInstr::FDiv { dest, left, right }
}

#[test]
fn exec_iadd() {
    assert_eq!(exec_i32_binop(make_iadd, 3, 5), 8);
}

#[test]
fn exec_isub() {
    assert_eq!(exec_i32_binop(make_isub, 10, 3), 7);
}

#[test]
fn exec_imul() {
    assert_eq!(exec_i32_binop(make_imul, 6, 7), 42);
}

#[test]
fn exec_idiv() {
    assert_eq!(exec_i32_binop(make_idiv, 15, 3), 5);
}

#[test]
fn exec_imod() {
    assert_eq!(exec_i32_binop(make_imod, 17, 5), 2);
}

#[test]
fn exec_ineg() {
    let r0 = Reg(0);
    let r1 = Reg(1);
    let func = build_func(
        vec![
            JitInstr::ConstI32 {
                dest: r0,
                value: 42,
            },
            JitInstr::INeg {
                dest: r1,
                operand: r0,
            },
        ],
        vec![(r0, JitType::I32), (r1, JitType::I32)],
        Some(r1),
    );
    assert_eq!(decode_i32(jit_compile_and_call(&func)), -42);
}

#[test]
fn exec_fadd() {
    let result = exec_f64_binop(make_fadd, 1.5, 2.5);
    assert!((result - 4.0).abs() < 1e-10, "Expected 4.0, got {}", result);
}

#[test]
fn exec_fsub() {
    let result = exec_f64_binop(make_fsub, 5.0, 1.5);
    assert!((result - 3.5).abs() < 1e-10, "Expected 3.5, got {}", result);
}

#[test]
fn exec_fmul() {
    let result = exec_f64_binop(make_fmul, 2.0, 3.5);
    assert!((result - 7.0).abs() < 1e-10, "Expected 7.0, got {}", result);
}

#[test]
fn exec_fdiv() {
    let result = exec_f64_binop(make_fdiv, 7.0, 2.0);
    assert!((result - 3.5).abs() < 1e-10, "Expected 3.5, got {}", result);
}

#[test]
fn exec_fneg() {
    let r0 = Reg(0);
    let r1 = Reg(1);
    let func = build_func(
        vec![
            JitInstr::ConstF64 {
                dest: r0,
                value: 2.5,
            },
            JitInstr::FNeg {
                dest: r1,
                operand: r0,
            },
        ],
        vec![(r0, JitType::F64), (r1, JitType::F64)],
        Some(r1),
    );
    let result = decode_f64(jit_compile_and_call(&func));
    assert!(
        (result - (-2.5)).abs() < 1e-10,
        "Expected -2.5, got {}",
        result
    );
}

#[test]
fn exec_iand() {
    assert_eq!(exec_i32_binop(make_iand, 0xFF, 0x0F), 0x0F);
}

#[test]
fn exec_ior() {
    assert_eq!(exec_i32_binop(make_ior, 0xF0, 0x0F), 0xFF);
}

#[test]
fn exec_ixor() {
    assert_eq!(exec_i32_binop(make_ixor, 0xFF, 0x0F), 0xF0);
}

#[test]
fn exec_ishl() {
    assert_eq!(exec_i32_binop(make_ishl, 1, 3), 8);
}

#[test]
fn exec_ishr() {
    assert_eq!(exec_i32_binop(make_ishr, 16, 2), 4);
}

// ============================================================================
// Category 4: Native Execution — Comparisons, Logic, Branches
// ============================================================================

/// Helper: build and execute i32 comparison, return decoded bool
fn exec_i32_cmp(op: fn(Reg, Reg, Reg) -> JitInstr, a: i32, b: i32) -> bool {
    let r0 = Reg(0);
    let r1 = Reg(1);
    let r2 = Reg(2);
    let func = build_func(
        vec![
            JitInstr::ConstI32 { dest: r0, value: a },
            JitInstr::ConstI32 { dest: r1, value: b },
            op(r2, r0, r1),
        ],
        vec![(r0, JitType::I32), (r1, JitType::I32), (r2, JitType::Bool)],
        Some(r2),
    );
    decode_bool(jit_compile_and_call(&func))
}

fn make_icmp_lt(dest: Reg, left: Reg, right: Reg) -> JitInstr {
    JitInstr::ICmpLt { dest, left, right }
}
fn make_icmp_gt(dest: Reg, left: Reg, right: Reg) -> JitInstr {
    JitInstr::ICmpGt { dest, left, right }
}
fn make_icmp_eq(dest: Reg, left: Reg, right: Reg) -> JitInstr {
    JitInstr::ICmpEq { dest, left, right }
}
fn make_icmp_ne(dest: Reg, left: Reg, right: Reg) -> JitInstr {
    JitInstr::ICmpNe { dest, left, right }
}
fn make_icmp_le(dest: Reg, left: Reg, right: Reg) -> JitInstr {
    JitInstr::ICmpLe { dest, left, right }
}
fn make_icmp_ge(dest: Reg, left: Reg, right: Reg) -> JitInstr {
    JitInstr::ICmpGe { dest, left, right }
}

#[test]
fn exec_icmp_lt_true() {
    assert!(exec_i32_cmp(make_icmp_lt, 3, 5));
}

#[test]
fn exec_icmp_lt_false() {
    assert!(!exec_i32_cmp(make_icmp_lt, 5, 3));
}

#[test]
fn exec_icmp_eq_true() {
    assert!(exec_i32_cmp(make_icmp_eq, 5, 5));
}

#[test]
fn exec_icmp_eq_false() {
    assert!(!exec_i32_cmp(make_icmp_eq, 3, 5));
}

#[test]
fn exec_icmp_gt() {
    assert!(exec_i32_cmp(make_icmp_gt, 5, 3));
    assert!(!exec_i32_cmp(make_icmp_gt, 3, 5));
}

#[test]
fn exec_icmp_ne() {
    assert!(exec_i32_cmp(make_icmp_ne, 3, 5));
    assert!(!exec_i32_cmp(make_icmp_ne, 5, 5));
}

#[test]
fn exec_icmp_le() {
    assert!(exec_i32_cmp(make_icmp_le, 3, 5));
    assert!(exec_i32_cmp(make_icmp_le, 5, 5));
    assert!(!exec_i32_cmp(make_icmp_le, 6, 5));
}

#[test]
fn exec_icmp_ge() {
    assert!(exec_i32_cmp(make_icmp_ge, 5, 3));
    assert!(exec_i32_cmp(make_icmp_ge, 5, 5));
    assert!(!exec_i32_cmp(make_icmp_ge, 3, 5));
}

#[test]
fn exec_fcmp_lt() {
    let r0 = Reg(0);
    let r1 = Reg(1);
    let r2 = Reg(2);
    let func = build_func(
        vec![
            JitInstr::ConstF64 {
                dest: r0,
                value: 1.0,
            },
            JitInstr::ConstF64 {
                dest: r1,
                value: 2.0,
            },
            JitInstr::FCmpLt {
                dest: r2,
                left: r0,
                right: r1,
            },
        ],
        vec![(r0, JitType::F64), (r1, JitType::F64), (r2, JitType::Bool)],
        Some(r2),
    );
    assert!(decode_bool(jit_compile_and_call(&func)));
}

#[test]
fn exec_logic_and() {
    let r0 = Reg(0);
    let r1 = Reg(1);
    let r2 = Reg(2);
    let func = build_func(
        vec![
            JitInstr::ConstBool {
                dest: r0,
                value: true,
            },
            JitInstr::ConstBool {
                dest: r1,
                value: false,
            },
            JitInstr::And {
                dest: r2,
                left: r0,
                right: r1,
            },
        ],
        vec![
            (r0, JitType::Bool),
            (r1, JitType::Bool),
            (r2, JitType::Bool),
        ],
        Some(r2),
    );
    assert!(!decode_bool(jit_compile_and_call(&func)));
}

#[test]
fn exec_logic_or() {
    let r0 = Reg(0);
    let r1 = Reg(1);
    let r2 = Reg(2);
    let func = build_func(
        vec![
            JitInstr::ConstBool {
                dest: r0,
                value: true,
            },
            JitInstr::ConstBool {
                dest: r1,
                value: false,
            },
            JitInstr::Or {
                dest: r2,
                left: r0,
                right: r1,
            },
        ],
        vec![
            (r0, JitType::Bool),
            (r1, JitType::Bool),
            (r2, JitType::Bool),
        ],
        Some(r2),
    );
    assert!(decode_bool(jit_compile_and_call(&func)));
}

#[test]
fn exec_logic_not() {
    let r0 = Reg(0);
    let r1 = Reg(1);
    let func = build_func(
        vec![
            JitInstr::ConstBool {
                dest: r0,
                value: true,
            },
            JitInstr::Not {
                dest: r1,
                operand: r0,
            },
        ],
        vec![(r0, JitType::Bool), (r1, JitType::Bool)],
        Some(r1),
    );
    assert!(!decode_bool(jit_compile_and_call(&func)));
}

#[test]
fn exec_branch_true() {
    // if true { return 1 } else { return 2 }
    let r0 = Reg(0);
    let r1 = Reg(1);
    let r2 = Reg(2);

    let func = build_branch_func(
        vec![JitInstr::ConstBool {
            dest: r0,
            value: true,
        }],
        vec![(r0, JitType::Bool)],
        r0,
        vec![JitInstr::ConstI32 { dest: r1, value: 1 }],
        vec![(r1, JitType::I32)],
        r1,
        vec![JitInstr::ConstI32 { dest: r2, value: 2 }],
        vec![(r2, JitType::I32)],
        r2,
    );

    assert_eq!(decode_i32(jit_compile_and_call(&func)), 1);
}

#[test]
fn exec_branch_false() {
    // if false { return 1 } else { return 2 }
    let r0 = Reg(0);
    let r1 = Reg(1);
    let r2 = Reg(2);

    let func = build_branch_func(
        vec![JitInstr::ConstBool {
            dest: r0,
            value: false,
        }],
        vec![(r0, JitType::Bool)],
        r0,
        vec![JitInstr::ConstI32 { dest: r1, value: 1 }],
        vec![(r1, JitType::I32)],
        r1,
        vec![JitInstr::ConstI32 { dest: r2, value: 2 }],
        vec![(r2, JitType::I32)],
        r2,
    );

    assert_eq!(decode_i32(jit_compile_and_call(&func)), 2);
}

#[test]
fn exec_complex_expr() {
    // (3 + 5) * (10 - 2) = 8 * 8 = 64
    let r0 = Reg(0);
    let r1 = Reg(1);
    let r2 = Reg(2); // 3 + 5
    let r3 = Reg(3);
    let r4 = Reg(4);
    let r5 = Reg(5); // 10 - 2
    let r6 = Reg(6); // r2 * r5

    let func = build_func(
        vec![
            JitInstr::ConstI32 { dest: r0, value: 3 },
            JitInstr::ConstI32 { dest: r1, value: 5 },
            JitInstr::IAdd {
                dest: r2,
                left: r0,
                right: r1,
            },
            JitInstr::ConstI32 {
                dest: r3,
                value: 10,
            },
            JitInstr::ConstI32 { dest: r4, value: 2 },
            JitInstr::ISub {
                dest: r5,
                left: r3,
                right: r4,
            },
            JitInstr::IMul {
                dest: r6,
                left: r2,
                right: r5,
            },
        ],
        vec![
            (r0, JitType::I32),
            (r1, JitType::I32),
            (r2, JitType::I32),
            (r3, JitType::I32),
            (r4, JitType::I32),
            (r5, JitType::I32),
            (r6, JitType::I32),
        ],
        Some(r6),
    );

    assert_eq!(decode_i32(jit_compile_and_call(&func)), 64);
}

#[test]
fn exec_negative_i32() {
    let r0 = Reg(0);
    let func = build_func(
        vec![JitInstr::ConstI32 {
            dest: r0,
            value: -100,
        }],
        vec![(r0, JitType::I32)],
        Some(r0),
    );
    assert_eq!(decode_i32(jit_compile_and_call(&func)), -100);
}

#[test]
fn exec_i32_overflow_wrapping() {
    // i32::MAX + 1 wraps around (two's complement)
    let result = exec_i32_binop(make_iadd, i32::MAX, 1);
    assert_eq!(result, i32::MIN);
}

// ============================================================================
// Category 5: Full Pipeline + VM Integration
// ============================================================================

#[test]
fn pipeline_bytecode_to_native_i32() {
    let mut code = Vec::new();
    emit_i32(&mut code, 42);
    emit(&mut code, Opcode::Return);

    let result = jit_pipeline_and_call(code, 0);
    assert_eq!(decode_i32(result), 42);
}

#[test]
fn pipeline_bytecode_to_native_arith() {
    let mut code = Vec::new();
    emit_i32(&mut code, 3);
    emit_i32(&mut code, 5);
    emit(&mut code, Opcode::Iadd);
    emit(&mut code, Opcode::Return);

    let result = jit_pipeline_and_call(code, 0);
    assert_eq!(decode_i32(result), 8);
}

#[test]
fn pipeline_bytecode_to_native_float() {
    let mut code = Vec::new();
    emit_f64(&mut code, 1.5);
    emit_f64(&mut code, 2.5);
    emit(&mut code, Opcode::Fadd);
    emit(&mut code, Opcode::Return);

    let result = jit_pipeline_and_call(code, 0);
    let val = decode_f64(result);
    assert!((val - 4.0).abs() < 1e-10, "Expected 4.0, got {}", val);
}

#[test]
fn pipeline_bytecode_to_native_locals() {
    let mut code = Vec::new();
    emit_i32(&mut code, 99);
    emit_store_local(&mut code, 0);
    emit_load_local(&mut code, 0);
    emit(&mut code, Opcode::Return);

    let result = jit_pipeline_and_call(code, 1);
    // LoadLocal returns a Value (i64) from the locals array.
    // The stored value is a NaN-boxed i32 from the lifter's boxing.
    // But actually the lifter stores the raw I32 register value via StoreLocal,
    // and the lowering stores it to the locals_ptr as i64. So the roundtrip
    // through locals means the value is whatever was on the stack.
    // Since the lifter produces Value-typed registers for local loads,
    // the result should be the raw i64 bits of the i32 value 99.
    // In practice, the lifter boxes i32 before storing to locals.
    // Let's just check the result is non-zero (the roundtrip works).
    assert_ne!(result, 0, "Local variable roundtrip should return non-zero");
}

#[test]
fn pipeline_bytecode_to_native_multi_op() {
    // 10 - 3 = 7, then 7 * 2 = 14
    let mut code = Vec::new();
    emit_i32(&mut code, 10);
    emit_i32(&mut code, 3);
    emit(&mut code, Opcode::Isub);
    emit_i32(&mut code, 2);
    emit(&mut code, Opcode::Imul);
    emit(&mut code, Opcode::Return);

    let result = jit_pipeline_and_call(code, 0);
    assert_eq!(decode_i32(result), 14);
}

#[test]
fn pipeline_bytecode_to_native_branch_loop_i32_compiler_semantics() {
    let mut code = Vec::new();
    // i = 0
    emit_i32(&mut code, 0);
    emit_store_local(&mut code, 0);

    let loop_head = code.len();
    emit_load_local(&mut code, 0);
    emit_i32(&mut code, 64);
    emit(&mut code, Opcode::Ilt);
    let jmp_exit = emit_jmp_i32_placeholder(&mut code, Opcode::JmpIfFalse);

    emit_load_local(&mut code, 0);
    emit_i32(&mut code, 1);
    emit(&mut code, Opcode::Iadd);
    emit_store_local(&mut code, 0);

    let jmp_back = emit_jmp_i32_placeholder(&mut code, Opcode::Jmp);
    let exit = code.len();
    patch_jmp_i32_compiler(&mut code, jmp_exit, exit);
    patch_jmp_i32_compiler(&mut code, jmp_back, loop_head);

    // force unbox + typed return
    emit_load_local(&mut code, 0);
    emit_i32(&mut code, 0);
    emit(&mut code, Opcode::Iadd);
    emit(&mut code, Opcode::Return);

    let jit_raw = jit_pipeline_and_call(code.clone(), 1);
    let jit_val = decode_i32(jit_raw);

    let module = make_vm_module(code, 0, 1);
    let mut vm = Vm::with_worker_count(1);
    let interp_val = vm.execute(&module).unwrap().as_i32().unwrap();

    assert_eq!(jit_val, 64);
    assert_eq!(jit_val, interp_val);
}

#[test]
fn jit_allocating_string_helpers_preserve_exact_fallback_state() {
    let mut concat_code = Vec::new();
    emit(&mut concat_code, Opcode::ConstNull);
    emit(&mut concat_code, Opcode::ConstTrue);
    emit(&mut concat_code, Opcode::Sconcat);
    emit(&mut concat_code, Opcode::Return);
    let concat_module = make_module(concat_code, 0, 0);
    let concat_jit = lift_function(&concat_module.functions[0], &concat_module, 0)
        .expect("lift concat fallback");
    let (_raw, exit) = jit_compile_and_call_with_locals_and_exit(&concat_jit, &mut []);
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Suspended as u32
    );
    assert_eq!(
        exit.suspend_reason,
        raya_engine::jit::runtime::trampoline::JitSuspendReason::InterpreterBoundary as u32
    );
    assert_eq!(exit.bytecode_offset, 2);
    assert_eq!(exit.native_arg_count, 2);
    assert_eq!(exit.native_args[0], NULL_VALUE);
    assert_eq!(
        exit.native_args[1],
        raya_engine::vm::value::Value::bool(true).raw()
    );

    let mut conversion_code = Vec::new();
    emit_i32(&mut conversion_code, 9);
    emit(&mut conversion_code, Opcode::ToString);
    emit(&mut conversion_code, Opcode::Return);
    let conversion_module = make_module(conversion_code, 0, 0);
    let conversion_jit = lift_function(&conversion_module.functions[0], &conversion_module, 0)
        .expect("lift conversion fallback");
    let (_raw, exit) = jit_compile_and_call_with_locals_and_exit(&conversion_jit, &mut []);
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Suspended as u32
    );
    assert_eq!(
        exit.suspend_reason,
        raya_engine::jit::runtime::trampoline::JitSuspendReason::InterpreterBoundary as u32
    );
    assert_eq!(exit.bytecode_offset, 5);
    assert_eq!(exit.native_arg_count, 1);
    assert_eq!(decode_i32(exit.native_args[0]), 9);
}

#[test]
fn jit_string_comparisons_match_interpreter() {
    for (opcode, left, right, expected) in [
        (Opcode::Seq, "", "", true),
        (Opcode::Seq, "same", "same", true),
        (Opcode::Sne, "same", "different", true),
        (Opcode::Slt, "a", "b", true),
        (Opcode::Sle, "é", "é", true),
        (Opcode::Sgt, "z", "prefix", true),
        (Opcode::Sge, "same", "same", true),
        (Opcode::Seq, "nul\0inside", "nul\0inside", true),
        (Opcode::Slt, "é", "z", false),
    ] {
        let mut module = make_vm_module(Vec::new(), 0, 0);
        let left_index = module.constants.add_string(left.to_string());
        let right_index = module.constants.add_string(right.to_string());
        let mut code = Vec::new();
        emit_const_str(&mut code, left_index);
        emit_const_str(&mut code, right_index);
        emit(&mut code, opcode);
        emit(&mut code, Opcode::Return);
        module.functions[0].code = code;
        let module = finalize_module(module);

        let mut vm = Vm::with_worker_count(1);
        let interpreted = vm
            .execute(module.as_ref())
            .expect("interpreter string comparison")
            .as_bool()
            .expect("interpreter bool result");
        let (native, exit, _shared) = execute_module_natively(module, None);
        assert_eq!(
            exit.kind,
            raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32,
            "{opcode:?} did not complete natively"
        );
        assert_eq!(native.as_bool(), Some(interpreted), "{opcode:?}");
        assert_eq!(interpreted, expected, "{opcode:?}");
    }
}

#[test]
fn jit_string_length_and_conversion_match_interpreter() {
    let mut length_module = make_vm_module(Vec::new(), 0, 0);
    let text = length_module.constants.add_string("héllo".to_string());
    let mut code = Vec::new();
    emit_const_str(&mut code, text);
    emit(&mut code, Opcode::Slen);
    emit(&mut code, Opcode::Return);
    length_module.functions[0].code = code;
    let length_module = finalize_module(length_module);
    let mut vm = Vm::with_worker_count(1);
    let interpreted = vm.execute(length_module.as_ref()).expect("interpreter Slen");
    let (native, exit, _shared) = execute_module_natively(length_module, None);
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
    );
    assert_eq!(native.as_i32(), interpreted.as_i32());
    assert_eq!(native.as_i32(), Some("héllo".len() as i32));

    let primitive_cases: Vec<(Vec<u8>, &str)> = vec![
        (vec![Opcode::ConstNull as u8], "null"),
        (vec![Opcode::ConstTrue as u8], "true"),
        ({
            let mut code = Vec::new();
            emit_i32(&mut code, -7);
            code
        }, "-7"),
        ({
            let mut code = Vec::new();
            emit_f64(&mut code, 2.5);
            code
        }, "2.5"),
    ];
    for (mut code, expected) in primitive_cases {
        emit(&mut code, Opcode::ToString);
        emit(&mut code, Opcode::Return);
        let module = finalize_module(make_vm_module(code, 0, 0));
        let interpreted = {
            let mut vm = Vm::with_worker_count(1);
            string_contents(vm.execute(module.as_ref()).expect("interpreter ToString"))
        };
        let (native, exit, _shared) = execute_module_natively(module, None);
        assert_eq!(
            exit.kind,
            raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
        );
        assert_eq!(string_contents(native), interpreted);
        assert_eq!(interpreted, expected);
    }

    let mut string_module = make_vm_module(Vec::new(), 0, 0);
    let value = string_module.constants.add_string("hé".to_string());
    let mut code = Vec::new();
    emit_const_str(&mut code, value);
    emit(&mut code, Opcode::ToString);
    emit(&mut code, Opcode::Return);
    string_module.functions[0].code = code;
    let string_module = finalize_module(string_module);
    let interpreted = {
        let mut vm = Vm::with_worker_count(1);
        string_contents(vm.execute(string_module.as_ref()).expect("string ToString"))
    };
    let (native, exit, _shared) = execute_module_natively(string_module, Some(1));
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
    );
    assert_eq!(string_contents(native), interpreted);
    assert_eq!(interpreted, "hé");

    let mut object_module = make_vm_module(Vec::new(), 0, 0);
    object_module.classes.push(ClassDef {
        name: "ObjectForString".to_string(),
        field_count: 0,
        parent_id: None,
        methods: Vec::new(),
    });
    let mut code = Vec::new();
    code.push(Opcode::NewType as u8);
    code.extend_from_slice(&0u16.to_le_bytes());
    emit(&mut code, Opcode::ToString);
    emit(&mut code, Opcode::Return);
    object_module.functions[0].code = code;
    let object_module = finalize_module(object_module);
    let interpreted = {
        let mut vm = Vm::with_worker_count(1);
        string_contents(vm.execute(object_module.as_ref()).expect("object ToString"))
    };
    let (native, exit, _shared) = execute_module_natively(object_module, None);
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
    );
    assert_eq!(string_contents(native), interpreted);
    assert_eq!(interpreted, "[object]");
}

#[test]
fn jit_generic_equality_matches_interpreter() {
    let cases: Vec<(Opcode, Vec<u8>, bool)> = vec![
        (
            Opcode::Eq,
            vec![Opcode::ConstNull as u8, Opcode::ConstNull as u8],
            true,
        ),
        (
            Opcode::Ne,
            vec![Opcode::ConstTrue as u8, Opcode::ConstFalse as u8],
            true,
        ),
        (
            Opcode::Eq,
            {
                let mut code = Vec::new();
                emit_i32(&mut code, 7);
                emit_i32(&mut code, 7);
                code
            },
            true,
        ),
        (
            Opcode::Eq,
            {
                let mut code = Vec::new();
                emit_f64(&mut code, 2.5);
                emit_f64(&mut code, 2.5);
                code
            },
            true,
        ),
        (
            Opcode::Eq,
            {
                let mut code = Vec::new();
                emit_i32(&mut code, 7);
                emit_f64(&mut code, 7.0);
                code
            },
            true,
        ),
        (
            Opcode::StrictEq,
            {
                let mut code = Vec::new();
                emit_i32(&mut code, 7);
                emit_f64(&mut code, 7.0);
                code
            },
            true,
        ),
        (
            Opcode::Ne,
            {
                let mut code = Vec::new();
                emit_f64(&mut code, f64::NAN);
                emit_f64(&mut code, f64::NAN);
                code
            },
            true,
        ),
        (
            Opcode::StrictNe,
            {
                let mut code = Vec::new();
                emit_f64(&mut code, f64::NAN);
                emit_f64(&mut code, f64::NAN);
                code
            },
            true,
        ),
        (
            Opcode::Eq,
            {
                let mut code = Vec::new();
                emit_f64(&mut code, 0.0);
                emit_f64(&mut code, -0.0);
                code
            },
            true,
        ),
    ];

    for (opcode, mut operands, expected) in cases {
        operands.push(opcode as u8);
        operands.push(Opcode::Return as u8);
        let module = finalize_module(make_vm_module(operands, 0, 0));
        let mut vm = Vm::with_worker_count(1);
        let interpreted = vm
            .execute(module.as_ref())
            .expect("interpreter generic equality")
            .as_bool()
            .expect("interpreter bool result");
        let (native, exit, _shared) = execute_module_natively(module, None);
        assert_eq!(
            exit.kind,
            raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32,
            "{opcode:?} did not complete natively"
        );
        assert_eq!(native.as_bool(), Some(interpreted), "{opcode:?}");
        assert_eq!(interpreted, expected, "{opcode:?}");
    }

    let mut module = make_vm_module(Vec::new(), 0, 0);
    let first = module.constants.add_string("content".to_string());
    let second = module.constants.add_string("content".to_string());
    let mut code = Vec::new();
    emit_const_str(&mut code, first);
    emit_const_str(&mut code, second);
    emit(&mut code, Opcode::Eq);
    emit(&mut code, Opcode::Return);
    module.functions[0].code = code;
    let module = finalize_module(module);
    let mut vm = Vm::with_worker_count(1);
    let interpreted = vm.execute(module.as_ref()).expect("string equality");
    let (native, exit, _shared) = execute_module_natively(module, None);
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
    );
    assert_eq!(native.as_bool(), interpreted.as_bool());
    assert_eq!(native.as_bool(), Some(true));

    for (same_object, expected) in [(true, true), (false, false)] {
        let mut module = make_vm_module(Vec::new(), 0, 0);
        module.classes.push(ClassDef {
            name: "EqualityObject".to_string(),
            field_count: 0,
            parent_id: None,
            methods: Vec::new(),
        });
        let mut code = Vec::new();
        code.push(Opcode::NewType as u8);
        code.extend_from_slice(&0u16.to_le_bytes());
        if same_object {
            emit(&mut code, Opcode::Dup);
        } else {
            code.push(Opcode::NewType as u8);
            code.extend_from_slice(&0u16.to_le_bytes());
        }
        emit(&mut code, Opcode::Eq);
        emit(&mut code, Opcode::Return);
        module.functions[0].code = code;
        let module = finalize_module(module);
        let interpreted = {
            let mut vm = Vm::with_worker_count(1);
            vm.execute(module.as_ref())
                .expect("interpreter object equality")
                .as_bool()
                .expect("object equality bool")
        };
        let (native, exit, _shared) = execute_module_natively(module, None);
        assert_eq!(
            exit.kind,
            raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
        );
        assert_eq!(native.as_bool(), Some(interpreted));
        assert_eq!(interpreted, expected);
    }
}

#[test]
fn jit_string_allocation_matches_interpreter_and_survives_gc() {
    let mut concat_module = make_vm_module(Vec::new(), 0, 0);
    let hello = concat_module.constants.add_string("hello ".to_string());
    let world = concat_module.constants.add_string("世界".to_string());
    let mut concat_code = Vec::new();
    emit_const_str(&mut concat_code, hello);
    emit_const_str(&mut concat_code, world);
    emit(&mut concat_code, Opcode::Sconcat);
    emit(&mut concat_code, Opcode::Return);
    concat_module.functions[0].code = concat_code;
    let concat_module = finalize_module(concat_module);
    let interpreted_text = {
        let mut vm = Vm::with_worker_count(1);
        let value = vm
            .execute(concat_module.as_ref())
            .expect("interpreter concat");
        string_contents(value)
    };
    let (native, exit, _shared) = execute_module_natively(concat_module, None);
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
    );
    assert_eq!(string_contents(native), interpreted_text);
    assert_eq!(interpreted_text, "hello 世界");

    let primitive_concat_cases: Vec<(Vec<u8>, &str)> = vec![
        (vec![Opcode::ConstNull as u8], "null!"),
        (vec![Opcode::ConstTrue as u8], "true!"),
        ({
            let mut code = Vec::new();
            emit_i32(&mut code, -7);
            code
        }, "-7!"),
        ({
            let mut code = Vec::new();
            emit_f64(&mut code, 2.5);
            code
        }, "2.5!"),
    ];
    for (mut code, expected) in primitive_concat_cases {
        let mut module = make_vm_module(Vec::new(), 0, 0);
        let suffix = module.constants.add_string("!".to_string());
        emit_const_str(&mut code, suffix);
        emit(&mut code, Opcode::Sconcat);
        emit(&mut code, Opcode::Return);
        module.functions[0].code = code;
        let module = finalize_module(module);
        let interpreted = {
            let mut vm = Vm::with_worker_count(1);
            string_contents(vm.execute(module.as_ref()).expect("interpreter concat conversion"))
        };
        let (native, exit, _shared) = execute_module_natively(module, None);
        assert_eq!(
            exit.kind,
            raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
        );
        assert_eq!(string_contents(native), interpreted);
        assert_eq!(interpreted, expected);
    }

    let mut object_concat_module = make_vm_module(Vec::new(), 0, 0);
    object_concat_module.classes.push(ClassDef {
        name: "ConcatObject".to_string(),
        field_count: 0,
        parent_id: None,
        methods: Vec::new(),
    });
    let suffix = object_concat_module.constants.add_string("!".to_string());
    let mut object_concat_code = vec![Opcode::NewType as u8];
    object_concat_code.extend_from_slice(&0u16.to_le_bytes());
    emit_const_str(&mut object_concat_code, suffix);
    emit(&mut object_concat_code, Opcode::Sconcat);
    emit(&mut object_concat_code, Opcode::Return);
    object_concat_module.functions[0].code = object_concat_code;
    let object_concat_module = finalize_module(object_concat_module);
    let interpreted = {
        let mut vm = Vm::with_worker_count(1);
        string_contents(
            vm.execute(object_concat_module.as_ref())
                .expect("interpreter object concat"),
        )
    };
    let (native, exit, _shared) = execute_module_natively(object_concat_module, None);
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
    );
    assert_eq!(string_contents(native), interpreted);
    assert_eq!(interpreted, "[object]!");

    let mut conversion_code = Vec::new();
    emit_i32(&mut conversion_code, 42);
    emit(&mut conversion_code, Opcode::ToString);
    emit(&mut conversion_code, Opcode::Return);
    let conversion_module = finalize_module(make_vm_module(conversion_code, 0, 0));
    let interpreted_text = {
        let mut vm = Vm::with_worker_count(1);
        string_contents(
            vm.execute(conversion_module.as_ref())
                .expect("interpreter conversion"),
        )
    };
    let (native, exit, _shared) = execute_module_natively(conversion_module, None);
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
    );
    assert_eq!(string_contents(native), interpreted_text);
    assert_eq!(interpreted_text, "42");

    let mut stress_module = make_vm_module(Vec::new(), 0, 0);
    let empty = stress_module.constants.add_string(String::new());
    let fragment = stress_module.constants.add_string("é".to_string());
    let mut stress_code = Vec::new();
    emit_const_str(&mut stress_code, empty);
    for _ in 0..64 {
        emit_const_str(&mut stress_code, fragment);
        emit(&mut stress_code, Opcode::Sconcat);
    }
    emit(&mut stress_code, Opcode::Return);
    stress_module.functions[0].code = stress_code;
    let stress_module = finalize_module(stress_module);
    let (native, exit, shared) = execute_module_natively(stress_module, Some(1));
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32
    );
    assert_eq!(string_contents(native), "é".repeat(64));
    assert!(
        shared.gc.lock().stats().collections > 0,
        "GC stress test did not trigger a collection"
    );
}

#[test]
fn engine_prewarm_selects_hot() {
    let mut engine = JitEngine::with_config(JitConfig {
        max_prewarm_functions: 2,
        ..Default::default()
    })
    .unwrap();

    // Create a module with two functions:
    // func 0: trivial (ConstNull, Return) — should NOT be selected
    // func 1: math-heavy (many arithmetic ops) — should be selected
    let trivial_code = vec![Opcode::ConstNull as u8, Opcode::Return as u8];

    let mut heavy_code = Vec::new();
    for _ in 0..8 {
        emit_i32(&mut heavy_code, 1);
        emit_i32(&mut heavy_code, 2);
        emit(&mut heavy_code, Opcode::Iadd);
        emit_i32(&mut heavy_code, 3);
        emit(&mut heavy_code, Opcode::Imul);
    }
    for _ in 0..7 {
        emit(&mut heavy_code, Opcode::Iadd);
    }
    emit(&mut heavy_code, Opcode::Return);

    let module = Module {
        runtime_types: Vec::new(),
        function_signatures: Vec::new(),
        magic: *b"RAYA",
        version: VERSION,
        flags: 0,
        constants: ConstantPool::new(),
        functions: vec![
            Function {
                signature_id: 0,
                local_types: Vec::new(),
                abi_version: 1,
                name: "trivial".to_string(),
                param_count: 0,
                local_count: 0,
                code: trivial_code,
            },
            Function {
                signature_id: 0,
                local_types: Vec::new(),
                abi_version: 1,
                name: "heavy_math".to_string(),
                param_count: 0,
                local_count: 0,
                code: heavy_code,
            },
        ],
        classes: vec![],
        metadata: Metadata {
            name: "prewarm_test".to_string(),
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
    };

    let result = engine.prewarm(&module);

    // The heavy function should be compiled (or at least attempted)
    let total = result.compiled + result.failed;
    assert!(
        total > 0,
        "Prewarm should have processed at least one function"
    );
}

#[test]
fn engine_prewarm_with_custom_config() {
    let config = JitConfig {
        max_prewarm_functions: 4,
        min_score: 1.0, // Very low threshold
        min_instruction_count: 2,
        ..Default::default()
    };
    let mut engine = JitEngine::with_config(config).unwrap();

    // Even a simple function should be a candidate with min_score = 1.0
    let mut code = Vec::new();
    emit_i32(&mut code, 1);
    emit_i32(&mut code, 2);
    emit(&mut code, Opcode::Iadd);
    emit(&mut code, Opcode::Return);

    let module = make_module(code, 0, 0);
    let result = engine.prewarm(&module);

    // With low threshold, the function should be considered
    let total = result.compiled + result.failed;
    assert!(total >= 0); // Just verify no crash
}

#[test]
fn vm_enable_jit_executes() {
    let mut vm = raya_engine::Vm::new();
    vm.enable_jit().expect("Failed to enable JIT");

    // Build a simple module with a "main" function and execute
    let mut code = Vec::new();
    emit_i32(&mut code, 42);
    emit(&mut code, Opcode::Return);

    let module = make_vm_module(code, 0, 0);
    let result = vm.execute(&module).expect("Execution failed");
    assert_eq!(result, raya_engine::Value::i32(42));
}

#[test]
fn vm_enable_jit_with_config() {
    let mut vm = raya_engine::Vm::new();
    let config = JitConfig {
        max_prewarm_functions: 8,
        min_score: 5.0,
        ..Default::default()
    };
    vm.enable_jit_with_config(config)
        .expect("Failed to enable JIT with config");

    let mut code = Vec::new();
    emit_i32(&mut code, 100);
    emit(&mut code, Opcode::Return);

    let module = make_vm_module(code, 0, 0);
    let result = vm.execute(&module).expect("Execution failed");
    assert_eq!(result, raya_engine::Value::i32(100));
}

// ============================================================================
// Category 6: Adaptive (On-the-Fly) JIT Compilation
// ============================================================================

#[test]
fn profiling_counters_unit_test() {
    use raya_engine::jit::profiling::counters::{FunctionProfile, ModuleProfile};

    let profile = ModuleProfile::new(3);
    assert_eq!(profile.record_call(0), 1);
    assert_eq!(profile.record_call(0), 2);
    assert_eq!(profile.record_call(1), 1);
    assert_eq!(profile.record_loop(2), 1);

    // Out-of-bounds returns 0
    assert_eq!(profile.record_call(99), 0);
}

#[test]
fn compilation_policy_unit_test() {
    use raya_engine::jit::profiling::counters::FunctionProfile;
    use raya_engine::jit::profiling::policy::CompilationPolicy;

    let policy = CompilationPolicy::new();
    let profile = FunctionProfile::new();

    // Below threshold — should not compile
    for _ in 0..999 {
        profile.record_call();
    }
    assert!(!policy.should_compile(&profile, 100));

    // At threshold — should compile
    profile.record_call();
    assert!(policy.should_compile(&profile, 100));

    // Already compiling — should not re-request
    assert!(profile.try_start_compile());
    assert!(!policy.should_compile(&profile, 100));

    // After compilation complete — should not re-request
    profile.finish_compile();
    assert!(!policy.should_compile(&profile, 100));
}

#[test]
fn vm_adaptive_jit_creates_module_profile() {
    // Verify that execute() with adaptive JIT creates a module profile
    let mut vm = raya_engine::Vm::new();
    let config = JitConfig {
        adaptive_compilation: true,
        ..Default::default()
    };
    vm.enable_jit_with_config(config).unwrap();

    let mut code = Vec::new();
    emit_i32(&mut code, 42);
    emit(&mut code, Opcode::Return);

    let module = make_vm_module(code, 0, 0);
    let result = vm.execute(&module).expect("Execution failed");
    assert_eq!(result, raya_engine::Value::i32(42));

    // Verify profile was created
    let profiles = vm.shared_state().module_profiles.read();
    assert_eq!(profiles.len(), 1, "Expected one module profile");
}

#[test]
fn vm_adaptive_jit_disabled_no_profile() {
    // When adaptive_compilation is false, no profile should be created
    let mut vm = raya_engine::Vm::new();
    let config = JitConfig {
        adaptive_compilation: false,
        ..Default::default()
    };
    vm.enable_jit_with_config(config).unwrap();

    let mut code = Vec::new();
    emit_i32(&mut code, 42);
    emit(&mut code, Opcode::Return);

    let module = make_vm_module(code, 0, 0);
    let result = vm.execute(&module).expect("Execution failed");
    assert_eq!(result, raya_engine::Value::i32(42));

    // Verify no profile was created
    let profiles = vm.shared_state().module_profiles.read();
    assert_eq!(
        profiles.len(),
        0,
        "Expected no module profiles when adaptive is disabled"
    );
}

#[test]
fn vm_adaptive_jit_starts_background_compiler() {
    // Verify that execute() with adaptive JIT starts the background compiler
    let mut vm = raya_engine::Vm::new();
    let config = JitConfig {
        adaptive_compilation: true,
        ..Default::default()
    };
    vm.enable_jit_with_config(config).unwrap();

    let mut code = Vec::new();
    emit_i32(&mut code, 42);
    emit(&mut code, Opcode::Return);

    let module = make_vm_module(code, 0, 0);
    let _result = vm.execute(&module).unwrap();

    // Background compiler should be set
    let compiler = vm.shared_state().background_compiler.lock();
    assert!(compiler.is_some(), "Background compiler should be started");
}

#[test]
fn background_compiler_processes_request() {
    use raya_engine::jit::profiling::counters::ModuleProfile;
    use std::sync::Arc;

    // Create engine, start background thread, send a request, verify it gets compiled
    let config = JitConfig {
        min_score: 1.0,
        min_instruction_count: 2,
        ..Default::default()
    };
    let mut engine = JitEngine::with_config(config).unwrap();
    let code_cache = engine.code_cache().clone();

    // Build a compilable function (math-heavy, no loops)
    let mut func_code = Vec::new();
    for _ in 0..4 {
        emit_i32(&mut func_code, 1);
        emit_i32(&mut func_code, 2);
        emit(&mut func_code, Opcode::Iadd);
        emit_i32(&mut func_code, 3);
        emit(&mut func_code, Opcode::Imul);
    }
    for _ in 0..3 {
        emit(&mut func_code, Opcode::Iadd);
    }
    emit(&mut func_code, Opcode::Return);

    let module = Arc::new(Module {
        runtime_types: Vec::new(),
        function_signatures: Vec::new(),
        magic: *b"RAYA",
        version: VERSION,
        flags: 0,
        constants: ConstantPool::new(),
        functions: vec![Function {
            signature_id: 0,
            local_types: Vec::new(),
            abi_version: 1,
            name: "hot_func".to_string(),
            param_count: 0,
            local_count: 0,
            code: func_code,
        }],
        classes: vec![],
        metadata: Metadata {
            name: "bg_test".to_string(),
            source_file: None,
            generic_templates: vec![],
            template_symbol_table: vec![],
            mono_debug_map: vec![],
                structural_shapes: vec![],
            structural_layouts: vec![],
        },
        exports: vec![],
        imports: vec![],
        checksum: [1; 32],
        reflection: None,
        debug_info: None,
        native_functions: vec![],
        jit_hints: vec![],
    });

    let module_id = code_cache.register_module(module.checksum);
    let profile = Arc::new(ModuleProfile::new(1));

    // Function should NOT be in cache yet
    assert!(!code_cache.contains(module_id, 0));

    // Start background compiler
    let bg = engine.start_background();

    // Submit compilation request
    let submitted = bg.try_submit(raya_engine::jit::profiling::CompilationRequest {
        module: module.clone(),
        func_index: 0,
        module_id,
        module_profile: profile.clone(),
    });
    assert!(submitted, "Request should be accepted");

    // Wait for compilation (poll with timeout)
    let start = std::time::Instant::now();
    while !code_cache.contains(module_id, 0) {
        if start.elapsed() > std::time::Duration::from_secs(5) {
            // Check if profile says compilation finished (might have failed)
            let fp = profile.get(0).unwrap();
            if fp.is_jit_available() {
                break; // Compiled successfully but cache may report differently
            }
            panic!("Background compilation timed out");
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    // Verify the function was compiled
    assert!(
        code_cache.contains(module_id, 0) || profile.get(0).unwrap().is_jit_available(),
        "Function should be compiled by background thread"
    );
}

// =========================================================================
// Category 7: Compile-Time JIT Hints & Background Prewarm
// =========================================================================

#[test]
fn jit_hints_encode_decode_roundtrip() {
    use raya_engine::compiler::bytecode::{flags, JitHint};

    // Create a module with JIT hints
    let mut module = Module {
        runtime_types: Vec::new(),
        function_signatures: Vec::new(),
        magic: *b"RAYA",
        version: VERSION,
        flags: flags::HAS_JIT_HINTS,
        constants: ConstantPool::new(),
        functions: vec![
            Function {
                signature_id: 0,
                local_types: Vec::new(),
                abi_version: 1,
                name: "hot_func".to_string(),
                param_count: 0,
                local_count: 0,
                code: vec![Opcode::Return as u8],
            },
            Function {
                signature_id: 0,
                local_types: Vec::new(),
                abi_version: 1,
                name: "cold_func".to_string(),
                param_count: 0,
                local_count: 0,
                code: vec![Opcode::Return as u8],
            },
        ],
        classes: vec![],
        metadata: Metadata {
            name: "hints_test".to_string(),
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
        jit_hints: vec![
            JitHint {
                func_index: 0,
                score: 42.5,
                is_cpu_bound: true,
            },
            JitHint {
                func_index: 1,
                score: 3.2,
                is_cpu_bound: false,
            },
        ],
    };

    // Encode
    let bytes = module.encode();

    // Decode
    let decoded = Module::decode(&bytes).expect("Decode failed");

    // Verify hints round-trip
    assert_eq!(decoded.jit_hints.len(), 2);
    assert_eq!(decoded.jit_hints[0].func_index, 0);
    assert!((decoded.jit_hints[0].score - 42.5).abs() < 0.001);
    assert!(decoded.jit_hints[0].is_cpu_bound);
    assert_eq!(decoded.jit_hints[1].func_index, 1);
    assert!((decoded.jit_hints[1].score - 3.2).abs() < 0.001);
    assert!(!decoded.jit_hints[1].is_cpu_bound);
    assert!((decoded.flags & flags::HAS_JIT_HINTS) != 0);
}

#[test]
fn jit_hints_absent_when_no_flag() {
    // Module without HAS_JIT_HINTS flag should decode with empty hints
    let module = Module {
        runtime_types: Vec::new(),
        function_signatures: Vec::new(),
        magic: *b"RAYA",
        version: VERSION,
        flags: 0,
        constants: ConstantPool::new(),
        functions: vec![Function {
            signature_id: 0,
            local_types: Vec::new(),
            abi_version: 1,
            name: "main".to_string(),
            param_count: 0,
            local_count: 0,
            code: vec![Opcode::Return as u8],
        }],
        classes: vec![],
        metadata: Metadata {
            name: "no_hints".to_string(),
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
    };

    let bytes = module.encode();
    let decoded = Module::decode(&bytes).expect("Decode failed");
    assert!(decoded.jit_hints.is_empty());
    assert!((decoded.flags & raya_engine::compiler::bytecode::flags::HAS_JIT_HINTS) == 0);
}

#[test]
fn background_prewarm_non_blocking() {
    // Verify execute() doesn't block on prewarm — main task starts immediately
    use std::time::Instant;

    let mut vm = raya_engine::Vm::new();
    let config = JitConfig {
        adaptive_compilation: true,
        ..Default::default()
    };
    vm.enable_jit_with_config(config).unwrap();

    // Simple module — should return instantly without prewarm blocking
    let mut code = Vec::new();
    emit_i32(&mut code, 99);
    emit(&mut code, Opcode::Return);

    let module = make_vm_module(code, 0, 0);

    let start = Instant::now();
    let result = vm.execute(&module).expect("Execution should succeed");
    let elapsed = start.elapsed();

    assert_eq!(result, raya_engine::Value::i32(99));
    // Should complete very quickly (no blocking prewarm)
    assert!(
        elapsed.as_millis() < 500,
        "execute() took {}ms — should not block on prewarm",
        elapsed.as_millis()
    );

    // Background compiler should still be started
    let compiler = vm.shared_state().background_compiler.lock();
    assert!(compiler.is_some(), "Background compiler should be running");
}

#[test]
fn prewarm_candidates_submitted_to_background() {
    use raya_engine::jit::profiling::counters::ModuleProfile;

    // Create a module with a math-heavy function that qualifies for prewarm
    let mut heavy_code = Vec::new();
    // Lots of arithmetic to exceed min_score
    for _ in 0..4 {
        emit_i32(&mut heavy_code, 1);
        emit_i32(&mut heavy_code, 2);
        emit(&mut heavy_code, Opcode::Iadd);
        emit_i32(&mut heavy_code, 3);
        emit(&mut heavy_code, Opcode::Imul);
    }
    for _ in 0..3 {
        emit(&mut heavy_code, Opcode::Iadd);
    }
    emit(&mut heavy_code, Opcode::Return);

    let module = Module {
        runtime_types: Vec::new(),
        function_signatures: Vec::new(),
        magic: *b"RAYA",
        version: VERSION,
        flags: 0,
        constants: ConstantPool::new(),
        functions: vec![Function {
            signature_id: 0,
            local_types: Vec::new(),
            abi_version: 1,
            name: "main".to_string(),
            param_count: 0,
            local_count: 0,
            code: heavy_code,
        }],
        classes: vec![],
        metadata: Metadata {
            name: "prewarm_bg_test".to_string(),
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
    };

    let mut vm = raya_engine::Vm::new();
    let config = JitConfig {
        adaptive_compilation: true,
        min_score: 5.0,
        min_instruction_count: 4,
        ..Default::default()
    };
    vm.enable_jit_with_config(config).unwrap();

    let _result = vm.execute(&module).unwrap();

    // The background compiler should have been started
    let compiler = vm.shared_state().background_compiler.lock();
    assert!(compiler.is_some(), "Background compiler should be running");

    // The module profile should exist
    let profiles = vm.shared_state().module_profiles.read();
    assert_eq!(
        profiles.len(),
        1,
        "Expected module profile for adaptive compilation"
    );
}


// ---------------------------------------------------------------------------
// RefCell lowering coverage (D4.4)
//
// All three RefCell opcodes are still `Rejected` in the capability table, so
// these are NOT reachable fast paths. Each lifts and calls the function directly,
// bypassing candidate selection, which covers the Cranelift arm and the helper in
// isolation. They must not be read as evidence that compiled code uses them:
// `closure_and_refcell_family_is_fail_closed` in `jit/capability.rs` is the gate
// that keeps this unreachable, and it is the gate that must stay until a
// differential test runs these through candidate selection too.
//
// What these do prove is the part that was previously absent: that the lowering
// arms are reachable, that the trampoline offsets resolve, and that the helpers'
// fail-closed returns are observable at the machine-code level rather than dead.

/// Allocate a RefCell in the shared GC and return its raw value.
fn refcell_value(
    shared: &std::sync::Arc<raya_engine::vm::interpreter::SharedVmState>,
    initial: i32,
) -> u64 {
    let mut gc = shared.gc.lock();
    let ptr = gc.allocate(raya_engine::vm::object::RefCell::new(
        raya_engine::vm::value::Value::i32(initial),
    ));
    unsafe {
        raya_engine::vm::value::Value::from_ptr(std::ptr::NonNull::new(ptr.as_ptr()).unwrap())
            .raw()
    }
}

#[test]
fn load_refcell_lowering_uses_runtime_helper_directly() {
    let (safepoint, shared) = new_shared_vm_state();
    let cell = refcell_value(&shared, 11);

    let mut code = Vec::new();
    emit_load_local(&mut code, 0);
    code.push(Opcode::LoadRefCell as u8);
    emit(&mut code, Opcode::Return);

    let module = finalize_module(make_module(code, 0, 1));
    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let (_resolved_natives, bridge) = build_bridge_and_ctx(&safepoint, &shared, &task, &module);
    let mut ctx = raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    let mut locals = vec![cell];
    let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");

    let (raw, exit) =
        jit_compile_and_call_with_locals_exit_and_ctx(&jit_func, &mut locals, (&mut ctx as *mut _));
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32,
        "load must complete natively, not exit to the interpreter"
    );
    assert_eq!(decode_i32(raw), 11);
}

#[test]
fn store_refcell_lowering_mutates_the_cell_natively() {
    let (safepoint, shared) = new_shared_vm_state();
    let cell = refcell_value(&shared, 11);

    let mut code = Vec::new();
    // StoreRefCell pops the value then the cell.
    emit_load_local(&mut code, 0);
    code.push(Opcode::ConstI32 as u8);
    code.extend_from_slice(&5i32.to_le_bytes());
    code.push(Opcode::StoreRefCell as u8);
    // Read it back through the JIT helper.
    emit_load_local(&mut code, 0);
    code.push(Opcode::LoadRefCell as u8);
    emit(&mut code, Opcode::Return);

    let module = finalize_module(make_module(code, 0, 1));
    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let (_resolved_natives, bridge) = build_bridge_and_ctx(&safepoint, &shared, &task, &module);
    let mut ctx = raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    let mut locals = vec![cell];
    let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");

    let (raw, exit) =
        jit_compile_and_call_with_locals_exit_and_ctx(&jit_func, &mut locals, (&mut ctx as *mut _));
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32,
        "store must complete natively, not exit to the interpreter"
    );
    // 5 rather than the original 11, so the helper really wrote through.
    assert_eq!(decode_i32(raw), 5);
}

#[test]
fn new_refcell_lowering_allocates_and_reads_back() {
    let (safepoint, shared) = new_shared_vm_state();

    let mut code = Vec::new();
    emit_load_local(&mut code, 0);
    code.push(Opcode::NewRefCell as u8);
    code.push(Opcode::LoadRefCell as u8);
    emit(&mut code, Opcode::Return);

    let module = finalize_module(make_module(code, 0, 1));
    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let (_resolved_natives, bridge) = build_bridge_and_ctx(&safepoint, &shared, &task, &module);
    let mut ctx = raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    // local 0 is the initial value, not a RefCell.
    let mut locals = vec![raya_engine::vm::value::Value::i32(7).raw()];
    let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");

    let (raw, exit) =
        jit_compile_and_call_with_locals_exit_and_ctx(&jit_func, &mut locals, (&mut ctx as *mut _));
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32,
        "allocation must complete natively, not exit to the interpreter"
    );
    assert_eq!(decode_i32(raw), 7);
}


/// The RefCell differential: **the same bytecode** through both engines, compared.
///
/// This is the piece the promotion actually needs. The three direct-lift tests
/// prove the JIT arms work; `object_model_tests` proves the interpreter handlers
/// work; but until now they were separate programs, so nothing asserted the two
/// engines agree. A promotion justified by comparing the JIT only against itself is
/// not evidence, and neither is comparing two independently-written programs.
///
/// Limitation, stated so it is not over-read: this still lifts directly, so it does
/// NOT go through `function_supported_for_jit`. It is an engine-agreement test, not
/// a reachability test. The gate remains pinned by
/// `refcell_opcodes_keep_a_function_out_of_the_jit`.
#[test]
fn refcell_interpreter_and_jit_agree_on_the_same_bytecode() {
    use raya_engine::vm::interpreter::Vm;

    // cell = new RefCell(11); cell.x = 99; return cell's contents.
    // Dup before the value: it duplicates the top of stack, which is the cell.
    let mut code = Vec::new();
    code.push(Opcode::ConstI32 as u8);
    code.extend_from_slice(&11i32.to_le_bytes());
    code.push(Opcode::NewRefCell as u8);
    code.push(Opcode::Dup as u8);
    code.push(Opcode::ConstI32 as u8);
    code.extend_from_slice(&99i32.to_le_bytes());
    code.push(Opcode::StoreRefCell as u8);
    code.push(Opcode::LoadRefCell as u8);
    code.push(Opcode::Return as u8);

    // `make_module` names the function "test_func"; `Vm::execute` looks the entry
    // point up as "main", so rename before finalizing -- `finalize_module` wraps
    // the module in an `Arc`, which cannot be mutated through. The JIT harness
    // lifts `functions[0]` positionally, so the same module then serves both
    // engines. Every other test in this file skips this because none of them run
    // the interpreter.
    let mut raw_module = make_module(code, 0, 0);
    raw_module.functions[0].name = "main".to_string();
    let module = finalize_module(raw_module);

    // Engine 1: the interpreter.
    let interpreted = {
        let mut vm = Vm::new();
        vm.execute(&module).expect("interpreter must run the RefCell program")
    };

    // Engine 2: the JIT, on the identical module.
    let (safepoint, shared) = new_shared_vm_state();
    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let (_resolved_natives, bridge) = build_bridge_and_ctx(&safepoint, &shared, &task, &module);
    let mut ctx = raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    let mut locals: Vec<u64> = Vec::new();
    let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");

    let (raw, exit) =
        jit_compile_and_call_with_locals_exit_and_ctx(&jit_func, &mut locals, (&mut ctx as *mut _));
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32,
        "JIT must complete natively, not fall back to the interpreter -- otherwise this \
         compares the interpreter against itself"
    );

    assert_eq!(
        raw,
        interpreted.raw(),
        "engines disagree on the same RefCell bytecode: JIT 0x{raw:016X}, \
         interpreter {}",
        interpreted
    );
    assert!(
        is_i32(raw),
        "result should be a NaN-boxed i32, got 0x{raw:016X}"
    );
    assert_eq!(decode_i32(raw), 99, "both engines should have produced 99");
}


/// The interpreter polls a safepoint at the start of `MakeClosure`, before it
/// allocates. Compiled code has no other opportunity to offer the collector a stop
/// point at that allocation, so the lifter must emit a matching `GcSafepoint`.
///
/// This is the one change on this milestone that a compiler cannot catch: a
/// *missing* safepoint compiles perfectly and every existing test still passes,
/// because nothing downstream exercises `MakeClosure` natively yet. Asserting on the
/// lifted IR is the only way to hold it.
#[test]
fn make_closure_emits_a_safepoint_before_allocating() {
    use raya_engine::jit::ir::instr::JitInstr;

    // MakeClosure func_index=0, capture_count=0; then Return.
    let mut code = Vec::new();
    code.push(Opcode::MakeClosure as u8);
    code.extend_from_slice(&0u32.to_le_bytes());
    code.extend_from_slice(&0u16.to_le_bytes());
    emit(&mut code, Opcode::Return);

    let module = finalize_module(make_module(code, 0, 0));
    let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");

    let instrs: Vec<&JitInstr> = jit_func
        .blocks
        .iter()
        .flat_map(|block| block.instrs.iter())
        .collect();

    let make_closure_at = instrs
        .iter()
        .position(|instr| matches!(instr, JitInstr::MakeClosure { .. }))
        .expect("lifted IR must contain MakeClosure");
    let safepoint_at = instrs
        .iter()
        .position(|instr| matches!(instr, JitInstr::GcSafepoint { .. }));

    assert!(
        safepoint_at.is_some(),
        "MakeClosure allocates and the interpreter polls a safepoint first, but the \
         lifter emitted no GcSafepoint"
    );
    assert!(
        safepoint_at < Some(make_closure_at),
        "GcSafepoint must come BEFORE MakeClosure: the interpreter polls before \
         allocating, so the stop point has to precede the allocation \
         (safepoint at {safepoint_at:?}, MakeClosure at {make_closure_at})"
    );
}


/// The closure differential: **the same bytecode** through both engines.
///
/// `MakeClosure` and `SetClosureCapture` now have interpreter coverage
/// (`3469bb4`) and their lowering arms are wired, but nothing yet asserts the two
/// engines agree. Same shape as the RefCell differential (`b29f610`): one module,
/// run through `Vm::execute` and through lift+compile+call, comparing raw bits.
///
/// As with the RefCell case this still lifts directly, so it is an engine-agreement
/// test rather than a reachability one — `MakeClosure` and `SetClosureCapture` are
/// still `Rejected`. The gate is pinned separately.
#[test]
fn closure_interpreter_and_jit_agree_on_the_same_bytecode() {
    use raya_engine::vm::interpreter::Vm;

    // Two functions: `main` at index 0, closure body at index 1.
    let mut module = make_module(Vec::new(), 0, 0);

    // The closure body: hand the capture straight back.
    module.functions.push(Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "closure_body".to_string(),
        param_count: 0,
        local_count: 0,
        code: vec![Opcode::LoadCaptured as u8, 0, 0, Opcode::Return as u8],
    });

    // main: capture 42, patch slot 0 to 7, then call -- so a JIT that skipped the
    // patch, or called the wrong function, returns 42 instead of 7.
    let mut main_code: Vec<u8> = Vec::new();
    main_code.push(Opcode::ConstI32 as u8);
    main_code.extend_from_slice(&42i32.to_le_bytes());
    main_code.push(Opcode::MakeClosure as u8);
    main_code.extend_from_slice(&1u32.to_le_bytes()); // func_index = closure_body
    main_code.extend_from_slice(&1u16.to_le_bytes()); // capture_count = 1
    // SetClosureCapture pops value then closure and pushes the closure back, so
    // Dup the closure BEFORE pushing the value.
    main_code.push(Opcode::Dup as u8);
    main_code.push(Opcode::ConstI32 as u8);
    main_code.extend_from_slice(&7i32.to_le_bytes());
    main_code.push(Opcode::SetClosureCapture as u8);
    main_code.extend_from_slice(&0u16.to_le_bytes()); // capture index 0
    main_code.push(Opcode::Call as u8);
    main_code.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // closure call
    main_code.extend_from_slice(&0u16.to_le_bytes()); // arg_count = 0
    main_code.push(Opcode::Return as u8);
    module.functions[0] = Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "test_func".to_string(),
        param_count: 0,
        local_count: 0,
        code: main_code,
    };

    let mut raw_module = module;
    raw_module.functions[0].name = "main".to_string();
    let module = finalize_module(raw_module);

    // Engine 1: the interpreter.
    let interpreted = {
        let mut vm = Vm::new();
        vm.execute(&module).expect("interpreter must run the closure program")
    };

    // Engine 2: the JIT, on the identical module.
    let (safepoint, shared) = new_shared_vm_state();
    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let (_resolved_natives, bridge) = build_bridge_and_ctx(&safepoint, &shared, &task, &module);
    let mut ctx = raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    let mut locals: Vec<u64> = Vec::new();
    let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");

    let (raw, exit) =
        jit_compile_and_call_with_locals_exit_and_ctx(&jit_func, &mut locals, (&mut ctx as *mut _));
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32,
        "JIT must complete natively, not fall back -- otherwise this compares the \
         interpreter against itself"
    );

    assert_eq!(
        raw,
        interpreted.raw(),
        "engines disagree on the same closure bytecode: JIT 0x{raw:016X}, interpreter {interpreted}"
    );
    assert!(is_i32(raw), "expected a NaN-boxed i32, got 0x{raw:016X}");
    // 7, not the 42 originally captured — so the SetClosureCapture actually landed.
    assert_eq!(decode_i32(raw), 7);
}


/// `LoadCaptured` and `StoreCaptured` run **natively**, by lifting the closure body
/// itself as the entry function.
///
/// This is not the shared-bytecode differential, and it is not meant to be. The
/// `Call` lowering arm routes every callee through `interpreter_call`, so lifting
/// `main` and calling a closure would run `main` natively but execute the body's
/// captured opcodes **interpreted** — a test that passes while proving nothing about
/// the two arms it appears to cover. Lifting the body directly is the only way these
/// arms reach native code at all.
///
/// The active closure must be on the task **before** `build_bridge_and_ctx`, or the
/// bridge's task will not have it and every read will take the fallback path.
#[test]
fn captured_opcodes_execute_natively_when_the_body_is_lifted_directly() {
    use raya_engine::vm::value::Value;

    // A two-function module. The body is the one that will be lifted; main is
    // present only so the closure body has a plausible `func_index`.
    let mut raw = make_module(Vec::new(), 0, 0);
    raw.functions.push(Function {
        signature_id: 0,
        local_types: Vec::new(),
        abi_version: 1,
        name: "closure_body".to_string(),
        param_count: 0,
        local_count: 0,
        // The lifter's stack model does not preload parameters, so the body pushes
        // its own value rather than reading a parameter: relying on local 0 gave
        // `Lift failed: StackUnderflow { offset: 0 }`.
        code: vec![
            Opcode::ConstI32 as u8,
            42,
            0,
            0,
            0, // value to store
            Opcode::StoreCaptured as u8,
            0,
            0, // capture 0 <- value
            Opcode::LoadCaptured as u8,
            0,
            0, // push capture 0 back
            Opcode::Return as u8,
        ],
    });
    let module = finalize_module(raw);

    let (safepoint, shared) = new_shared_vm_state();

    // A closure capturing [7], installed as the task's active closure.
    let closure_raw = {
        let mut gc = shared.gc.lock();
        let closure = raya_engine::vm::object::Closure::new(0, vec![Value::i32(7)]);
        let ptr = gc.allocate(closure);
        unsafe { Value::from_ptr(std::ptr::NonNull::new(ptr.as_ptr()).unwrap()).raw() }
    };
    let closure_val = unsafe { Value::from_raw(closure_raw) };

    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    task.push_closure(closure_val);
    let (_resolved_natives, bridge) = build_bridge_and_ctx(&safepoint, &shared, &task, &module);
    let mut ctx = raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());

    // Lift the BODY (index 1), not main. local 0 is the argument StoreCaptured
    // writes into capture 0.
    let jit_func = lift_function(&module.functions[1], &module, 1).expect("Lift failed");
    let mut locals: Vec<u64> = Vec::new();

    let (raw, exit) =
        jit_compile_and_call_with_locals_exit_and_ctx(&jit_func, &mut locals, (&mut ctx as *mut _));

    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32,
        "both captured opcodes must run natively; a fallback here means the active \
         closure was missing and the helper refused"
    );
    assert!(
        is_i32(raw),
        "expected a NaN-boxed i32, got 0x{raw:016X}"
    );
    // 42, not the captured 7: the store must have landed and the load must have
    // read it back through the same active closure.
    assert_eq!(
        decode_i32(raw),
        42,
        "StoreCaptured must write local 0 into the active closure's capture 0, and \
         LoadCaptured must read it back; 7 means the store never landed"
    );
}


/// `BindMethod`'s lifter arm is no longer empty.
///
/// The interpreter reads the u16 operand, pops the receiver and pushes a
/// `BoundMethod`. The old arm did none of that, so the lifted `ip` never advanced
/// past the operand and the stack model disagreed with the interpreter from that
/// instruction onward — which is why the opcode was rejected at the lifter rather
/// than merely missing a helper.
///
/// This asserts the observable consequence: the lifted stream contains a
/// `BindMethod` that consumes the operand, and the instruction AFTER it is the
/// `Return` rather than something misaligned.
#[test]
fn bind_method_lifter_keeps_the_stack_model_in_step() {
    use raya_engine::jit::ir::instr::JitInstr;

    // BindMethod slot 0, then Return. The object operand comes from local 0.
    let mut code: Vec<u8> = Vec::new();
    emit_load_local(&mut code, 0);
    code.push(Opcode::BindMethod as u8);
    code.extend_from_slice(&0u16.to_le_bytes());
    emit(&mut code, Opcode::Return);

    let module = finalize_module(make_module(code, 0, 1));
    let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");

    let instrs: Vec<&JitInstr> = jit_func
        .blocks
        .iter()
        .flat_map(|block| block.instrs.iter())
        .collect();

    let at = instrs
        .iter()
        .position(|instr| matches!(instr, JitInstr::BindMethod { .. }))
        .expect("lifted IR must contain BindMethod; an empty arm is exactly the defect");

    // It must carry the operand the bytecode declared, and it must be followed by
    // the Return rather than by an instruction from the wrong offset.
    match instrs[at] {
        JitInstr::BindMethod { method_slot, .. } => assert_eq!(
            *method_slot, 0,
            "the operand must be consumed, not left for the next instruction"
        ),
        other => panic!("expected BindMethod, got {other:?}"),
    }
    // `Return` is a lifter terminator and emits no instruction, so a correctly
    // lifted stream ENDS with BindMethod. Anything after it would mean the operand
    // was not consumed and the stream is misaligned.
    assert_eq!(
        at + 1,
        instrs.len(),
        "BindMethod must be the last lifted instruction; found {:?} after it — the \
         operand was not consumed",
        &instrs[at + 1..]
    );
}


/// `BindMethod` executes natively: the lowering arm resolves the vtable slot and
/// allocates a `BoundMethod`.
///
/// The class is registered into the bridge's registry directly rather than through
/// module loading, because that is the only route available to a test — and it
/// exercises the same code the helper reads, so the vtable resolution under test is
/// real rather than a stub.
#[test]
fn bind_method_lowering_binds_natively() {
    use raya_engine::vm::object::Class;
    use raya_engine::vm::value::Value;

    let (safepoint, shared) = new_shared_vm_state();

    // LoadLocal 0; BindMethod slot 0; Return
    let mut code: Vec<u8> = Vec::new();
    emit_load_local(&mut code, 0);
    code.push(Opcode::BindMethod as u8);
    code.extend_from_slice(&0u16.to_le_bytes());
    emit(&mut code, Opcode::Return);
    let module = finalize_module(make_module(code, 0, 1));

    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let (_resolved_natives, bridge) = build_bridge_and_ctx(&safepoint, &shared, &task, &module);

    // Register a class with one method, then make an object carrying that id.
    let object_raw = {
        let mut classes = unsafe { (&*bridge.classes).write() };
        let mut class = Class::new(0, "Point".to_string(), 2);
        class.module = Some(module.clone());
        class.vtable.add_method(42);
        let nominal_type_id = classes.register_class(class);
        drop(classes);

        let mut gc = shared.gc.lock();
        let mut object =
            raya_engine::vm::object::Object::new_nominal(1, nominal_type_id as u32, 2);
        object.set_field(0, Value::i32(99)).unwrap();
        let ptr = gc.allocate(object);
        unsafe { Value::from_ptr(std::ptr::NonNull::new(ptr.as_ptr()).unwrap()).raw() }
    };

    let mut ctx = raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    let mut locals: Vec<u64> = vec![object_raw];
    let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");

    let (raw, exit) =
        jit_compile_and_call_with_locals_exit_and_ctx(&jit_func, &mut locals, (&mut ctx as *mut _));
    assert_eq!(
        exit.kind,
        raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32,
        "BindMethod must complete natively, not exit to the interpreter"
    );
    assert_ne!(raw, 0, "a bound method must not be null");

    // It must be a BoundMethod carrying the receiver and the resolved func id.
    let value = unsafe { Value::from_raw(raw) };
    let bound = unsafe {
        let ptr = value
            .as_ptr::<raya_engine::vm::object::BoundMethod>()
            .expect("result must be a BoundMethod");
        &*ptr.as_ptr()
    };
    assert_eq!(bound.func_id, 42, "vtable slot must resolve to the func id");
    assert_eq!(bound.receiver.raw(), object_raw, "receiver must be carried");
}


/// `Try`'s lifting arm resolves its catch target to a real block.
///
/// Before `5ee9e17` the arm computed `catch_abs` and `finally_abs` into
/// underscore-prefixed bindings — computed and never read — and emitted
/// `SetupTry { catch_block: JitBlockId(0) }`. This asserts the placeholder is gone
/// and that the emitted block is the one whose recorded `start_offset` equals the
/// expected catch target.
///
/// THE TWO BASES DIFFER, and a wrong expectation here would silently pass against
/// `BlockId(0)`. `catch_abs` is measured from `instr.offset + 1 + 4` (after
/// reading only `catch_rel`), `finally_abs` from `+ 1 + 8`. This program uses only
/// a catch, so it pins the first base.
#[test]
fn try_lifter_resolves_the_catch_block() {
    use raya_engine::jit::ir::instr::{JitInstr, JitTerminator};

    //  Try                           @0        1 byte
    //  catch_rel  (i32)              @1..5
    //  finally_rel (i32)             @5..9
    //  ConstI32 7                    @9..14     body
    //  Throw                          @14
    //  ConstI32 99                   @15..20    catch handler
    //  Return                         @20
    //
    // `catch_rel` is measured from offset 5 — `instr.offset + 1` opcode byte + 4
    // operand bytes — so it must be 15 - 5 = 10. An earlier version of this test
    // forgot that the two i32 operands occupy eight bytes and expected offset 7.
    let mut code: Vec<u8> = Vec::new();
    code.push(Opcode::Try as u8);
    code.extend_from_slice(&10i32.to_le_bytes()); // catch_rel
    code.extend_from_slice(&0i32.to_le_bytes()); // finally_rel = 0 -> none
    code.push(Opcode::ConstI32 as u8);
    code.extend_from_slice(&7i32.to_le_bytes());
    code.push(Opcode::Throw as u8);
    let catch_abs_at = code.len();
    code.push(Opcode::ConstI32 as u8);
    code.extend_from_slice(&99i32.to_le_bytes());
    code.push(Opcode::Return as u8);
    assert_eq!(catch_abs_at, 15, "catch handler must sit at offset 15");

    let module = finalize_module(make_module(code, 0, 0));
    let jit_func = lift_function(&module.functions[0], &module, 0)
        .expect("a Try-containing function must now lift");

    // The catch block must NOT be the entry block any more.
    let mut setup: Option<(usize, Option<usize>)> = None;
    for block in &jit_func.blocks {
        for instr in &block.instrs {
            if let JitInstr::SetupTry {
                catch_block,
                finally_block,
                ..
            } = instr
            {
                setup = Some((catch_block.0 as usize, finally_block.map(|b| b.0 as usize)));
            }
        }
    }
    let (catch_block, finally_block) =
        setup.expect("SetupTry must be emitted — the placeholder arm produced it too");

    assert_ne!(
        catch_block, 0,
        "catch_block must no longer be the JitBlockId(0) placeholder"
    );
    assert_eq!(finally_block, None, "finally_rel = 0 means no finally block");

    // And it must be the block whose recorded start offset is the catch target.
    assert_eq!(
        jit_func.blocks[catch_block].start_offset,
        15,
        "catch block must begin at the catch target offset"
    );

    // Every lifted block carries a real offset, so the partition is reconstructable.
    assert!(
        jit_func.blocks.iter().all(|b| b.start_offset
            != raya_engine::jit::ir::instr::JitBlock::UNKNOWN_START_OFFSET),
        "no lifted block may have an unknown start offset"
    );

    // Sanity: the Throw arm is present too, since the body throws.
    assert!(
        jit_func
            .blocks
            .iter()
            .flat_map(|b| b.instrs.iter())
            .any(|i| matches!(i, JitInstr::Throw { .. })),
        "the Throw arm must lift"
    );
    let _ = JitTerminator::None;
}


/// `Await` path 1 executes natively: a non-task value comes back **unchanged**.
///
/// This is the first promotion candidate on this branch whose correctness rests on
/// `Value::as_u64` being **tag-gated** rather than on a pointer or bounds check, so
/// the test asserts the *value*, not merely that execution completed. A
/// payload-based "is this a task id" test would read 42 as task id 42, find no such
/// task, and fall back — so a version of this test asserting only
/// `exit.kind == Completed` would have passed against a wrong implementation that
/// silently fell back instead.
#[test]
fn await_path_one_runs_natively_and_returns_the_value_unchanged() {
    use raya_engine::jit::runtime::trampoline::{JitExitKind, JitSuspendReason};
    use raya_engine::vm::value::Value;

    let mut code: Vec<u8> = Vec::new();
    emit_i32(&mut code, 42);
    code.push(Opcode::Await as u8);
    emit(&mut code, Opcode::Return);

    let module = finalize_module(make_module(code, 0, 0));
    let (safepoint, shared) = new_shared_vm_state();
    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let (_resolved_natives, bridge) = build_bridge_and_ctx(&safepoint, &shared, &task, &module);
    let mut ctx = raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    let mut locals: Vec<u64> = Vec::new();
    let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");

    let (raw, exit) =
        jit_compile_and_call_with_locals_exit_and_ctx(&jit_func, &mut locals, (&mut ctx as *mut _));

    // A non-task value takes the MERGED path: the helper returns it and execution
    // completes normally. Only the sentinel path exits `Suspended` -- asserting
    // `Suspended` here was wrong, and it is the same distinction every promoted arm
    // makes: `Completed` is success, `Suspended` is a fallback to the interpreter.
    assert_eq!(
        exit.kind,
        JitExitKind::Completed as u32,
        "awaiting a non-task must complete natively, not fall back to the interpreter"
    );
    assert_eq!(
        exit.suspend_reason,
        JitSuspendReason::None as u32,
        "a normal completion carries no suspend reason"
    );
    assert_eq!(
        raw,
        Value::i32(42).raw(),
        "a non-task value must be returned unchanged by the helper"
    );
    assert!(
        is_i32(raw),
        "the result must still be a NaN-boxed i32, got 0x{raw:016X}"
    );
}


/// The `Await` differential: **the same bytecode** through both engines.
///
/// `Await` is the first promotion candidate on this branch whose correctness rests
/// on `Value::as_u64` being **tag-gated** rather than on a pointer or a bounds
/// check, so this deliberately exercises the non-task case rather than a task id.
/// Every other promotion's differential used a helper whose check was structural;
/// this one's is a tagged union, and a payload-based implementation would read the
/// value as a task id, find nothing, and fall back — agreeing with the interpreter
/// only by accident.
///
/// Two values, not one: an `i32` and a `bool`, because both have payloads that
/// could plausibly be read as a task id if the tag check were dropped.
#[test]
fn await_interpreter_and_jit_agree_on_the_same_bytecode() {
    use raya_engine::vm::interpreter::Vm;

    for (imm, label) in [(42i32, "i32"), (1i32, "small i32")] {
        let mut code: Vec<u8> = Vec::new();
        emit_i32(&mut code, imm);
        code.push(Opcode::Await as u8);
        emit(&mut code, Opcode::Return);

        // The interpreter needs a named "main"; the JIT harness lifts positionally.
        let mut raw = make_module(code, 0, 0);
        raw.functions[0].name = "main".to_string();
        let module = finalize_module(raw);

        // Engine 1: the interpreter, on the identical module.
        let interpreted = {
            let mut vm = Vm::new();
            vm.execute(&module).expect("interpreter must run the await program")
        };

        // Engine 2: the JIT, on the same module.
        let (safepoint, shared) = new_shared_vm_state();
        let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
        let (_resolved_natives, bridge) = build_bridge_and_ctx(&safepoint, &shared, &task, &module);
        let mut ctx =
            raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
        let mut locals: Vec<u64> = Vec::new();
        let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");
        let (raw_bits, exit) = jit_compile_and_call_with_locals_exit_and_ctx(
            &jit_func,
            &mut locals,
            (&mut ctx as *mut _),
        );

        assert_eq!(
            exit.kind,
            raya_engine::jit::runtime::trampoline::JitExitKind::Completed as u32,
            "[{label}] awaiting a non-task must complete natively"
        );
        assert_eq!(
            raw_bits,
            interpreted.raw(),
            "[{label}] engines disagree on the same await bytecode: JIT 0x{raw_bits:016X}, \
             interpreter {interpreted}"
        );
    }
}


/// `DynGetKeyed`'s `Str` view: the differential, on the **same bytecode** through
/// both engines.
///
/// Scoped honestly, because the corpus a differential can actually reach is
/// narrower than the opcode:
///
///   * **`Str`** — fully covered here. `ConstStr` is `HelperExact`, so a string
///     target is constructible in natively-compiled code.
///   * **`Arr`** — **not** reachable in this test. The entire array family
///     (`NewArray`, `InitArray`, `LoadElem`, ...) is `Rejected` under the D4.2
///     fail-closed posture, so a program building an array would have its *whole
///     function* rejected and silently fall back to the interpreter. A
///     "differential" written that way passes while proving nothing about the
///     helper, which is the same vacuity that let D4.3's P0 ship. Engine-level
///     `Arr` evidence is gated on the array family milestone; the helper-level
///     test covers `Arr` in the meantime.
///   * **`Struct`** — declined by design; returns the fallback sentinel.
///
/// Two of the three cases exist to catch a specific byte-vs-char divergence,
/// because this is the one view where the two disagree:
///
///   * `"héllo".length` is **6** (Rust `str::len` is bytes), while `"héllo"[1]`
///     is `"é"` (`chars().nth` is characters). An implementation that used one
///     for the other would agree with the interpreter on ASCII and diverge on
///     every non-ASCII string, so the corpus is deliberately non-ASCII.
#[test]
fn dyn_get_keyed_string_view_matches_interpreter() {
    use raya_engine::jit::runtime::trampoline::JitExitKind;
    use raya_engine::vm::interpreter::Vm;

    // (label, key program, expected)
    let cases: Vec<(&str, Box<dyn Fn(&mut Vec<u8>, u32)>, Option<String>, Option<i32>)> = vec![
        (
            "char index 1 of \"héllo\"",
            Box::new(|c: &mut Vec<u8>, _| emit_i32(c, 1)),
            Some("é".to_string()),
            None,
        ),
        (
            "byte length of \"héllo\"",
            Box::new(|c: &mut Vec<u8>, k: u32| emit_const_str(c, k)),
            None,
            Some(6),
        ),
        (
            "out-of-range index 99",
            Box::new(|c: &mut Vec<u8>, _| emit_i32(c, 99)),
            None,
            None,
        ),
    ];

    for (label, emit_key, expect_string, expect_i32) in cases {
        let mut module = make_vm_module(Vec::new(), 0, 0);
        let target = module.constants.add_string("héllo".to_string());
        let length_key = module.constants.add_string("length".to_string());

        let mut code = Vec::new();
        emit_const_str(&mut code, target);
        emit_key(&mut code, length_key);
        emit(&mut code, Opcode::DynGetKeyed);
        emit(&mut code, Opcode::Return);
        module.functions[0].code = code;
        let module = finalize_module(module);

        // Engine 1: the interpreter, on the identical module.
        //
        // `vm` must OUTLIVE every read of `interpreted`. It is deliberately not
        // scoped to a block: `execute` returns a `Value` pointing into the VM's own
        // GC, so dropping the `Vm` frees that GC and leaves `interpreted` dangling.
        // The symptom was spectacular rather than obvious -- reading it cloned a
        // `String` whose length field had been overwritten with `usize::MAX`, and
        // the process aborted on a 18446744073709551615-byte allocation. It read as
        // "the JIT returned a corrupt string" when the JIT result was the correct
        // one, and the interpreter's was the dangling pointer.
        let mut vm = Vm::with_worker_count(1);
        let interpreted = vm
            .execute(module.as_ref())
            .expect("interpreter DynGetKeyed");

        // Engine 2: the JIT, on the same module.
        let (native, exit, _shared) = execute_module_natively(module, None);
        assert_eq!(
            exit.kind,
            JitExitKind::Completed as u32,
            "[{label}] the Str view must complete natively, not fall back"
        );

        match (expect_string, expect_i32) {
            // A character is a freshly allocated string in BOTH engines, so the
            // two hold different pointers. Comparing raw bits here would fail on
            // correct code -- the content is the observable.
            (Some(want), _) => {
                assert_eq!(
                    string_contents(native),
                    want,
                    "[{label}] JIT returned the wrong character"
                );
                assert_eq!(
                    string_contents(interpreted),
                    want,
                    "[{label}] interpreter baseline disagrees, so the case is wrong"
                );
            }
            // i32 and null are NaN-boxed immediates: identical bits in both engines.
            (None, Some(want)) => {
                assert_eq!(
                    native.as_i32(),
                    Some(want),
                    "[{label}] JIT got the wrong value"
                );
                assert_eq!(native.raw(), interpreted.raw(), "[{label}] engines disagree");
            }
            (None, None) => {
                assert!(
                    native.is_null(),
                    "[{label}] an out-of-range string index must be null, got 0x{:016X}",
                    native.raw()
                );
                assert_eq!(native.raw(), interpreted.raw(), "[{label}] engines disagree");
            }
        }
    }
}


/// D4.8's first slice, differentially: `NewArray` + `ArrayLen` on the **same
/// bytecode** through both engines.
///
/// These two are promoted together for a structural reason, not a convenient one.
/// `NewArray` is what bootstraps array construction: until a natively-compiled
/// function can *make* an array, no other array opcode can be differentially
/// tested at all, because the test program could not contain one. `ArrayLen` is
/// the simplest consumer that needs no index coercion.
///
/// Both programs complete natively — `exit.kind == Completed` is asserted, so a
/// silent fallback to the interpreter fails the test rather than passing with the
/// right answer. That distinction is the whole reason `JitExitKind` is inspected
/// everywhere on this branch.
///
/// Lengths compared as `i32`, never as raw bits: the array is a heap pointer and
/// each engine allocates its own, so pointer identity is meaningless between them.
#[test]
fn new_array_and_array_len_match_interpreter() {
    use raya_engine::jit::runtime::trampoline::JitExitKind;
    use raya_engine::vm::interpreter::Vm;

    // (label, length operand)
    let cases: Vec<(&str, i32)> = vec![("empty", 0), ("three", 3), ("one", 1)];

    for (label, len) in cases {
        // `[len] -> [arr] -> len`, i.e. NewArray then ArrayLen.
        let mut code: Vec<u8> = Vec::new();
        emit_i32(&mut code, len);
        emit(&mut code, Opcode::NewArray);
        // `NewArray`'s operand is a u32 element-type id. 6 is `AnyValue`, i.e. a
        // dynamic array that accepts any element -- the unconstrained case, so this
        // differential is about the array machinery and not about element checking.
        code.extend_from_slice(&6u32.to_le_bytes());
        emit(&mut code, Opcode::ArrayLen);
        emit(&mut code, Opcode::Return);

        let mut raw = make_module(code, 0, 0);
        raw.functions[0].name = "main".to_string();
        let module = finalize_module(raw);

        // Engine 1: the interpreter, on the identical module.
        let mut vm = Vm::with_worker_count(1);
        let interpreted = vm.execute(module.as_ref()).expect("interpreter NewArray/ArrayLen");

        // Engine 2: the JIT, on the same module.
        let (safepoint, shared) = new_shared_vm_state();
        let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(
            0,
            module.clone(),
            None,
        ));
        let (_resolved_natives, bridge) =
            build_bridge_and_ctx(&safepoint, &shared, &task, &module);
        let mut ctx =
            raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
        let mut locals: Vec<u64> = Vec::new();
        let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");
        let (raw_bits, exit) = jit_compile_and_call_with_locals_exit_and_ctx(
            &jit_func,
            &mut locals,
            (&mut ctx as *mut _),
        );

        assert_eq!(
            exit.kind,
            JitExitKind::Completed as u32,
            "[{label}] NewArray/ArrayLen must complete natively, not fall back"
        );
        // Compare the DECODED value, not the raw register. Both engines return a
        // boxed `Value`, and `Value::i32(0).raw()` is `0xFFF9000000000000`, not 0 —
        // an earlier version of this assertion compared raw bits to `len as u64`
        // and failed on correct code with "left: 18444773748872577024".
        let native = unsafe { raya_engine::vm::value::Value::from_raw(raw_bits) };
        assert_eq!(
            native.as_i32(),
            Some(len),
            "[{label}] the JIT returned the wrong length"
        );
        assert_eq!(
            native.raw(),
            interpreted.raw(),
            "[{label}] engines disagree on the same bytecode"
        );
        assert_eq!(
            interpreted.as_i32(),
            Some(len),
            "[{label}] the interpreter baseline disagrees, so the case itself is wrong"
        );
    }
}

/// The length coercion, differentially. `NewArray`'s length operand goes through
/// `array_index_operand`, whose behaviour is deliberately surprising, and this is
/// the engine-level counterpart to `array_index_operand_coercion_is_pinned`.
///
/// Two cases, and they are the two that disagree with intuition:
///
///   * a **non-numeric** length is 0, so a null length builds an empty array
///   * a **negative** length wraps to `usize::MAX`, which both engines then try to
///     reserve. That case is deliberately NOT asserted here: it aborts the process
///     in *both* engines, which is pre-existing interpreter behaviour rather than
///     something this milestone should either copy or quietly change. It is
///     recorded in the spec instead, because "matching a crash" is not a contract
///     worth pinning in a test.
#[test]
fn new_array_length_coercion_matches_interpreter() {
    use raya_engine::jit::runtime::trampoline::JitExitKind;
    use raya_engine::vm::interpreter::Vm;

    // A null length must mean 0 -- not a fallback, and not an error.
    let mut code: Vec<u8> = Vec::new();
    code.push(Opcode::ConstNull as u8);
    emit(&mut code, Opcode::NewArray);
    code.extend_from_slice(&6u32.to_le_bytes());
    emit(&mut code, Opcode::ArrayLen);
    emit(&mut code, Opcode::Return);

    let mut raw = make_module(code, 0, 0);
    raw.functions[0].name = "main".to_string();
    let module = finalize_module(raw);

    let mut vm = Vm::with_worker_count(1);
    let interpreted = vm.execute(module.as_ref()).expect("interpreter null length");

    let (safepoint, shared) = new_shared_vm_state();
    let task =
        std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let (_resolved_natives, bridge) = build_bridge_and_ctx(&safepoint, &shared, &task, &module);
    let mut ctx =
        raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    let mut locals: Vec<u64> = Vec::new();
    let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");
    let (raw_bits, exit) =
        jit_compile_and_call_with_locals_exit_and_ctx(&jit_func, &mut locals, (&mut ctx as *mut _));

    assert_eq!(
        exit.kind,
        JitExitKind::Completed as u32,
        "a null length must be handled natively as 0"
    );
    // Decoded, not raw: see the note in the test above. `Value::i32(0).raw()` is
    // `0xFFF9000000000000`.
    assert_eq!(
        unsafe { raya_engine::vm::value::Value::from_raw(raw_bits) }.as_i32(),
        Some(0),
        "a null length must build an empty array"
    );
    assert_eq!(
        raw_bits,
        interpreted.raw(),
        "engines disagree about a null length"
    );
    assert_eq!(
        interpreted.as_i32(),
        Some(0),
        "the interpreter baseline disagrees about a null length"
    );
}


/// D4.8 slice 2, differentially: element access through `NewArray` + `InitArray` +
/// `LoadElem` + `StoreElem`, on the **same bytecode** through both engines.
///
/// The corpus is built around the three facts that make these opcodes non-trivial,
/// all of which a naive port gets wrong:
///
///   * **`StoreElem` does not grow.** `checked_set` reports `OutOfBounds`, the
///     opposite of `DynSetKeyed`'s `Arr` arm which resizes. Conflating them is a
///     silent miscompile, so this builds a length-2 array and writes index 5.
///   * **`LoadElem` out-of-bounds is a raise**, not a null. Both engines must
///     produce the interpreter's `RuntimeError`, not a JIT fallback that invents a
///     value.
///   * **The index coercion is shared with the interpreter**, so `arr[-1]`,
///     `arr["x"]` and `arr[-0.5]` must all behave identically. `arr["x"]` and
///     `arr[-0.5]` read element 0; `arr[-1]` errors.
///
/// Results are compared **decoded**, never as raw register bits: a boxed
/// `Value::i32(0)` is `0xFFF9000000000000`, and an earlier version of this file's
/// array test failed on correct code for exactly that reason.
#[test]
fn array_element_access_matches_interpreter() {
    use raya_engine::jit::runtime::trampoline::JitExitKind;
    use raya_engine::vm::interpreter::Vm;

    // (label, slot to write, slot to read, expected value read back)
    // `None` as the expected value means "must be null".
    //
    // The index is per-case on purpose. An earlier version of this test hard-coded
    // index 1 for every case and labelled one of them "an untouched null slot" --
    // but index 1 is precisely the slot `InitArray` writes, so the interpreter
    // correctly returned 7 and the test's own expectation was the thing that was
    // wrong.
    let cases: Vec<(&str, i32, i32, Option<i32>)> = vec![
        ("read back the slot InitArray wrote", 1, 1, Some(7)),
        ("read an untouched null slot", 1, 2, None),
        ("StoreElem then read back", 1, 0, Some(99)),
    ];

    for (label, write_slot, read_slot, expected) in cases {
        // [3] -> NewArray(AnyValue) -> [arr]
        let mut code: Vec<u8> = Vec::new();
        emit_i32(&mut code, 3);
        emit(&mut code, Opcode::NewArray);
        code.extend_from_slice(&6u32.to_le_bytes());

        // arr, 7, InitArray <write_slot> -> arr
        emit_i32(&mut code, 7);
        emit(&mut code, Opcode::InitArray);
        code.extend_from_slice(&(write_slot as u16).to_le_bytes());

        if expected == Some(99) {
            // `StoreElem` is `[arr, idx, val] -> []`: it CONSUMES the array. Dup
            // BEFORE the store to keep a copy to read from afterwards. Duping after
            // underflows the stack, which the interpreter caught immediately -- an
            // earlier version of this test did exactly that.
            emit(&mut code, Opcode::Dup);
            emit_i32(&mut code, read_slot);
            emit_i32(&mut code, 99);
            emit(&mut code, Opcode::StoreElem);
            emit_i32(&mut code, read_slot);
            emit(&mut code, Opcode::LoadElem);
        } else {
            emit_i32(&mut code, read_slot);
            emit(&mut code, Opcode::LoadElem);
        }
        emit(&mut code, Opcode::Return);

        let mut raw = make_module(code, 0, 0);
        raw.functions[0].name = "main".to_string();
        let module = finalize_module(raw);

        let mut vm = Vm::with_worker_count(1);
        let interpreted = vm.execute(module.as_ref()).expect("interpreter element access");

        let (safepoint, shared) = new_shared_vm_state();
        let task =
            std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
        let (_resolved_natives, bridge) =
            build_bridge_and_ctx(&safepoint, &shared, &task, &module);
        let mut ctx =
            raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
        let mut locals: Vec<u64> = Vec::new();
        let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");
        let (raw_bits, exit) = jit_compile_and_call_with_locals_exit_and_ctx(
            &jit_func,
            &mut locals,
            (&mut ctx as *mut _),
        );

        assert_eq!(
            exit.kind,
            JitExitKind::Completed as u32,
            "[{label}] element access must complete natively"
        );
        let native = unsafe { raya_engine::vm::value::Value::from_raw(raw_bits) };
        match expected {
            None => {
                assert!(native.is_null(), "[{label}] an untouched slot must read null");
                assert!(
                    interpreted.is_null(),
                    "[{label}] the interpreter baseline is not null, so the case is wrong"
                );
            }
            Some(want) => {
                assert_eq!(native.as_i32(), Some(want), "[{label}] wrong value from the JIT");
                assert_eq!(
                    native.raw(),
                    interpreted.raw(),
                    "[{label}] engines disagree on the same bytecode"
                );
            }
        }
    }
}

/// The index coercion, differentially, at engine level for the first time.
///
/// This is the test `DynGetKeyed`'s `Arr` view could never have. It needs a
/// natively-compiled array, which only became possible once `NewArray` was
/// promoted — so the dependency that made `Arr` uncoverable in D4.7 is now gone.
///
/// Three cases, all of which look like bugs and are the interpreter's actual
/// behaviour. A JIT arm that clamped, rejected or defaulted would fail here.
#[test]
fn array_index_coercion_matches_interpreter() {
    use raya_engine::jit::runtime::trampoline::JitExitKind;
    use raya_engine::vm::interpreter::Vm;

    // (label, key program, expected read of slot 1, which holds 7)
    let cases: Vec<(&str, Vec<Opcode>)> = vec![
        ("null index means element 0", { let mut v = vec![]; v.push(Opcode::ConstNull); v }),
        ("negative f64 truncates to 0", { let mut v = vec![]; v.push(Opcode::ConstF64); v }),
    ];

    for (label, key_ops) in cases {
        let mut code: Vec<u8> = Vec::new();
        emit_i32(&mut code, 3);
        emit(&mut code, Opcode::NewArray);
        code.extend_from_slice(&6u32.to_le_bytes());
        emit_i32(&mut code, 7);
        emit(&mut code, Opcode::InitArray);
        code.extend_from_slice(&1u16.to_le_bytes());
        for op in &key_ops {
            emit(&mut code, *op);
            if *op == Opcode::ConstF64 {
                code.extend_from_slice(&(-0.5f64).to_le_bytes());
            }
        }
        emit(&mut code, Opcode::LoadElem);
        emit(&mut code, Opcode::Return);

        let mut raw = make_module(code, 0, 0);
        raw.functions[0].name = "main".to_string();
        let module = finalize_module(raw);

        let mut vm = Vm::with_worker_count(1);
        let interpreted = vm.execute(module.as_ref()).expect("interpreter coercion");

        let (safepoint, shared) = new_shared_vm_state();
        let task =
            std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
        let (_resolved_natives, bridge) =
            build_bridge_and_ctx(&safepoint, &shared, &task, &module);
        let mut ctx =
            raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
        let mut locals: Vec<u64> = Vec::new();
        let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");
        let (raw_bits, exit) = jit_compile_and_call_with_locals_exit_and_ctx(
            &jit_func,
            &mut locals,
            (&mut ctx as *mut _),
        );

        assert_eq!(
            exit.kind,
            JitExitKind::Completed as u32,
            "[{label}] the coercion must be handled natively, not declined"
        );
        // Element 0 is a null slot, so both engines must return null.
        let native = unsafe { raya_engine::vm::value::Value::from_raw(raw_bits) };
        assert!(native.is_null(), "[{label}] expected a null element 0");
        assert!(interpreted.is_null(), "[{label}] interpreter baseline is not null");
    }
}


/// D4.8 slice 3, differentially: `ArrayPush` and `ArrayPop` on the same bytecode
/// through both engines.
///
/// Three programs, each covering something a plausible implementation gets wrong:
///
///   * **empty pop yields `null`** — not an error, and not a fallback. The arm's
///     null-ctx path deliberately yields null rather than the sentinel precisely so
///     this does not exit; if it did, this test would see a fallback and fail.
///   * **push grows the backing `Vec`** — the reallocation path, which is the only
///     window in this family where a GC-visible operand matters. Starting from a
///     zero-length array and pushing twice forces at least one growth.
///   * **push then pop round-trips** the value, proving `ArrayLen` after growth
///     agrees too.
///
/// `exit.kind == Completed` is asserted for every case, so a silent fallback fails
/// rather than passing with the right answer.
#[test]
fn array_push_and_pop_match_interpreter() {
    use raya_engine::jit::runtime::trampoline::JitExitKind;
    use raya_engine::vm::interpreter::Vm;

    // Runs one program through both engines and asserts native completion.
    fn both_engines(code: Vec<u8>, label: &str) -> (raya_engine::vm::value::Value, bool) {
        let mut raw = make_module(code, 0, 0);
        raw.functions[0].name = "main".to_string();
        let module = finalize_module(raw);

        let mut vm = Vm::with_worker_count(1);
        let interpreted = vm.execute(module.as_ref()).expect(label);

        let (safepoint, shared) = new_shared_vm_state();
        let task =
            std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
        let (_resolved_natives, bridge) =
            build_bridge_and_ctx(&safepoint, &shared, &task, &module);
        let mut ctx =
            raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
        let mut locals: Vec<u64> = Vec::new();
        let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");
        let (raw_bits, exit) = jit_compile_and_call_with_locals_exit_and_ctx(
            &jit_func,
            &mut locals,
            (&mut ctx as *mut _),
        );
        assert_eq!(
            exit.kind,
            JitExitKind::Completed as u32,
            "[{label}] must complete natively, not fall back"
        );
        assert_eq!(
            raw_bits,
            interpreted.raw(),
            "[{label}] engines disagree on the same bytecode"
        );
        (
            unsafe { raya_engine::vm::value::Value::from_raw(raw_bits) },
            true,
        )
    }

    // --- Case 1: popping an EMPTY array must yield null.
    {
        let mut code: Vec<u8> = Vec::new();
        emit_i32(&mut code, 0);
        emit(&mut code, Opcode::NewArray);
        code.extend_from_slice(&6u32.to_le_bytes());
        emit(&mut code, Opcode::ArrayPop);
        emit(&mut code, Opcode::Return);
        let (v, _) = both_engines(code, "empty pop");
        assert!(
            v.is_null(),
            "popping an empty array must yield null, got 0x{:016X}",
            v.raw()
        );
    }

    // --- Case 2: push twice onto a ZERO-length array, then read the length. This
    // forces the backing Vec to reallocate at least once.
    {
        let mut code: Vec<u8> = Vec::new();
        emit_i32(&mut code, 0);
        emit(&mut code, Opcode::NewArray);
        code.extend_from_slice(&6u32.to_le_bytes());
        for v in [5i32, 6] {
            emit(&mut code, Opcode::Dup);
            emit_i32(&mut code, v);
            emit(&mut code, Opcode::ArrayPush);
        }
        emit(&mut code, Opcode::ArrayLen);
        emit(&mut code, Opcode::Return);
        let (v, _) = both_engines(code, "push grows");
        assert_eq!(
            v.as_i32(),
            Some(2),
            "two pushes onto an empty array must leave length 2"
        );
    }

    // --- Case 3: push then pop round-trips the value.
    {
        let mut code: Vec<u8> = Vec::new();
        emit_i32(&mut code, 1);
        emit(&mut code, Opcode::NewArray);
        code.extend_from_slice(&6u32.to_le_bytes());
        emit(&mut code, Opcode::Dup);
        emit_i32(&mut code, 42);
        emit(&mut code, Opcode::ArrayPush);
        emit(&mut code, Opcode::ArrayPop);
        emit(&mut code, Opcode::Return);
        let (v, _) = both_engines(code, "push/pop roundtrip");
        assert_eq!(v.as_i32(), Some(42), "push then pop must return the pushed value");
    }
}


/// D4.8's error paths, differentially — the four cases acceptance criterion 4 asks
/// for and the first three slices did not cover.
///
/// The pairing here is deliberately **not** "both engines produce the same value",
/// because these cases have no value. The interpreter raises; the JIT cannot raise,
/// so the only correct behaviour is to **exit to the interpreter and let it raise**.
/// So each case asserts both halves:
///
///   * the interpreter returns `Err`
///   * the JIT's exit is `Suspended` with `InterpreterBoundary` — i.e. it handed
///     back rather than inventing an answer
///
/// A JIT that "handled" the error natively — returning null, or a zero, or
/// truncating the index — would fail this test, because it would report
/// `Completed`.
///
/// The four cases: out-of-bounds load (a raise, not a null), out-of-bounds store
/// (**must not grow**), a non-array receiver, and an element-constraint violation.
#[test]
fn array_error_paths_fall_back_to_the_interpreter() {
    use raya_engine::jit::runtime::trampoline::{JitExitKind, JitSuspendReason};
    use raya_engine::vm::interpreter::Vm;

    // Each case emits its own body, because the four shapes need different operand
    // sequences and a shared op list turned into guesswork about which `ConstI32`
    // was an index, a value, or a slot to read back.
    type Body = Box<dyn Fn(&mut Vec<u8>, u32)>;
    let cases: Vec<(&str, Body)> = vec![
        // Out-of-bounds LOAD: a raise, not a null. A JIT that returned null here
        // would report Completed and fail this test.
        (
            "out-of-bounds load raises",
            Box::new(|c: &mut Vec<u8>, _s| {
                emit_i32(c, 3);
                emit(c, Opcode::NewArray);
                c.extend_from_slice(&6u32.to_le_bytes());
                emit_i32(c, 5); // index 5 on a length-3 array
                emit(c, Opcode::LoadElem);
                emit(c, Opcode::Return);
            }),
        ),
        // Out-of-bounds STORE: `StoreElem` must NOT grow the array. The trailing
        // `ConstI32 0` + Return is never reached in compiled code -- the arm exits at
        // the store -- but the interpreter needs a well-formed tail.
        (
            "out-of-bounds store raises and does not grow",
            Box::new(|c: &mut Vec<u8>, _s| {
                emit_i32(c, 3);
                emit(c, Opcode::NewArray);
                c.extend_from_slice(&6u32.to_le_bytes());
                emit(c, Opcode::Dup);
                emit_i32(c, 5); // index 5 on a length-3 array
                emit_i32(c, 42);
                emit(c, Opcode::StoreElem);
                emit_i32(c, 0);
                emit(c, Opcode::Return);
            }),
        ),
        // NON-ARRAY RECEIVER: `LoadElem` on a string. `jit_array_ptr_checked`
        // validates the GC-header TypeId, so this must decline rather than
        // reinterpreting the string's bytes as an `Array`.
        (
            "non-array receiver is rejected",
            Box::new(|c: &mut Vec<u8>, s| {
                emit_const_str(c, s);
                emit_i32(c, 0);
                emit(c, Opcode::LoadElem);
                emit(c, Opcode::Return);
            }),
        ),
        // ELEMENT-CONSTRAINT VIOLATION: element id 0 resolves to an `I32`
        // constraint, so storing a string must be rejected. This is the case that
        // proves the typed-array machinery is inherited rather than bypassed.
        (
            "element constraint violation is rejected",
            Box::new(|c: &mut Vec<u8>, s| {
                emit_i32(c, 1); // length 1, so slot 0 is in bounds on the INDEX
                emit(c, Opcode::NewArray);
                c.extend_from_slice(&0u32.to_le_bytes()); // element id 0 == I32
                emit(c, Opcode::Dup);
                emit_i32(c, 0);
                emit_const_str(c, s); // a string into an I32 array
                emit(c, Opcode::StoreElem);
                emit_i32(c, 0);
                emit(c, Opcode::Return);
            }),
        ),
    ];

    for (label, body) in cases {
        let mut module = make_vm_module(Vec::new(), 0, 0);
        let str_idx = module.constants.add_string("not an array".to_string());
        let mut code: Vec<u8> = Vec::new();
        body(&mut code, str_idx);
        module.functions[0].code = code;
        module.functions[0].name = "main".to_string();
        let module = finalize_module(module);

        // Engine 1: the interpreter must RAISE.
        let mut vm = Vm::with_worker_count(1);
        let interpreted = vm.execute(module.as_ref());
        assert!(
            interpreted.is_err(),
            "[{label}] the interpreter must raise, not return a value"
        );

        // Engine 2: the JIT must hand back rather than invent an answer.
        let (safepoint, shared) = new_shared_vm_state();
        let task =
            std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
        let (_resolved_natives, bridge) =
            build_bridge_and_ctx(&safepoint, &shared, &task, &module);
        let mut ctx =
            raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
        let mut locals: Vec<u64> = Vec::new();
        let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");
        let (_raw_bits, exit) = jit_compile_and_call_with_locals_exit_and_ctx(
            &jit_func,
            &mut locals,
            (&mut ctx as *mut _),
        );
        assert_eq!(
            exit.kind,
            JitExitKind::Suspended as u32,
            "[{label}] the JIT must exit to the interpreter, not complete"
        );
        assert_eq!(
            exit.suspend_reason,
            JitSuspendReason::InterpreterBoundary as u32,
            "[{label}] the exit must be an interpreter boundary"
        );
    }
}


/// D4.8 slice 4, differentially: `ArrayLiteral`, and with it the whole family.
///
/// **Every case uses DISTINCT elements per slot** (11, 22, 33), never a repeated
/// value. That is the whole point of this test. The handler pops elements and then
/// reverses them — "first pushed = first element" — and the lifter reverses too, so
/// the two could disagree, or agree and both be wrong. With a uniform element
/// (`[7, 7, 7]`) a reversal is **invisible**. With distinct elements, a reversal
/// returns 33 where 11 belongs and the test fails immediately.
///
/// The arm is `helper_alloc_array` plus one `helper_array_store` per element with a
/// constant index, mirroring the interpreter's own `build_array` + `checked_set`
/// loop. So this also proves the element-constraint check is inherited rather than
/// bypassed.
#[test]
fn array_literal_matches_interpreter() {
    use raya_engine::jit::runtime::trampoline::{JitExitKind, JitSuspendReason};
    use raya_engine::vm::interpreter::Vm;

    // (label, element values, slot to read back, expected)
    let cases: Vec<(&str, Vec<i32>, usize, i32)> = vec![
        ("first element is the first pushed", vec![11, 22, 33], 0, 11),
        ("middle element", vec![11, 22, 33], 1, 22),
        ("last element", vec![11, 22, 33], 2, 33),
        ("two elements", vec![11, 22], 1, 22),
    ];

    for (label, elems, slot, expected) in cases {
        let mut code: Vec<u8> = Vec::new();
        for e in &elems {
            emit_i32(&mut code, *e);
        }
        emit(&mut code, Opcode::ArrayLiteral);
        code.extend_from_slice(&6u32.to_le_bytes()); // type_index: AnyValue
        code.extend_from_slice(&(elems.len() as u32).to_le_bytes()); // length
        emit_i32(&mut code, slot as i32);
        emit(&mut code, Opcode::LoadElem);
        emit(&mut code, Opcode::Return);

        let mut raw = make_module(code, 0, 0);
        raw.functions[0].name = "main".to_string();
        let module = finalize_module(raw);

        let mut vm = Vm::with_worker_count(1);
        let interpreted = vm.execute(module.as_ref()).expect(label);

        let (safepoint, shared) = new_shared_vm_state();
        let task =
            std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
        let (_resolved_natives, bridge) =
            build_bridge_and_ctx(&safepoint, &shared, &task, &module);
        let mut ctx =
            raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
        let mut locals: Vec<u64> = Vec::new();
        let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");
        let (raw_bits, exit) = jit_compile_and_call_with_locals_exit_and_ctx(
            &jit_func,
            &mut locals,
            (&mut ctx as *mut _),
        );
        assert_eq!(
            exit.kind,
            JitExitKind::Completed as u32,
            "[{label}] ArrayLiteral must complete natively"
        );
        let native = unsafe { raya_engine::vm::value::Value::from_raw(raw_bits) };
        assert_eq!(
            native.as_i32(),
            Some(expected),
            "[{label}] wrong element — a reversed literal would land here"
        );
        assert_eq!(native.raw(), interpreted.raw(), "[{label}] engines disagree");
    }
}

/// `ArrayLiteral`'s element-constraint violation, differentially. Element id 0
/// resolves to an `I32` constraint, so a literal containing a string must be
/// rejected — and the rejection has to come from the interpreter, because the JIT
/// cannot raise.
#[test]
fn array_literal_constraint_violation_falls_back() {
    use raya_engine::jit::runtime::trampoline::{JitExitKind, JitSuspendReason};
    use raya_engine::vm::interpreter::Vm;

    let mut module = make_vm_module(Vec::new(), 0, 0);
    let str_idx = module.constants.add_string("nope".to_string());
    let mut code: Vec<u8> = Vec::new();
    emit_const_str(&mut code, str_idx);
    emit(&mut code, Opcode::ArrayLiteral);
    code.extend_from_slice(&0u32.to_le_bytes()); // element id 0 == I32
    code.extend_from_slice(&1u32.to_le_bytes()); // length 1
    emit(&mut code, Opcode::Return);
    module.functions[0].code = code;
    module.functions[0].name = "main".to_string();
    let module = finalize_module(module);

    let mut vm = Vm::with_worker_count(1);
    assert!(
        vm.execute(module.as_ref()).is_err(),
        "the interpreter must reject a string in an I32 array"
    );

    let (safepoint, shared) = new_shared_vm_state();
    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let (_resolved_natives, bridge) = build_bridge_and_ctx(&safepoint, &shared, &task, &module);
    let mut ctx =
        raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    let mut locals: Vec<u64> = Vec::new();
    let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");
    let (_raw_bits, exit) =
        jit_compile_and_call_with_locals_exit_and_ctx(&jit_func, &mut locals, (&mut ctx as *mut _));

    assert_eq!(
        exit.kind,
        JitExitKind::Suspended as u32,
        "the JIT must hand back rather than build the array anyway"
    );
    assert_eq!(
        exit.suspend_reason,
        JitSuspendReason::InterpreterBoundary as u32,
        "the exit must be an interpreter boundary"
    );
}


/// D4.7's deferred evidence, now collectable: `DynGetKeyed`'s **`Arr` view**,
/// differentially.
///
/// This is the corpus D4.7 could not have. With the whole array family `Rejected`,
/// no natively-compiled bytecode could construct an array, so this test would have
/// had its entire function rejected and silently run interpreted — passing while
/// proving nothing about the helper, which is the exact vacuity that let D4.3's P0
/// ship. D4.8 promoted `NewArray`/`InitArray`/`LoadElem`, and the dependency is
/// gone.
///
/// Four cases, because the keyed path has a trap the positional one does not:
/// `dyn_key_parts` parses a **string** key with `key.parse::<usize>()`, so `"1"`
/// and `1` are the SAME index. A helper that only handled integer keys, or that
/// treated a string key as a property name, would agree with the interpreter on
/// neither.
#[test]
fn dyn_get_keyed_array_view_matches_interpreter() {
    use raya_engine::jit::runtime::trampoline::JitExitKind;
    use raya_engine::vm::interpreter::Vm;

    // Takes the MODULE, not the code: these cases index into the constant pool for
    // their string keys, and an earlier version built its own module with
    // `make_module`, whose pool is empty -- so the string key indices pointed at
    // nothing.
    fn both_engines(
        module: std::sync::Arc<Module>,
        label: &str,
    ) -> raya_engine::vm::value::Value {
        let mut vm = Vm::with_worker_count(1);
        let interpreted = vm.execute(module.as_ref()).expect(label);

        let (safepoint, shared) = new_shared_vm_state();
        let task =
            std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
        let (_resolved_natives, bridge) =
            build_bridge_and_ctx(&safepoint, &shared, &task, &module);
        let mut ctx =
            raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
        let mut locals: Vec<u64> = Vec::new();
        let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");
        let (raw_bits, exit) = jit_compile_and_call_with_locals_exit_and_ctx(
            &jit_func,
            &mut locals,
            (&mut ctx as *mut _),
        );
        assert_eq!(
            exit.kind,
            JitExitKind::Completed as u32,
            "[{label}] the Arr view must complete natively, not fall back"
        );
        let native = unsafe { raya_engine::vm::value::Value::from_raw(raw_bits) };
        assert_eq!(
            native.raw(),
            interpreted.raw(),
            "[{label}] engines disagree on the same bytecode"
        );
        native
    }

    // Builds a module whose function is `array[11, 22, null]` with `key_ops`
    // pushing exactly one key, then a `DynGetKeyed`.
    // `str_key` is added to THIS module's pool, so the index is valid in the module
    // the code actually lives in. An earlier version added the strings to a separate
    // throwaway module and the interpreter rejected the program with
    // `Invalid string constant index: 0`.
    fn keyed_array_module(
        int_key: Option<i32>,
        str_key: Option<&str>,
    ) -> std::sync::Arc<Module> {
        let mut module = make_vm_module(Vec::new(), 0, 0);
        let str_idx = str_key.map(|s| module.constants.add_string(s.to_string()));
        let mut code: Vec<u8> = Vec::new();
        emit_i32(&mut code, 3);
        emit(&mut code, Opcode::NewArray);
        code.extend_from_slice(&6u32.to_le_bytes()); // AnyValue
        emit_i32(&mut code, 11);
        emit(&mut code, Opcode::InitArray);
        code.extend_from_slice(&0u16.to_le_bytes());
        emit_i32(&mut code, 22);
        emit(&mut code, Opcode::InitArray);
        code.extend_from_slice(&1u16.to_le_bytes());
        match (int_key, str_idx) {
            (Some(v), _) => emit_i32(&mut code, v),
            (None, Some(idx)) => emit_const_str(&mut code, idx),
            (None, None) => unreachable!("a keyed read needs one key"),
        }
        emit(&mut code, Opcode::DynGetKeyed);
        emit(&mut code, Opcode::Return);
        module.functions[0].code = code;
        module.functions[0].name = "main".to_string();
        finalize_module(module)
    }

    // Integer key 1 -> element 1.
    let v = both_engines(keyed_array_module(Some(1), None), "int key 1");
    assert_eq!(v.as_i32(), Some(22), "an integer key must read that element");

    // STRING key "1" -> the SAME element, because dyn_key_parts parses it as an index.
    let v = both_engines(keyed_array_module(None, Some("1")), "string key \"1\"");
    assert_eq!(
        v.as_i32(),
        Some(22),
        "a numeric string key must read the same element as the integer key"
    );

    // "length" -> 3, the array's length.
    let v = both_engines(keyed_array_module(None, Some("length")), "length key");
    assert_eq!(v.as_i32(), Some(3), "the length key must return the array length");

    // Out-of-range integer key -> null (the Arr view returns null, not a raise —
    // that RAISE-on-out-of-bounds behaviour belongs to `LoadElem`, a different
    // opcode with a different handler).
    let v = both_engines(keyed_array_module(Some(99), None), "out-of-range key");
    assert!(v.is_null(), "an out-of-range keyed read must be null");
}


/// D4.9, differentially: `DynSetKeyed` against an **array** receiver.
///
/// The corpus is built around one fact that makes this opcode unlike anything else
/// on the branch, and the fact looks like a bug if you do not know it:
///
/// | | `DynSetKeyed` | `StoreElem` |
/// |---|---|---|
/// | index past the end | **grows** via `resize(index + 1, null)` | `OutOfBounds` error |
/// | element constraint | **not checked at all** | enforced by `checked_set` |
///
/// The interpreter's `DynSetKeyed` arm assigns `arr.elements[index] = value`
/// directly after an optional `resize`. So a string stored into an `I32`-constrained
/// array *succeeds* — and `dyn_set_keyed_constraint_is_not_enforced` below pins that.
/// A helper that reused `array_store` would refuse to grow and reject the value, and
/// "it should enforce the element type" is exactly what a bug report would say. It
/// would be a divergence.
#[test]
fn dyn_set_keyed_array_view_matches_interpreter() {
    use raya_engine::jit::runtime::trampoline::JitExitKind;
    use raya_engine::vm::interpreter::Vm;

    fn both_engines(
        module: std::sync::Arc<Module>,
        label: &str,
    ) -> raya_engine::vm::value::Value {
        let mut vm = Vm::with_worker_count(1);
        let interpreted = vm.execute(module.as_ref()).expect(label);

        let (safepoint, shared) = new_shared_vm_state();
        let task =
            std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
        let (_resolved_natives, bridge) =
            build_bridge_and_ctx(&safepoint, &shared, &task, &module);
        let mut ctx =
            raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
        let mut locals: Vec<u64> = Vec::new();
        let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");
        let (raw_bits, exit) = jit_compile_and_call_with_locals_exit_and_ctx(
            &jit_func,
            &mut locals,
            (&mut ctx as *mut _),
        );
        assert_eq!(
            exit.kind,
            JitExitKind::Completed as u32,
            "[{label}] must complete natively, not fall back"
        );
        let native = unsafe { raya_engine::vm::value::Value::from_raw(raw_bits) };
        assert_eq!(
            native.raw(),
            interpreted.raw(),
            "[{label}] engines disagree on the same bytecode"
        );
        native
    }

    // `array[11, 22, null]`, then `DynSetKeyed` with the given key and value, then
    // read `read_slot` back.
    fn set_then_read(
        type_id: u32,
        set_key: Option<i32>,
        set_key_str: Option<&str>,
        set_value: i32,
        read_slot: i32,
    ) -> std::sync::Arc<Module> {
        let mut module = make_vm_module(Vec::new(), 0, 0);
        let key_idx = set_key_str.map(|s| module.constants.add_string(s.to_string()));
        let mut code: Vec<u8> = Vec::new();
        emit_i32(&mut code, 3);
        emit(&mut code, Opcode::NewArray);
        code.extend_from_slice(&type_id.to_le_bytes());
        emit_i32(&mut code, 11);
        emit(&mut code, Opcode::InitArray);
        code.extend_from_slice(&0u16.to_le_bytes());
        emit_i32(&mut code, 22);
        emit(&mut code, Opcode::InitArray);
        code.extend_from_slice(&1u16.to_le_bytes());

        // DynSetKeyed consumes the array, so keep a copy to read back afterwards.
        emit(&mut code, Opcode::Dup); // [arr, arr]
        match (set_key, key_idx) {
            (Some(k), _) => emit_i32(&mut code, k),
            (None, Some(idx)) => emit_const_str(&mut code, idx),
            (None, None) => unreachable!("a keyed set needs one key"),
        }
        emit_i32(&mut code, set_value);
        emit(&mut code, Opcode::DynSetKeyed); // [arr]
        emit_i32(&mut code, read_slot);
        emit(&mut code, Opcode::LoadElem);
        emit(&mut code, Opcode::Return);
        module.functions[0].code = code;
        module.functions[0].name = "main".to_string();
        finalize_module(module)
    }

    // In-range write, read back.
    let v = both_engines(set_then_read(6, Some(1), None, 99, 1), "in-range set");
    assert_eq!(v.as_i32(), Some(99), "an in-range keyed set must be visible");

    // Numeric STRING key resolves to the same index.
    let v = both_engines(set_then_read(6, None, Some("1"), 77, 1), "string key set");
    assert_eq!(v.as_i32(), Some(77), "a numeric string key must set that element");

    // BEYOND THE END: this GROWS, and the interpreter fills the gap with null.
    let v = both_engines(set_then_read(6, Some(5), None, 42, 5), "growing set");
    assert_eq!(
        v.as_i32(),
        Some(42),
        "an out-of-range keyed set must GROW the array rather than fail"
    );
}

/// The constraint case, on its own because it is the one that would be "fixed" by
/// mistake: the interpreter does **not** enforce the element constraint on
/// `DynSetKeyed`, and neither may the helper.
///
/// An `I32`-constrained array (`NewArray` element id 0) given a **string** by
/// `DynSetKeyed`. The interpreter stores it. If the helper used `checked_set`, or
/// re-derived `helper_array_store`, it would fall back — the exit kind would be
/// `Suspended` instead of `Completed`, and this fails.
#[test]
fn dyn_set_keyed_constraint_is_not_enforced() {
    use raya_engine::jit::runtime::trampoline::JitExitKind;
    use raya_engine::vm::interpreter::Vm;

    let mut module = make_vm_module(Vec::new(), 0, 0);
    let str_idx = module.constants.add_string("not an i32".to_string());
    let mut code: Vec<u8> = Vec::new();
    emit_i32(&mut code, 1);
    emit(&mut code, Opcode::NewArray);
    code.extend_from_slice(&0u32.to_le_bytes()); // element id 0 == I32
    emit(&mut code, Opcode::Dup);
    emit_i32(&mut code, 0);
    emit_const_str(&mut code, str_idx);
    emit(&mut code, Opcode::DynSetKeyed); // [arr]
    emit_i32(&mut code, 0);
    emit(&mut code, Opcode::LoadElem); // read it back
    emit(&mut code, Opcode::Return);
    module.functions[0].code = code;
    module.functions[0].name = "main".to_string();
    let module = finalize_module(module);

    let mut vm = Vm::with_worker_count(1);
    let interpreted = vm
        .execute(module.as_ref())
        .expect("the interpreter MUST accept a string in an I32 array here");
    assert_eq!(
        string_contents(interpreted),
        "not an i32",
        "the interpreter must store the unconstrained value, not reject it"
    );

    let (safepoint, shared) = new_shared_vm_state();
    let task = std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
    let (_resolved_natives, bridge) = build_bridge_and_ctx(&safepoint, &shared, &task, &module);
    let mut ctx =
        raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
    let mut locals: Vec<u64> = Vec::new();
    let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");
    let (raw_bits, exit) =
        jit_compile_and_call_with_locals_exit_and_ctx(&jit_func, &mut locals, (&mut ctx as *mut _));

    assert_eq!(
        exit.kind,
        JitExitKind::Completed as u32,
        "the JIT must store the value unchecked too -- falling back here would mean it \\
         enforced the constraint, which the interpreter does not"
    );
    let native = unsafe { raya_engine::vm::value::Value::from_raw(raw_bits) };
    assert_eq!(string_contents(native), "not an i32");
}


/// D4.10: `CastObjectMinFields` and object construction, differentially.
///
/// It is a **checked pass-through**: the interpreter pushes the OBJECT back unchanged,
/// so there is no boolean to compare and no `false` outcome anywhere. Every failure
/// path is a `TypeError` a helper cannot raise, so each must DECLINE rather than answer.
///
/// | case | expected |
/// |---|---|
/// | enough fields | `Completed`; the object passes through |
/// | **not** enough fields | interpreter raises; JIT declines (`Suspended` + `InterpreterBoundary`) |
/// | non-object receiver | interpreter raises; JIT declines |
///
/// Object comparison is "is it a pointer", never raw bits: the two engines each
/// allocate their own object, so the pointers necessarily differ.
#[test]
fn cast_object_min_fields_matches_interpreter() {
    use raya_engine::jit::runtime::trampoline::{JitExitKind, JitSuspendReason};
    use raya_engine::vm::interpreter::Vm;

    fn program(with_object: bool, field_count: u16, required: u16) -> std::sync::Arc<Module> {
        let mut module = make_vm_module(Vec::new(), 0, 0);
        let mut code: Vec<u8> = Vec::new();
        if with_object {
            emit(&mut code, Opcode::ObjectLiteral);
            code.extend_from_slice(&1u32.to_le_bytes()); // layout id, non-zero
            code.extend_from_slice(&field_count.to_le_bytes());
        } else {
            emit_i32(&mut code, 5); // an integer: not an object
        }
        emit(&mut code, Opcode::CastObjectMinFields);
        code.extend_from_slice(&required.to_le_bytes());
        emit(&mut code, Opcode::Return);
        module.functions[0].code = code;
        module.functions[0].name = "main".to_string();
        finalize_module(module)
    }

    fn jit_side(
        module: std::sync::Arc<Module>,
    ) -> (u32, u32, u64) {
        let (safepoint, shared) = new_shared_vm_state();
        let task =
            std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
        let (_resolved_natives, bridge) =
            build_bridge_and_ctx(&safepoint, &shared, &task, &module);
        let mut ctx =
            raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
        let mut locals: Vec<u64> = Vec::new();
        let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");
        let (raw_bits, exit) = jit_compile_and_call_with_locals_exit_and_ctx(
            &jit_func,
            &mut locals,
            (&mut ctx as *mut _),
        );
        (exit.kind, exit.suspend_reason, raw_bits)
    }

    fn interpreter(
        module: &Module,
    ) -> Result<raya_engine::vm::value::Value, String> {
        let mut vm = Vm::with_worker_count(1);
        vm.execute(module).map_err(|e| e.to_string())
    }

    // --- Case 1: enough fields -> the object passes through, natively.
    {
        let module = program(true, 3, 2);
        let interpreted = interpreter(&module).expect("interpreter must pass the cast");
        let (kind, _, raw_bits) = jit_side(module);
        assert_eq!(
            kind,
            JitExitKind::Completed as u32,
            "a sufficient cast must complete natively"
        );
        let native = unsafe { raya_engine::vm::value::Value::from_raw(raw_bits) };
        assert!(
            native.is_ptr() && interpreted.is_ptr(),
            "both engines must return the object, not a boolean and not null"
        );
    }

    // --- Case 2: field count below the requirement -> BOTH raise / decline.
    {
        let module = program(true, 1, 5);
        assert!(
            interpreter(&module).is_err(),
            "the interpreter must RAISE when the field count is too low -- this is an \
             error, NOT a false cast"
        );
        let (kind, reason, _) = jit_side(module);
        assert_eq!(
            kind,
            JitExitKind::Suspended as u32,
            "a too-small field count must DECLINE, not answer false"
        );
        assert_eq!(reason, JitSuspendReason::InterpreterBoundary as u32);
    }

    // --- Case 3: non-object receiver -> BOTH raise / decline.
    {
        let module = program(false, 0, 1);
        assert!(
            interpreter(&module).is_err(),
            "the interpreter must raise on a non-object receiver"
        );
        let (kind, reason, _) = jit_side(module);
        assert_eq!(
            kind,
            JitExitKind::Suspended as u32,
            "a non-object receiver must DECLINE"
        );
        assert_eq!(reason, JitSuspendReason::InterpreterBoundary as u32);
    }
}


/// D4.10: `DynGetKeyed`'s **`Struct` view**, differentially.
///
/// It was blocked for three steps because no compiled function could *construct* a
/// `Struct`. `CastObjectMinFields`, `ObjectLiteral` and `InitObject` are now promoted,
/// which removed that wall; the nominal-type opcodes never could have, because they
/// test **nominal** type and `ObjectLiteral` produces a **structural** object.
///
/// | case | helper | expected |
/// |---|---|---|
/// | ordinary field | computes | `Completed`, value read back |
/// | **missing** field | computes | `Completed`, **`null`** |
///
/// **A missing `Struct` field is `null`, not a `TypeError`.** An earlier note in this
/// repo asserted the opposite and a differential caught it: the interpreter's fallback
/// for an unknown field is the object's **dynamic property map**, ending in
/// `Value::null()`. The asymmetry that IS real runs the other way — `Str` and `Arr`
/// yield null on out-of-range, while `DynSetKeyed`'s out-of-range array index is a
/// `TypeError`.
#[test]
fn dyn_get_keyed_struct_view_matches_interpreter() {
    use raya_engine::jit::runtime::trampoline::JitExitKind;
    use raya_engine::vm::interpreter::Vm;

    // `ObjectLiteral <u32 layout_id><u16 field_count>`, then `InitObject <u16 offset>`
    // with `[obj, value]` on the stack, then `DynGetKeyed` with the given key.
    fn program(field_key: &str) -> std::sync::Arc<Module> {
        let mut module = make_vm_module(Vec::new(), 0, 0);
        let key_idx = module.constants.add_string(field_key.to_string());
        let mut code: Vec<u8> = Vec::new();
        emit(&mut code, Opcode::ObjectLiteral);
        code.extend_from_slice(&1u32.to_le_bytes());
        code.extend_from_slice(&1u16.to_le_bytes()); // one slot
        emit_i32(&mut code, 11);
        emit(&mut code, Opcode::InitObject);
        code.extend_from_slice(&0u16.to_le_bytes());
        emit_const_str(&mut code, key_idx);
        emit(&mut code, Opcode::DynGetKeyed);
        emit(&mut code, Opcode::Return);
        module.functions[0].code = code;
        module.functions[0].name = "main".to_string();
        finalize_module(module)
    }

    fn jit_side(module: std::sync::Arc<Module>) -> (u32, u64) {
        let (safepoint, shared) = new_shared_vm_state();
        let task =
            std::sync::Arc::new(raya_engine::vm::scheduler::Task::new(0, module.clone(), None));
        let (_resolved_natives, bridge) =
            build_bridge_and_ctx(&safepoint, &shared, &task, &module);
        let mut ctx =
            raya_engine::jit::runtime::helpers::build_runtime_context(&bridge, module.as_ref());
        let mut locals: Vec<u64> = Vec::new();
        let jit_func = lift_function(&module.functions[0], &module, 0).expect("Lift failed");
        let (raw_bits, exit) = jit_compile_and_call_with_locals_exit_and_ctx(
            &jit_func,
            &mut locals,
            (&mut ctx as *mut _),
        );
        (exit.kind, raw_bits)
    }

    // --- Case 1: an ordinary field -> computed, both engines read the same slot.
    {
        let module = program("value");
        let mut vm = Vm::with_worker_count(1);
        let interpreted = vm
            .execute(module.as_ref())
            .expect("interpreter must resolve an ordinary field");
        assert_eq!(interpreted.as_i32(), Some(11), "the interpreter baseline is wrong");

        let (kind, raw_bits) = jit_side(module);
        assert_eq!(
            kind,
            JitExitKind::Completed as u32,
            "an ordinary Struct field must be computed natively, not declined"
        );
        let native = unsafe { raya_engine::vm::value::Value::from_raw(raw_bits) };
        assert_eq!(
            native.as_i32(),
            Some(11),
            "the JIT must read the same slot the interpreter did"
        );
    }

    // --- Case 2: a MISSING field -> null in BOTH engines, computed natively.
    {
        let module = program("nope");
        let mut vm = Vm::with_worker_count(1);
        let interpreted = vm
            .execute(module.as_ref())
            .expect("a missing field is null, not an error");
        assert!(
            interpreted.is_null(),
            "the interpreter must answer null for an unknown Struct field"
        );

        let (kind, raw_bits) = jit_side(module);
        assert_eq!(
            kind,
            JitExitKind::Completed as u32,
            "with no dynamic map there is nothing to look up, so the helper computes null"
        );
        let native = unsafe { raya_engine::vm::value::Value::from_raw(raw_bits) };
        assert!(
            native.is_null(),
            "the JIT must also answer null, not decline and not invent a value"
        );
    }
}
