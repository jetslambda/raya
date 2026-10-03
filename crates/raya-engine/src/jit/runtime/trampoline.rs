//! JIT trampolines and calling convention
//!
//! Defines the C-ABI interface between JIT-compiled code and the VM runtime.
//! JIT code calls back into the runtime through function pointers in
//! `RuntimeHelperTable` for GC, allocation, native calls, etc.

/// Entry point signature for JIT-compiled functions
///
/// JIT code receives arguments as NaN-boxed Value array, a locals buffer,
/// and a context pointer containing runtime helpers.
pub type JitEntryFn = unsafe extern "C" fn(
    args: *const u64, // NaN-boxed Value array
    arg_count: u32,
    locals: *mut u64, // pre-allocated locals
    local_count: u32,
    ctx: *mut RuntimeContext,
    exit_info: *mut JitExitInfo,
) -> u64; // returns NaN-boxed Value

/// Exit state for a JIT function invocation.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JitExitKind {
    Completed = 0,
    Suspended = 1,
    Deoptimized = 2,
    Failed = 3,
}

/// Suspension reasons written into `JitExitInfo.suspend_reason`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JitSuspendReason {
    None = 0,
    Preemption = 1,
    NativeCallBoundary = 2,
    InterpreterBoundary = 3,
}

/// Minimal native-frame snapshot to support resume/deopt plumbing.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct JitMachineFrameSnapshot {
    /// Native instruction pointer / continuation marker.
    pub resume_ip: u64,
    /// Native stack pointer captured at exit (if available).
    pub stack_ptr: u64,
    /// Native frame/base pointer captured at exit (if available).
    pub frame_ptr: u64,
}

/// Maximum operand materialization supported when handing control back to the interpreter.
pub const JIT_EXIT_MAX_NATIVE_ARGS: usize = 32;

/// Out-parameter written by JIT entry to describe exit behavior.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct JitExitInfo {
    /// How execution exited.
    pub kind: u32,
    /// Suspension reason discriminator (VM-specific; 0 = none).
    pub suspend_reason: u32,
    /// Bytecode offset for deopt/resume (if relevant).
    pub bytecode_offset: u32,
    /// Reserved for alignment/extension.
    pub _reserved: u32,
    /// Captured native frame metadata.
    pub frame: JitMachineFrameSnapshot,
    /// Materialized operand count for interpreter-boundary resume handoff.
    pub native_arg_count: u32,
    /// Reserved for alignment/extension.
    pub _native_reserved: u32,
    /// Materialized operands (NaN-boxed values) for interpreter resume.
    pub native_args: [u64; JIT_EXIT_MAX_NATIVE_ARGS],
}

impl Default for JitExitInfo {
    fn default() -> Self {
        Self {
            kind: JitExitKind::Completed as u32,
            suspend_reason: JitSuspendReason::None as u32,
            bytecode_offset: 0,
            _reserved: 0,
            frame: JitMachineFrameSnapshot::default(),
            native_arg_count: 0,
            _native_reserved: 0,
            native_args: [0; JIT_EXIT_MAX_NATIVE_ARGS],
        }
    }
}

/// Runtime context passed to JIT-compiled code
///
/// Contains opaque pointers to VM state and a table of helper function pointers
/// that JIT code can call for runtime services.
#[repr(C)]
pub struct RuntimeContext {
    /// Pointer to SharedVmState
    pub shared_state: *const (),
    /// Pointer to current Task
    pub current_task: *const (),
    /// Pointer to Module
    pub module: *const (),
    /// Table of runtime helper function pointers
    pub helpers: RuntimeHelperTable,
}

/// ABI layout constants consumed by Cranelift lowering. Keep these derived from
/// the Rust structs; never duplicate field offsets in native code.
// The array runtime helpers added for D4.2 are appended to the end of
// `RuntimeHelperTable`; existing field offsets are unchanged and the function
// calling convention is untouched, so this stays lockstep with the bytecode
// `CURRENT_ABI_VERSION`. Bumping it would falsely signal a calling-convention
// change and disable typed JIT entry (see the guard in interpreter/core.rs),
// so it is intentionally NOT bumped for a backward-compatible helper addition.
pub const RUNTIME_CONTEXT_ABI_VERSION: u16 = 3;
pub const RUNTIME_CONTEXT_MODULE_OFFSET: i32 =
    std::mem::offset_of!(RuntimeContext, module) as i32;
pub const RUNTIME_CONTEXT_HELPERS_OFFSET: i32 =
    std::mem::offset_of!(RuntimeContext, helpers) as i32;
pub const HELPER_SAFEPOINT_POLL_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET
        + std::mem::offset_of!(RuntimeHelperTable, safepoint_poll) as i32;
pub const HELPER_CHECK_PREEMPTION_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET
        + std::mem::offset_of!(RuntimeHelperTable, check_preemption) as i32;
pub const HELPER_NATIVE_CALL_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET
        + std::mem::offset_of!(RuntimeHelperTable, native_call_dispatch) as i32;
pub const HELPER_INTERPRETER_CALL_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET
        + std::mem::offset_of!(RuntimeHelperTable, interpreter_call) as i32;
pub const HELPER_THROW_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET + std::mem::offset_of!(RuntimeHelperTable, throw_exception) as i32;
pub const HELPER_DEOPT_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET + std::mem::offset_of!(RuntimeHelperTable, deoptimize) as i32;
pub const HELPER_STRING_CONCAT_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET + std::mem::offset_of!(RuntimeHelperTable, string_concat) as i32;
pub const HELPER_GENERIC_EQUALS_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET + std::mem::offset_of!(RuntimeHelperTable, generic_equals) as i32;
pub const HELPER_OBJECT_GET_FIELD_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET + std::mem::offset_of!(RuntimeHelperTable, object_get_field) as i32;
pub const HELPER_OBJECT_SET_FIELD_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET + std::mem::offset_of!(RuntimeHelperTable, object_set_field) as i32;
pub const HELPER_OBJECT_SHAPE_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET + std::mem::offset_of!(RuntimeHelperTable, object_implements_shape) as i32;
pub const HELPER_OBJECT_NOMINAL_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET + std::mem::offset_of!(RuntimeHelperTable, object_is_nominal) as i32;
pub const HELPER_OBJECT_GET_SHAPE_FIELD_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET + std::mem::offset_of!(RuntimeHelperTable, object_get_shape_field) as i32;
pub const HELPER_OBJECT_SET_SHAPE_FIELD_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET + std::mem::offset_of!(RuntimeHelperTable, object_set_shape_field) as i32;
pub const HELPER_STRING_LEN_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET + std::mem::offset_of!(RuntimeHelperTable, string_len) as i32;
pub const HELPER_STRING_COMPARE_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET + std::mem::offset_of!(RuntimeHelperTable, string_compare) as i32;
pub const HELPER_VALUE_TO_STRING_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET + std::mem::offset_of!(RuntimeHelperTable, value_to_string) as i32;
pub const HELPER_CONST_STRING_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET + std::mem::offset_of!(RuntimeHelperTable, const_string) as i32;
/// Byte offset of the array-load helper in [`RuntimeContext`].
pub const HELPER_ARRAY_LOAD_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET + std::mem::offset_of!(RuntimeHelperTable, array_load) as i32;
/// Byte offset of the array-store helper in [`RuntimeContext`].
pub const HELPER_ARRAY_STORE_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET + std::mem::offset_of!(RuntimeHelperTable, array_store) as i32;
/// Byte offset of the array-push helper in [`RuntimeContext`].
pub const HELPER_ARRAY_PUSH_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET + std::mem::offset_of!(RuntimeHelperTable, array_push) as i32;
/// Byte offset of the array-pop helper in [`RuntimeContext`].
pub const HELPER_ARRAY_POP_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET + std::mem::offset_of!(RuntimeHelperTable, array_pop) as i32;
/// Byte offset of the array-length helper in [`RuntimeContext`].
pub const HELPER_ARRAY_LEN_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET + std::mem::offset_of!(RuntimeHelperTable, array_len) as i32;

/// Byte offset of the RefCell-load helper in [`RuntimeContext`].
///
/// Appended after the array helpers: `RuntimeHelperTable` is layout-sensitive
/// because offsets are computed from it, so new fields go at the end.
pub const HELPER_REFCELL_LOAD_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET + std::mem::offset_of!(RuntimeHelperTable, refcell_load) as i32;
/// Byte offset of the RefCell-store helper in [`RuntimeContext`].
///
/// 1 = `JIT_STORE_SUCCESS`, 0 = `JIT_STORE_FALLBACK`.
pub const HELPER_REFCELL_STORE_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET + std::mem::offset_of!(RuntimeHelperTable, refcell_store) as i32;
/// Byte offset of the RefCell-allocate helper in [`RuntimeContext`].
///
/// Returns null when the root set is unavailable, matching the convention every
/// other allocating helper here follows.
pub const HELPER_REFCELL_NEW_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET + std::mem::offset_of!(RuntimeHelperTable, refcell_new) as i32;

/// Byte offset of the closure-capture-patch helper in [`RuntimeContext`].
///
/// 1 = `JIT_STORE_SUCCESS`, 0 = `JIT_STORE_FALLBACK`.
pub const HELPER_SET_CLOSURE_CAPTURE_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET
        + std::mem::offset_of!(RuntimeHelperTable, set_closure_capture) as i32;

/// Byte offset of the closure-allocate helper in [`RuntimeContext`].
///
/// Captures are passed as a pointer plus a count because a `Vec` cannot cross the
/// trampoline function-pointer ABI. Returns null when the root set is unavailable.
pub const HELPER_MAKE_CLOSURE_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET + std::mem::offset_of!(RuntimeHelperTable, make_closure) as i32;

/// Byte offset of the captured-value load helper in [`RuntimeContext`].
///
/// Returns the interpreter-fallback sentinel for a missing active closure or an
/// out-of-range capture index, since a leaf helper cannot raise those.
pub const HELPER_LOAD_CAPTURED_OFFSET: i32 =
    RUNTIME_CONTEXT_HELPERS_OFFSET
        + std::mem::offset_of!(RuntimeHelperTable, load_captured) as i32;

const _: () = assert!(std::mem::align_of::<RuntimeContext>() >= 8);
const _: () = assert!(std::mem::size_of::<RuntimeContext>() >= std::mem::size_of::<RuntimeHelperTable>());

/// C-ABI function pointer table for runtime helpers
///
/// JIT code calls these through the RuntimeContext to interact with the VM.
/// All functions take raw pointers and NaN-boxed u64 values.
#[repr(C)]
pub struct RuntimeHelperTable {
    /// Allocate a new nominal object: (local_nominal_type_index, module_ptr, shared_state) -> obj_ptr
    pub alloc_object: unsafe extern "C" fn(u32, *const (), *mut ()) -> *mut (),
    /// Allocate a new array: (element_type_index, capacity, module_ptr, shared_state) -> array_ptr
    pub alloc_array: unsafe extern "C" fn(u32, usize, *const (), *mut ()) -> *mut (),
    /// Allocate a new string: (data_ptr, len, shared_state) -> string_ptr
    pub alloc_string: unsafe extern "C" fn(*const u8, usize, *mut ()) -> *mut (),
    /// GC safepoint poll: (shared_state)
    pub safepoint_poll: unsafe extern "C" fn(*const ()),
    /// Check if current task should be preempted: (current_task) -> should_yield
    pub check_preemption: unsafe extern "C" fn(*const ()) -> bool,
    /// Dispatch a native call: (native_id, args_ptr, arg_count, shared_state) -> result
    pub native_call_dispatch: unsafe extern "C" fn(u16, *const u64, u8, *mut ()) -> u64,
    /// Execute a call-family opcode through the interpreter runtime:
    /// (opcode, operand_u64, operand_u32, receiver, args_ptr, arg_count, module_ptr, shared_state) -> result/sentinel
    pub interpreter_call:
        unsafe extern "C" fn(u8, u64, u32, u64, *const u64, u16, *const (), *mut ()) -> u64,
    /// Throw an exception: (exception_value, shared_state) -> !
    pub throw_exception: unsafe extern "C" fn(u64, *mut ()),
    /// Deoptimize: (bytecode_offset, shared_state) -> !
    pub deoptimize: unsafe extern "C" fn(u32, *mut ()),
    /// String concatenation: (left_val, right_val, shared_state) -> result_val
    pub string_concat: unsafe extern "C" fn(u64, u64, *mut ()) -> u64,
    /// Generic equality: (left_val, right_val, shared_state) -> bool
    pub generic_equals: unsafe extern "C" fn(u64, u64, *mut ()) -> bool,
    /// Structural/nominal field load with an AnyLayout generation guard.
    pub object_get_field:
        unsafe extern "C" fn(u64, u32, u64, u32, *const (), *mut ()) -> u64,
    /// Structural/nominal field store: (obj_val, expected_slot, value, func_id, module_ptr, shared_state) -> success
    pub object_set_field: unsafe extern "C" fn(u64, u32, u64, u32, *const (), *mut ()) -> bool,
    /// Structural shape check: (obj_val, shape_id, shared_state) -> implements
    pub object_implements_shape: unsafe extern "C" fn(u64, u64, *mut ()) -> bool,
    /// Nominal type check: (obj_val, local_nominal_type_index, module_ptr, shared_state) -> matches
    pub object_is_nominal: unsafe extern "C" fn(u64, u32, *const (), *mut ()) -> bool,
    /// Shape-aware field load: (obj_val, shape_id, expected_slot, optional, func_id, module_ptr, shared_state) -> result/sentinel
    pub object_get_shape_field:
        unsafe extern "C" fn(u64, u64, u32, u8, u32, *const (), *mut ()) -> u64,
    /// Shape-aware field store: (obj_val, shape_id, expected_slot, value, func_id, module_ptr, shared_state) -> status
    pub object_set_shape_field:
        unsafe extern "C" fn(u64, u64, u32, u64, u32, *const (), *mut ()) -> i8,
    /// String length: (string_val, shared_state) -> len or i32::MIN fallback sentinel
    pub string_len: unsafe extern "C" fn(u64, *mut ()) -> i32,
    /// String comparison: (left_val, right_val, shared_state) -> -1/0/1, or 2 for non-strings
    pub string_compare: unsafe extern "C" fn(u64, u64, *mut ()) -> i8,
    /// Value conversion: (value, shared_state) -> string value or fallback sentinel
    pub value_to_string: unsafe extern "C" fn(u64, *mut ()) -> u64,
    /// Interned constant string: (pool_index, module, shared_state) -> raw string pointer
    pub const_string: unsafe extern "C" fn(u32, *const (), *mut ()) -> *mut (),
    /// Array element load: (array_val, index, shared_state) -> element_val or fallback sentinel
    pub array_load: unsafe extern "C" fn(u64, i64, *mut ()) -> u64,
    /// Array element store: (array_val, index, value, shared_state) -> 1 success / 0 fallback
    pub array_store: unsafe extern "C" fn(u64, i64, u64, *mut ()) -> i8,
    /// Array push: (array_val, value, shared_state) -> 1 success / 0 fallback
    pub array_push: unsafe extern "C" fn(u64, u64, *mut ()) -> i8,
    /// Array pop: (array_val, shared_state) -> popped_val (null if empty) or fallback sentinel
    pub array_pop: unsafe extern "C" fn(u64, *mut ()) -> u64,
    /// Array length: (array_val, shared_state) -> len or i32::MIN fallback sentinel
    pub array_len: unsafe extern "C" fn(u64, *mut ()) -> i32,
    /// RefCell load: (refcell_val, shared_state) -> contained value, or the
    /// interpreter-fallback sentinel for a receiver that is not a pointer.
    pub refcell_load: unsafe extern "C" fn(u64, *mut ()) -> u64,
    /// RefCell store: (refcell_val, value, shared_state) -> 1 success / 0 fallback
    pub refcell_store: unsafe extern "C" fn(u64, u64, *mut ()) -> i8,
    /// RefCell allocate: (initial_value, shared_state) -> RefCell pointer, or null
    /// when the root set is unavailable.
    pub refcell_new: unsafe extern "C" fn(u64, *mut ()) -> u64,
    /// Patch one capture slot of an existing closure:
    /// (closure, capture_index, value, shared_state) -> 1 success / 0 fallback
    pub set_closure_capture: unsafe extern "C" fn(u64, u32, u64, *mut ()) -> i8,
    /// Closure allocate: (func_id, captures_ptr, capture_count, shared_state)
    /// -> closure pointer, or null when the root set is unavailable
    pub make_closure: unsafe extern "C" fn(u32, *const u64, u32, *mut ()) -> u64,
    /// Captured-value load: (capture_index, shared_state) -> value, or the
    /// interpreter-fallback sentinel.
    pub load_captured: unsafe extern "C" fn(u32, *mut ()) -> u64,
}

/// Validate the boxed arguments at a JIT entry boundary against a verified
/// signature. This is deliberately kept in the trampoline module so every
/// caller uses the same NaN-boxing rules as the native ABI.
pub fn boxed_arguments_match(
    args: &[u64],
    params: &[crate::compiler::bytecode::RuntimeTypeDescriptor],
) -> bool {
    use crate::vm::Value;
    args.iter().zip(params).all(|(raw, expected)| {
        let value = unsafe { Value::from_raw(*raw) };
        match expected {
            crate::compiler::bytecode::RuntimeTypeDescriptor::I32 => value.is_i32(),
            crate::compiler::bytecode::RuntimeTypeDescriptor::F64 => value.is_f64(),
            crate::compiler::bytecode::RuntimeTypeDescriptor::Bool => value.is_bool(),
            crate::compiler::bytecode::RuntimeTypeDescriptor::Null => value.is_null(),
            crate::compiler::bytecode::RuntimeTypeDescriptor::String
            | crate::compiler::bytecode::RuntimeTypeDescriptor::Ref
            | crate::compiler::bytecode::RuntimeTypeDescriptor::Object { .. }
            | crate::compiler::bytecode::RuntimeTypeDescriptor::Array { .. }
            | crate::compiler::bytecode::RuntimeTypeDescriptor::Tuple { .. }
            | crate::compiler::bytecode::RuntimeTypeDescriptor::Function { .. }
            | crate::compiler::bytecode::RuntimeTypeDescriptor::Task { .. } => value.is_ptr(),
            crate::compiler::bytecode::RuntimeTypeDescriptor::AnyValue => true,
            crate::compiler::bytecode::RuntimeTypeDescriptor::Void => false,
        }
    }) && args.len() == params.len()
}

#[cfg(test)]
mod tests {
    use super::boxed_arguments_match;
    use crate::compiler::bytecode::RuntimeTypeDescriptor as T;
    use crate::vm::Value;

    #[test]
    fn boxed_entry_guard_accepts_exact_primitive_tags() {
        let args = [Value::i32(7).raw(), Value::bool(true).raw(), Value::f64(2.5).raw()];
        assert!(boxed_arguments_match(&args, &[T::I32, T::Bool, T::F64]));
        assert!(!boxed_arguments_match(&args, &[T::F64, T::Bool, T::F64]));
    }

    #[test]
    fn boxed_entry_guard_rejects_wrong_arity_and_pointer_types() {
        let args = [Value::i32(7).raw()];
        assert!(!boxed_arguments_match(&args, &[T::I32, T::I32]));
        assert!(!boxed_arguments_match(&args, &[T::Ref]));
    }
}

#[cfg(test)]
mod abi_layout_tests {
    use super::*;

    #[test]
    fn helper_offsets_match_c_layout() {
        assert_eq!(RUNTIME_CONTEXT_HELPERS_OFFSET as usize, std::mem::offset_of!(RuntimeContext, helpers));
        assert_eq!(HELPER_SAFEPOINT_POLL_OFFSET as usize,
            std::mem::offset_of!(RuntimeContext, helpers) + std::mem::offset_of!(RuntimeHelperTable, safepoint_poll));
        assert_eq!(HELPER_CHECK_PREEMPTION_OFFSET as usize,
            std::mem::offset_of!(RuntimeContext, helpers) + std::mem::offset_of!(RuntimeHelperTable, check_preemption));
        assert_eq!(HELPER_NATIVE_CALL_OFFSET as usize,
            std::mem::offset_of!(RuntimeContext, helpers) + std::mem::offset_of!(RuntimeHelperTable, native_call_dispatch));
    }

    #[test]
    fn runtime_abi_is_pointer_aligned_and_versioned() {
        assert_eq!(std::mem::align_of::<RuntimeContext>(), std::mem::align_of::<*const ()>());
        assert_eq!(std::mem::align_of::<RuntimeHelperTable>(), std::mem::align_of::<*const ()>());
        assert_eq!(RUNTIME_CONTEXT_ABI_VERSION, 3);
        assert_eq!(
            RUNTIME_CONTEXT_ABI_VERSION,
            crate::compiler::bytecode::CURRENT_ABI_VERSION
        );
    }
}
