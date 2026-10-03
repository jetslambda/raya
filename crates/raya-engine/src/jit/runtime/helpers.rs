//! Runtime helper implementations for JIT RuntimeContext.
//!
//! Phase 3 focus:
//! - wire safepoint + preemption helpers used by lowered machine-code branches
//! - provide conservative stubs for not-yet-lowered runtime helpers

use crate::compiler::{Module, Opcode};
use crate::jit::runtime::trampoline::{RuntimeContext, RuntimeHelperTable};
use crate::vm::abi::{native_to_value, value_to_native, EngineContext};
use crate::vm::gc::GarbageCollector;
use crate::vm::interpreter::{
    ClassRegistry, ExecutionFrame, Interpreter, ModuleRuntimeLayout, ReturnAction,
    RuntimeLayoutRegistry, SafepointCoordinator, ShapeAdapter, StructuralAdapterKey,
    StructuralSlotBinding,
};
use crate::vm::native_handler::NativeHandler;
use crate::vm::native_registry::ResolvedNatives;
use crate::vm::object::{
    global_layout_names, Array, BoundMethod, BoundNativeMethod, Closure, Object, RayaString,
};
use crate::vm::reflect::ClassMetadataRegistry;
use crate::vm::scheduler::IoSubmission;
use crate::vm::scheduler::{Task, TaskId};
use crate::vm::stack::Stack;
use crate::vm::sync::{MutexRegistry, SemaphoreRegistry};
use crate::vm::value::Value;
use crate::vm::value_semantics::{
    compare_strings, raya_string_ptr_checked, value_to_string, values_equal,
};
use crate::vm::VmError;
use crossbeam_deque::Injector;
use raya_sdk::NativeCallResult;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::RefCell;
use std::ptr::NonNull;
use std::sync::Arc;

/// Sentinel returned by JIT native helper dispatch when the native call suspended.
/// Distinct from valid NaN-boxed Values.
pub const JIT_NATIVE_SUSPEND_SENTINEL: u64 = 0xFFFF_DEAD_0000_0001;
pub const JIT_INTERPRETER_FALLBACK_SENTINEL: u64 = 0xFFFF_DEAD_0000_0002;
pub const JIT_INTERPRETER_EXCEPTION_SENTINEL: u64 = 0xFFFF_DEAD_0000_0003;
pub const JIT_SHAPE_FIELD_FALLBACK_SENTINEL: u64 = 0xFFFF_DEAD_0000_0004;
// Uses the null tag with a non-zero payload. Raya constructors never produce
// this invalid NaN-boxed encoding, so it cannot alias a valid Value.
pub const JIT_LAYOUT_GUARD_FALLBACK_SENTINEL: u64 = 0xFFFE_DEAD_0000_0005;
pub const JIT_STRING_LEN_FALLBACK_SENTINEL: i32 = i32::MIN;
/// Returned by `helper_array_len` when the receiver is not an array. Distinct
/// from any valid non-negative length.
pub const JIT_ARRAY_LEN_FALLBACK_SENTINEL: i32 = i32::MIN;
const JIT_SHAPE_ADAPTER_PIC_CAPACITY: usize = 4;

thread_local! {
    static JIT_SHAPE_ADAPTER_LAST: RefCell<Option<(StructuralAdapterKey, u32, Arc<ShapeAdapter>)>> =
        const { RefCell::new(None) };
    static JIT_SHAPE_ADAPTER_PIC: RefCell<Vec<(StructuralAdapterKey, u32, Arc<ShapeAdapter>)>> =
        const { RefCell::new(Vec::new()) };
}

const JIT_STORE_SUCCESS: i8 = 1;
const JIT_STORE_FALLBACK: i8 = 0;

#[repr(C)]
pub struct JitRuntimeBridgeContext {
    pub safepoint: *const SafepointCoordinator,
    pub task: *const Task,
    pub task_arc: *const Arc<Task>,
    pub gc: *const parking_lot::Mutex<GarbageCollector>,
    pub classes: *const parking_lot::RwLock<ClassRegistry>,
    pub layouts: *const parking_lot::RwLock<RuntimeLayoutRegistry>,
    pub code_cache: *const crate::jit::runtime::code_cache::CodeCache,
    pub mutex_registry: *const MutexRegistry,
    pub semaphore_registry: *const SemaphoreRegistry,
    pub globals_by_index: *const parking_lot::RwLock<Vec<Value>>,
    pub builtin_global_slots: *const parking_lot::RwLock<FxHashMap<String, usize>>,
    pub constant_string_cache:
        *const parking_lot::RwLock<FxHashMap<([u8; 32], usize), Value>>,
    pub ephemeral_gc_roots: *const parking_lot::RwLock<Vec<Value>>,
    pub pinned_handles: *const parking_lot::RwLock<rustc_hash::FxHashSet<u64>>,
    pub tasks: *const Arc<parking_lot::RwLock<FxHashMap<TaskId, Arc<Task>>>>,
    pub injector: *const Arc<Injector<Arc<Task>>>,
    pub module_layouts:
        *const parking_lot::RwLock<FxHashMap<[u8; 32], ModuleRuntimeLayout>>,
    pub metadata: *const parking_lot::Mutex<crate::vm::reflect::MetadataStore>,
    pub class_metadata: *const parking_lot::RwLock<ClassMetadataRegistry>,
    pub native_handler: *const Arc<dyn NativeHandler>,
    pub resolved_natives: *const parking_lot::RwLock<ResolvedNatives>,
    pub structural_shape_names:
        *const parking_lot::RwLock<FxHashMap<u64, Vec<String>>>,
    pub structural_layout_shapes:
        *const parking_lot::RwLock<FxHashMap<crate::vm::object::LayoutId, Vec<String>>>,
    pub structural_shape_adapters: *const parking_lot::RwLock<
        FxHashMap<StructuralAdapterKey, Arc<ShapeAdapter>>,
    >,
    pub aot_profile: *const parking_lot::RwLock<crate::aot_profile::AotProfileCollector>,
    pub type_handles:
        *const parking_lot::RwLock<crate::vm::interpreter::RuntimeTypeHandleRegistry>,
    pub prop_keys: *const parking_lot::RwLock<crate::vm::interpreter::PropertyKeyRegistry>,
    pub stack_pool: *const crate::vm::scheduler::StackPool,
    pub io_submit_tx: *const crossbeam::channel::Sender<IoSubmission>,
    pub max_preemptions: u32,
    pub current_frame_depth: usize,
}

/// Build a runtime context for a JIT invocation running inside interpreter thread loop.
#[inline]
pub fn build_runtime_bridge_context(
    safepoint: &SafepointCoordinator,
    task: &Arc<Task>,
    gc: &parking_lot::Mutex<GarbageCollector>,
    classes: &parking_lot::RwLock<ClassRegistry>,
    layouts: &parking_lot::RwLock<RuntimeLayoutRegistry>,
    code_cache: &crate::jit::runtime::code_cache::CodeCache,
    mutex_registry: &MutexRegistry,
    semaphore_registry: &SemaphoreRegistry,
    globals_by_index: &parking_lot::RwLock<Vec<Value>>,
    builtin_global_slots: &parking_lot::RwLock<FxHashMap<String, usize>>,
    constant_string_cache: &parking_lot::RwLock<FxHashMap<([u8; 32], usize), Value>>,
    ephemeral_gc_roots: &parking_lot::RwLock<Vec<Value>>,
    pinned_handles: &parking_lot::RwLock<rustc_hash::FxHashSet<u64>>,
    tasks: &Arc<parking_lot::RwLock<FxHashMap<TaskId, Arc<Task>>>>,
    injector: &Arc<Injector<Arc<Task>>>,
    module_layouts: &parking_lot::RwLock<FxHashMap<[u8; 32], ModuleRuntimeLayout>>,
    metadata: &parking_lot::Mutex<crate::vm::reflect::MetadataStore>,
    class_metadata: &parking_lot::RwLock<ClassMetadataRegistry>,
    native_handler: &Arc<dyn NativeHandler>,
    resolved_natives: &parking_lot::RwLock<ResolvedNatives>,
    structural_shape_names: &parking_lot::RwLock<FxHashMap<u64, Vec<String>>>,
    structural_layout_shapes: &parking_lot::RwLock<FxHashMap<crate::vm::object::LayoutId, Vec<String>>>,
    structural_shape_adapters: &parking_lot::RwLock<
        FxHashMap<StructuralAdapterKey, Arc<ShapeAdapter>>,
    >,
    aot_profile: &parking_lot::RwLock<crate::aot_profile::AotProfileCollector>,
    type_handles: &parking_lot::RwLock<crate::vm::interpreter::RuntimeTypeHandleRegistry>,
    prop_keys: &parking_lot::RwLock<crate::vm::interpreter::PropertyKeyRegistry>,
    stack_pool: &crate::vm::scheduler::StackPool,
    max_preemptions: u32,
    current_frame_depth: usize,
    io_submit_tx: Option<&crossbeam::channel::Sender<IoSubmission>>,
) -> JitRuntimeBridgeContext {
    JitRuntimeBridgeContext {
        safepoint: safepoint as *const SafepointCoordinator,
        task: task.as_ref() as *const Task,
        task_arc: task as *const Arc<Task>,
        gc: gc as *const _,
        classes: classes as *const _,
        layouts: layouts as *const _,
        code_cache: code_cache as *const _,
        mutex_registry: mutex_registry as *const _,
        semaphore_registry: semaphore_registry as *const _,
        globals_by_index: globals_by_index as *const _,
        builtin_global_slots: builtin_global_slots as *const _,
        constant_string_cache: constant_string_cache as *const _,
        ephemeral_gc_roots: ephemeral_gc_roots as *const _,
        pinned_handles: pinned_handles as *const _,
        tasks: tasks as *const _,
        injector: injector as *const _,
        module_layouts: module_layouts as *const _,
        metadata: metadata as *const _,
        class_metadata: class_metadata as *const _,
        native_handler: native_handler as *const _,
        resolved_natives: resolved_natives as *const _,
        structural_shape_names: structural_shape_names as *const _,
        structural_layout_shapes: structural_layout_shapes as *const _,
        structural_shape_adapters: structural_shape_adapters as *const _,
        aot_profile: aot_profile as *const _,
        type_handles: type_handles as *const _,
        prop_keys: prop_keys as *const _,
        stack_pool: stack_pool as *const _,
        io_submit_tx: io_submit_tx.map_or(std::ptr::null(), |tx| tx as *const _),
        max_preemptions,
        current_frame_depth,
    }
}

#[inline]
pub fn build_runtime_context(bridge: &JitRuntimeBridgeContext, module: &Module) -> RuntimeContext {
    RuntimeContext {
        shared_state: (bridge as *const JitRuntimeBridgeContext).cast::<()>(),
        current_task: bridge.task.cast::<()>(),
        module: (module as *const Module).cast::<()>(),
        helpers: runtime_helpers(),
    }
}

#[inline]
pub fn runtime_helpers() -> RuntimeHelperTable {
    RuntimeHelperTable {
        alloc_object: helper_alloc_object,
        alloc_array: helper_alloc_array,
        alloc_string: helper_alloc_string,
        safepoint_poll: helper_safepoint_poll,
        check_preemption: helper_check_preemption,
        native_call_dispatch: helper_native_call_dispatch,
        interpreter_call: helper_interpreter_call,
        string_concat: helper_string_concat,
        generic_equals: helper_generic_equals,
        object_get_field: helper_object_get_field,
        object_set_field: helper_object_set_field,
        object_implements_shape: helper_object_implements_shape,
        object_is_nominal: helper_object_is_nominal,
        object_get_shape_field: helper_object_get_shape_field,
        object_set_shape_field: helper_object_set_shape_field,
        string_len: helper_string_len,
        string_compare: helper_string_compare,
        value_to_string: helper_value_to_string,
        const_string: helper_const_string,
        array_load: helper_array_load,
        array_store: helper_array_store,
        array_push: helper_array_push,
        array_pop: helper_array_pop,
        array_len: helper_array_len,
        refcell_load: helper_load_refcell,
        refcell_store: helper_store_refcell,
        refcell_new: helper_new_refcell,
        set_closure_capture: helper_set_closure_capture,
        make_closure: helper_make_closure,
        load_captured: helper_load_captured,
        store_captured: helper_store_captured,
        bind_method: helper_bind_method,
        await_task: helper_await_task,
        dyn_get_keyed: helper_dyn_get_keyed,
    }
}

#[inline]
unsafe fn jit_object_ptr_checked(value: Value) -> Option<NonNull<Object>> {
    if !value.is_ptr() {
        return None;
    }
    let ptr = value.as_ptr::<u8>()?;
    let header = &*crate::vm::gc::header_ptr_from_value_ptr(ptr.as_ptr());
    if header.type_id() == std::any::TypeId::of::<Object>() {
        value.as_ptr::<Object>()
    } else {
        None
    }
}

fn jit_layout_field_names(
    bridge: &JitRuntimeBridgeContext,
    object: &Object,
) -> Option<Vec<String>> {
    if !bridge.layouts.is_null() {
        let layouts = unsafe { &*bridge.layouts }.read();
        if let Some(names) = layouts.layout_field_names(object.layout_id()) {
            return Some(names.to_vec());
        }
    }
    global_layout_names(object.layout_id())
}

fn jit_build_shape_slot_map_for_object(
    bridge: &JitRuntimeBridgeContext,
    object: &Object,
    required_names: &[String],
) -> Option<Vec<StructuralSlotBinding>> {
    let layout_names = jit_layout_field_names(bridge, object);
    let dynamic_binding_for = |name: &str| -> Option<StructuralSlotBinding> {
        if bridge.prop_keys.is_null() {
            return None;
        }
        let key = unsafe { &*bridge.prop_keys }.write().intern(name);
        object
            .dyn_map()
            .and_then(|dyn_map| dyn_map.contains_key(&key).then_some(StructuralSlotBinding::Dynamic(key)))
    };

    if let Some(nominal_type_id) = object.nominal_type_id_usize() {
        let class_meta = if bridge.class_metadata.is_null() {
            None
        } else {
            unsafe { &*bridge.class_metadata }.read().get(nominal_type_id).cloned()
        };
        return Some(
            required_names
                .iter()
                .map(|name| {
                    class_meta
                        .as_ref()
                        .and_then(|meta| meta.get_field_index(name))
                        .and_then(|index| {
                            (index < object.field_count()).then_some(StructuralSlotBinding::Field(index))
                        })
                        .or_else(|| {
                            layout_names
                                .as_ref()
                                .and_then(|names| names.iter().position(|actual| actual == name))
                                .map(StructuralSlotBinding::Field)
                        })
                        .or_else(|| {
                            class_meta
                                .as_ref()
                                .and_then(|meta| meta.get_method_index(name))
                                .map(StructuralSlotBinding::Method)
                        })
                        .or_else(|| dynamic_binding_for(name))
                        .unwrap_or(StructuralSlotBinding::Missing)
                })
                .collect(),
        );
    }

    Some(
        required_names
            .iter()
            .map(|name| {
                layout_names
                    .as_ref()
                    .and_then(|names| names.iter().position(|actual| actual == name))
                    .map(StructuralSlotBinding::Field)
                    .or_else(|| dynamic_binding_for(name))
                    .unwrap_or(StructuralSlotBinding::Missing)
            })
            .collect(),
    )
}

fn jit_ensure_shape_adapter_for_object(
    bridge: &JitRuntimeBridgeContext,
    object: &Object,
    required_shape: u64,
) -> Option<Arc<ShapeAdapter>> {
    if bridge.structural_shape_adapters.is_null() {
        return None;
    }

    let adapter_key = StructuralAdapterKey {
        provider_layout: object.layout_id(),
        required_shape,
    };
    let current_epoch = if bridge.layouts.is_null() {
        0
    } else {
        unsafe { &*bridge.layouts }
            .read()
            .layout_epoch(object.layout_id())
            .unwrap_or(0)
    };
    if let Some(adapter) = JIT_SHAPE_ADAPTER_LAST.with(|cache| {
        let borrowed = cache.borrow();
        let Some((cached_key, cached_epoch, adapter)) = borrowed.as_ref() else {
            return None;
        };
        if *cached_key == adapter_key && *cached_epoch == current_epoch {
            Some(adapter.clone())
        } else {
            None
        }
    }) {
        return Some(adapter);
    }
    if let Some(adapter) = JIT_SHAPE_ADAPTER_PIC.with(|cache| {
        let borrowed = cache.borrow();
        borrowed
            .iter()
            .find(|(cached_key, cached_epoch, _)| {
                *cached_key == adapter_key && *cached_epoch == current_epoch
            })
            .map(|(_, _, adapter)| adapter.clone())
    }) {
        JIT_SHAPE_ADAPTER_LAST.with(|cache| {
            *cache.borrow_mut() = Some((adapter_key, current_epoch, adapter.clone()));
        });
        return Some(adapter);
    }
    if let Some(adapter) = unsafe { &*bridge.structural_shape_adapters }
        .read()
        .get(&adapter_key)
        .cloned()
    {
        if adapter.epoch == current_epoch {
            JIT_SHAPE_ADAPTER_LAST.with(|cache| {
                *cache.borrow_mut() = Some((adapter_key, current_epoch, adapter.clone()));
            });
            JIT_SHAPE_ADAPTER_PIC.with(|cache| {
                let mut cache = cache.borrow_mut();
                if let Some(pos) = cache
                    .iter()
                    .position(|(cached_key, _, _)| *cached_key == adapter_key)
                {
                    cache.remove(pos);
                }
                cache.insert(0, (adapter_key, current_epoch, adapter.clone()));
                cache.truncate(JIT_SHAPE_ADAPTER_PIC_CAPACITY);
            });
            return Some(adapter);
        }
    }

    if bridge.structural_shape_names.is_null() {
        return None;
    }
    let required_names = unsafe { &*bridge.structural_shape_names }
        .read()
        .get(&required_shape)
        .cloned()?;
    let slot_map = jit_build_shape_slot_map_for_object(bridge, object, &required_names)?;
    let adapter = Arc::new(ShapeAdapter::from_slot_map(
        object.layout_id(),
        required_shape,
        &slot_map,
        current_epoch,
    ));
    let mut adapters = unsafe { &*bridge.structural_shape_adapters }.write();
    let adapter = adapters
        .entry(adapter_key)
        .or_insert_with(|| adapter.clone())
        .clone();
    JIT_SHAPE_ADAPTER_LAST.with(|cache| {
        *cache.borrow_mut() = Some((adapter_key, current_epoch, adapter.clone()));
    });
    JIT_SHAPE_ADAPTER_PIC.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(pos) = cache
            .iter()
            .position(|(cached_key, _, _)| *cached_key == adapter_key)
        {
            cache.remove(pos);
        }
        cache.insert(0, (adapter_key, current_epoch, adapter.clone()));
        cache.truncate(JIT_SHAPE_ADAPTER_PIC_CAPACITY);
    });
    Some(adapter)
}

fn jit_resolve_nominal_type_id(
    bridge: &JitRuntimeBridgeContext,
    module: &Module,
    local_nominal_type_index: u32,
) -> Option<usize> {
    if bridge.module_layouts.is_null() {
        return None;
    }
    let module_layouts = unsafe { &*bridge.module_layouts }.read();
    let module_layout = module_layouts.get(&module.checksum)?;
    if local_nominal_type_index as usize >= module_layout.nominal_type_len {
        return None;
    }
    Some(module_layout.nominal_type_base + local_nominal_type_index as usize)
}

fn jit_object_matches_nominal_type(
    bridge: &JitRuntimeBridgeContext,
    object: &Object,
    target_nominal_type_id: usize,
) -> bool {
    let Some(mut current_nominal_type_id) = object.nominal_type_id_usize() else {
        return false;
    };
    if bridge.classes.is_null() {
        return false;
    }
    let classes = unsafe { &*bridge.classes }.read();
    loop {
        if current_nominal_type_id == target_nominal_type_id {
            return true;
        }
        let Some(class) = classes.get_class(current_nominal_type_id) else {
            return false;
        };
        let Some(parent_id) = class.parent_id else {
            return false;
        };
        current_nominal_type_id = parent_id;
    }
}

fn jit_build_interpreter<'a>(bridge: &'a JitRuntimeBridgeContext) -> Option<Interpreter<'a>> {
    if bridge.gc.is_null()
        || bridge.classes.is_null()
        || bridge.layouts.is_null()
        || bridge.mutex_registry.is_null()
        || bridge.semaphore_registry.is_null()
        || bridge.safepoint.is_null()
        || bridge.globals_by_index.is_null()
        || bridge.builtin_global_slots.is_null()
        || bridge.constant_string_cache.is_null()
        || bridge.ephemeral_gc_roots.is_null()
        || bridge.pinned_handles.is_null()
        || bridge.tasks.is_null()
        || bridge.injector.is_null()
        || bridge.metadata.is_null()
        || bridge.class_metadata.is_null()
        || bridge.native_handler.is_null()
        || bridge.module_layouts.is_null()
        || bridge.structural_shape_adapters.is_null()
        || bridge.structural_shape_names.is_null()
        || bridge.structural_layout_shapes.is_null()
        || bridge.type_handles.is_null()
        || bridge.prop_keys.is_null()
        || bridge.aot_profile.is_null()
        || bridge.stack_pool.is_null()
    {
        return None;
    }

    Some(Interpreter::new(
        unsafe { &*bridge.gc },
        unsafe { &*bridge.classes },
        unsafe { &*bridge.layouts },
        unsafe { &*bridge.mutex_registry },
        unsafe { &*bridge.semaphore_registry },
        unsafe { &*bridge.safepoint },
        unsafe { &*bridge.globals_by_index },
        unsafe { &*bridge.builtin_global_slots },
        unsafe { &*bridge.constant_string_cache },
        unsafe { &*bridge.ephemeral_gc_roots },
        unsafe { &*bridge.pinned_handles },
        unsafe { &*bridge.tasks },
        unsafe { &*bridge.injector },
        unsafe { &*bridge.metadata },
        unsafe { &*bridge.class_metadata },
        unsafe { &*bridge.native_handler },
        unsafe { &*bridge.module_layouts },
        unsafe { &*bridge.structural_shape_adapters },
        unsafe { &*bridge.structural_shape_names },
        unsafe { &*bridge.structural_layout_shapes },
        unsafe { &*bridge.type_handles },
        unsafe { &*bridge.prop_keys },
        unsafe { &*bridge.aot_profile },
        if bridge.io_submit_tx.is_null() {
            None
        } else {
            Some(unsafe { &*bridge.io_submit_tx })
        },
        bridge.max_preemptions,
        unsafe { &*bridge.stack_pool },
    ))
}

fn jit_raise_vm_error(bridge: &JitRuntimeBridgeContext, error: VmError) {
    if bridge.task.is_null() || bridge.gc.is_null() {
        return;
    }
    let task = unsafe { &*bridge.task };
    if task.has_exception() {
        return;
    }
    let raya_string = crate::vm::object::RayaString::new(error.to_string());
    let gc_ptr = unsafe { &*bridge.gc }.lock().allocate(raya_string);
    let exc_val = unsafe { Value::from_ptr(NonNull::new(gc_ptr.as_ptr()).unwrap()) };
    task.set_exception(exc_val);
}

#[derive(Clone, Copy)]
struct JitTaskStateSnapshot {
    exception_handler_count: usize,
    call_frame_count: usize,
    closure_count: usize,
    held_mutex_count: usize,
    current_exception: Option<Value>,
    caught_exception: Option<Value>,
}

fn jit_snapshot_task_state(task: &Task) -> JitTaskStateSnapshot {
    JitTaskStateSnapshot {
        exception_handler_count: task.exception_handler_count(),
        call_frame_count: task.call_frame_count(),
        closure_count: task.closure_count(),
        held_mutex_count: task.held_mutex_count(),
        current_exception: task.current_exception(),
        caught_exception: task.caught_exception(),
    }
}

fn jit_restore_task_exceptions(
    task: &Task,
    current_exception: Option<Value>,
    caught_exception: Option<Value>,
) {
    if let Some(exception) = current_exception {
        task.set_exception(exception);
    } else {
        task.clear_exception();
    }
    if let Some(exception) = caught_exception {
        task.set_caught_exception(exception);
    } else {
        task.clear_caught_exception();
    }
}

fn jit_rollback_mutexes(
    bridge: &JitRuntimeBridgeContext,
    task: &Task,
    snapshot: &JitTaskStateSnapshot,
) {
    let released = task.take_mutexes_since(snapshot.held_mutex_count);
    if released.is_empty() || bridge.mutex_registry.is_null() {
        return;
    }

    let registry = unsafe { &*bridge.mutex_registry };
    for mutex_id in released.into_iter().rev() {
        let Some(mutex) = registry.get(mutex_id) else {
            continue;
        };
        let Ok(next_waiter) = mutex.unlock(task.id()) else {
            continue;
        };
        if let Some(waiter_id) = next_waiter {
            if bridge.tasks.is_null() || bridge.injector.is_null() {
                continue;
            }
            let tasks = unsafe { &*bridge.tasks }.read();
            if let Some(waiter_task) = tasks.get(&waiter_id) {
                waiter_task.add_held_mutex(mutex_id);
                waiter_task.set_state(crate::vm::scheduler::TaskState::Resumed);
                waiter_task.clear_suspend_reason();
                unsafe { &*bridge.injector }.push(waiter_task.clone());
            }
        }
    }
}

fn jit_restore_task_state(
    bridge: &JitRuntimeBridgeContext,
    task: &Task,
    snapshot: &JitTaskStateSnapshot,
    preserve_current_exception: bool,
) {
    while task.exception_handler_count() > snapshot.exception_handler_count {
        let _ = task.pop_exception_handler();
    }
    while task.call_frame_count() > snapshot.call_frame_count {
        let _ = task.pop_call_frame();
    }
    while task.closure_count() > snapshot.closure_count {
        let _ = task.pop_closure();
    }
    jit_rollback_mutexes(bridge, task, snapshot);

    let current_exception = if preserve_current_exception {
        task.current_exception().or(snapshot.current_exception)
    } else {
        snapshot.current_exception
    };
    jit_restore_task_exceptions(task, current_exception, snapshot.caught_exception);
}

fn jit_function_is_sync_safe(
    module: &Module,
    func_id: usize,
    visiting: &mut FxHashSet<([u8; 32], usize)>,
) -> bool {
    let key = (module.checksum, func_id);
    if !visiting.insert(key) {
        return true;
    }
    let Some(func) = module.functions.get(func_id) else {
        return false;
    };
    let Ok(instrs) = crate::jit::analysis::decoder::decode_function(&func.code) else {
        return false;
    };
    for instr in instrs {
        use crate::jit::analysis::decoder::Operands;
        match instr.opcode {
            Opcode::Await
            | Opcode::WaitAll
            | Opcode::Sleep
            | Opcode::MutexLock
            | Opcode::Yield
            | Opcode::NativeCall
            | Opcode::ModuleNativeCall
            | Opcode::Spawn
            | Opcode::SpawnClosure
            | Opcode::TaskCancel => return false,
            Opcode::Call => match instr.operands {
                Operands::Call {
                    func_index: 0xFFFF_FFFF,
                    ..
                } => return false,
                Operands::Call { func_index, .. } => {
                    if !jit_function_is_sync_safe(module, func_index as usize, visiting) {
                        return false;
                    }
                }
                _ => return false,
            },
            Opcode::CallStatic => match instr.operands {
                Operands::Call { func_index, .. } => {
                    if !jit_function_is_sync_safe(module, func_index as usize, visiting) {
                        return false;
                    }
                }
                _ => return false,
            },
            Opcode::CallMethodExact
            | Opcode::OptionalCallMethodExact
            | Opcode::CallMethodShape
            | Opcode::OptionalCallMethodShape
            | Opcode::CallConstructor
            | Opcode::ConstructType
            | Opcode::CallSuper => return false,
            _ => {}
        }
    }
    true
}

enum JitNestedCallResult {
    Value(Value),
    Fallback,
    Exception,
}

fn jit_apply_return_action(
    stack: &mut Stack,
    return_value: Value,
    return_action: ReturnAction,
) -> Result<Option<Value>, VmError> {
    match return_action {
        ReturnAction::PushReturnValue => {
            stack.push(return_value)?;
            Ok(None)
        }
        ReturnAction::PushObject(obj) => {
            stack.push(obj)?;
            Ok(None)
        }
        ReturnAction::Discard => Ok(None),
    }
}

fn jit_execute_sync_frame(
    interpreter: &mut Interpreter<'_>,
    bridge: &JitRuntimeBridgeContext,
    stack: &mut Stack,
    initial_module: Arc<Module>,
    initial_func_id: usize,
    initial_arg_count: usize,
    initial_is_closure: bool,
    initial_closure_val: Option<Value>,
    initial_return_action: ReturnAction,
) -> JitNestedCallResult {
    let Some(task) = (!bridge.task_arc.is_null()).then(|| unsafe { &*bridge.task_arc }) else {
        return JitNestedCallResult::Fallback;
    };
    let task_snapshot = jit_snapshot_task_state(task.as_ref());

    let mut frames: Vec<ExecutionFrame> = Vec::new();
    let mut module = initial_module;
    let mut current_func_id = initial_func_id;
    let mut ip = 0usize;
    let mut current_arg_count = initial_arg_count;
    let mut current_is_closure = initial_is_closure;
    let mut current_return_action = initial_return_action;

    macro_rules! finish_nested_call {
        ($result:expr, $preserve_exception:expr) => {{
            jit_restore_task_state(bridge, task.as_ref(), &task_snapshot, $preserve_exception);
            return $result;
        }};
    }

    task.push_call_frame(current_func_id);
    if let Some(closure_val) = initial_closure_val {
        task.push_closure(closure_val);
    }

    let mut locals_base = stack.depth().saturating_sub(initial_arg_count);
    let local_count = module
        .functions
        .get(current_func_id)
        .map(|f| f.local_count)
        .unwrap_or(initial_arg_count);
    for _ in 0..local_count.saturating_sub(initial_arg_count) {
        if let Err(error) = stack.push(Value::null()) {
            jit_raise_vm_error(bridge, error);
            finish_nested_call!(JitNestedCallResult::Exception, true);
        }
    }

    loop {
        let code = &module.functions[current_func_id].code;
        if ip >= code.len() {
            let return_value = if stack.depth() > locals_base + module.functions[current_func_id].local_count
            {
                stack.pop().unwrap_or_else(|_| Value::null())
            } else {
                Value::null()
            };
            while stack.depth() > locals_base {
                let _ = stack.pop();
            }
            task.pop_call_frame();
            if current_is_closure {
                task.pop_closure();
            }
            if let Some(frame) = frames.pop() {
                if let Err(error) = jit_apply_return_action(stack, return_value, current_return_action)
                {
                    jit_raise_vm_error(bridge, error);
                    finish_nested_call!(JitNestedCallResult::Exception, true);
                }
                module = frame.module;
                current_func_id = frame.func_id;
                ip = frame.ip;
                locals_base = frame.locals_base;
                current_is_closure = frame.is_closure;
                current_return_action = frame.return_action;
                current_arg_count = frame.arg_count;
                continue;
            }
            finish_nested_call!(
                match current_return_action {
                ReturnAction::PushReturnValue => JitNestedCallResult::Value(return_value),
                ReturnAction::PushObject(obj) => JitNestedCallResult::Value(obj),
                ReturnAction::Discard => JitNestedCallResult::Value(Value::null()),
            },
                false
            );
        }

        let opcode = match Opcode::from_u8(code[ip]) {
            Some(op) => op,
            None => {
                jit_raise_vm_error(bridge, VmError::InvalidOpcode(code[ip]));
                finish_nested_call!(JitNestedCallResult::Exception, true);
            }
        };
        ip += 1;

        let frame_depth = bridge.current_frame_depth + 1 + frames.len();
        match interpreter.execute_opcode(
            task,
            stack,
            &mut ip,
            code,
            module.as_ref(),
            opcode,
            locals_base,
            frame_depth,
            current_arg_count,
            ) {
            crate::vm::interpreter::OpcodeResult::Continue => {}
            crate::vm::interpreter::OpcodeResult::Return(return_value) => {
                while stack.depth() > locals_base {
                    let _ = stack.pop();
                }
                task.pop_call_frame();
                if current_is_closure {
                    task.pop_closure();
                }
                if let Some(frame) = frames.pop() {
                    if let Err(error) = jit_apply_return_action(stack, return_value, current_return_action)
                    {
                        jit_raise_vm_error(bridge, error);
                        finish_nested_call!(JitNestedCallResult::Exception, true);
                    }
                    module = frame.module;
                    current_func_id = frame.func_id;
                    ip = frame.ip;
                    locals_base = frame.locals_base;
                    current_is_closure = frame.is_closure;
                    current_return_action = frame.return_action;
                    current_arg_count = frame.arg_count;
                } else {
                    finish_nested_call!(
                        match current_return_action {
                        ReturnAction::PushReturnValue => JitNestedCallResult::Value(return_value),
                        ReturnAction::PushObject(obj) => JitNestedCallResult::Value(obj),
                        ReturnAction::Discard => JitNestedCallResult::Value(Value::null()),
                    },
                        false
                    );
                }
            }
            crate::vm::interpreter::OpcodeResult::Suspend(_) => {
                finish_nested_call!(JitNestedCallResult::Fallback, false);
            }
            crate::vm::interpreter::OpcodeResult::PushFrame {
                func_id,
                arg_count,
                is_closure,
                closure_val,
                module: callee_module,
                return_action,
            } => {
                let callee_module = callee_module.unwrap_or_else(|| module.clone());
                if !jit_function_is_sync_safe(
                    callee_module.as_ref(),
                    func_id,
                    &mut FxHashSet::default(),
                ) {
                    finish_nested_call!(JitNestedCallResult::Fallback, false);
                }

                frames.push(ExecutionFrame {
                    module: module.clone(),
                    func_id: current_func_id,
                    ip,
                    locals_base,
                    is_closure: current_is_closure,
                    return_action: current_return_action,
                    arg_count: current_arg_count,
                });
                task.push_call_frame(func_id);
                if let Some(cv) = closure_val {
                    task.push_closure(cv);
                }

                locals_base = stack.depth().saturating_sub(arg_count);
                let local_count = callee_module
                    .functions
                    .get(func_id)
                    .map(|f| f.local_count)
                    .unwrap_or(arg_count);
                for _ in 0..local_count.saturating_sub(arg_count) {
                    if let Err(error) = stack.push(Value::null()) {
                        jit_raise_vm_error(bridge, error);
                        finish_nested_call!(JitNestedCallResult::Exception, true);
                    }
                }

                module = callee_module;
                current_func_id = func_id;
                ip = 0;
                current_arg_count = arg_count;
                current_is_closure = is_closure;
                current_return_action = return_action;
            }
            crate::vm::interpreter::OpcodeResult::Error(error) => {
                if !task.has_exception() {
                    jit_raise_vm_error(bridge, error);
                }

                let exception = task.current_exception().unwrap_or_else(Value::null);
                let mut handled = false;
                'exception_search: loop {
                    let current_frame_depth = bridge.current_frame_depth + 1 + frames.len();
                    while let Some(handler) = task.peek_exception_handler() {
                        if handler.frame_count != current_frame_depth {
                            break;
                        }

                        while stack.depth() > handler.stack_size {
                            let _ = stack.pop();
                        }

                        if handler.catch_offset != -1 {
                            task.pop_exception_handler();
                            task.set_caught_exception(exception);
                            task.clear_exception();
                            if let Err(push_error) = stack.push(exception) {
                                jit_raise_vm_error(bridge, push_error);
                                finish_nested_call!(JitNestedCallResult::Exception, true);
                            }
                            ip = handler.catch_offset as usize;
                            handled = true;
                            break 'exception_search;
                        }

                        if handler.finally_offset != -1 {
                            task.pop_exception_handler();
                            ip = handler.finally_offset as usize;
                            handled = true;
                            break 'exception_search;
                        }

                        task.pop_exception_handler();
                    }

                    if let Some(frame) = frames.pop() {
                        task.pop_call_frame();
                        if current_is_closure {
                            task.pop_closure();
                        }
                        module = frame.module;
                        current_func_id = frame.func_id;
                        ip = frame.ip;
                        locals_base = frame.locals_base;
                        current_is_closure = frame.is_closure;
                        current_return_action = frame.return_action;
                        current_arg_count = frame.arg_count;
                    } else {
                        break;
                    }
                }

                if !handled {
                    finish_nested_call!(JitNestedCallResult::Exception, true);
                }
            }
        }
    }
}

unsafe extern "C" fn helper_alloc_object(
    local_nominal_type_index: u32,
    module_ptr: *const (),
    shared_state: *mut (),
) -> *mut () {
    if shared_state.is_null() || module_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let bridge = &*(shared_state.cast::<JitRuntimeBridgeContext>());
    if bridge.gc.is_null() || bridge.layouts.is_null() || bridge.module_layouts.is_null() {
        return std::ptr::null_mut();
    }

    let module = &*(module_ptr.cast::<Module>());
    let Some(nominal_type_id) = jit_resolve_nominal_type_id(bridge, module, local_nominal_type_index)
    else {
        return std::ptr::null_mut();
    };
    let (field_count, layout_id) = {
        let layouts = (&*bridge.layouts).read();
        match layouts.nominal_allocation(nominal_type_id) {
            Some((layout_id, field_count)) => (field_count, layout_id),
            None => return std::ptr::null_mut(),
        }
    };

    let mut gc = (&*bridge.gc).lock();
    let obj_ptr = gc.allocate(Object::new_nominal(
        layout_id,
        nominal_type_id as u32,
        field_count,
    ));
    obj_ptr.as_ptr().cast::<()>()
}

/// Allocate a new array of `capacity` null-filled slots.
///
/// `type_index` is the emitted element descriptor id (see the interpreter's
/// `NewArray`); it is resolved into the array's self-contained element
/// constraint. Returns null on any misconfiguration so the caller fails closed.
///
/// A fresh array holds no live operands across the single `gc.allocate` call,
/// so no ephemeral-root scope is needed here; the returned pointer is published
/// to the caller immediately.
unsafe extern "C" fn helper_alloc_array(
    type_index: u32,
    capacity: usize,
    module_ptr: *const (),
    shared_state: *mut (),
) -> *mut () {
    if shared_state.is_null() {
        return std::ptr::null_mut();
    }
    let bridge = &*(shared_state.cast::<JitRuntimeBridgeContext>());
    if bridge.gc.is_null() {
        return std::ptr::null_mut();
    }
    let element_type = if module_ptr.is_null() {
        None
    } else {
        let module = &*(module_ptr.cast::<Module>());
        jit_resolve_array_element_descriptor(module, type_index)
    };
    let mut gc = (&*bridge.gc).lock();
    let array_ptr = gc.allocate(Array::with_element_type(
        type_index as usize,
        element_type,
        capacity,
    ));
    array_ptr.as_ptr().cast::<()>()
}

/// Resolve an emitted element descriptor id into a self-contained constraint,
/// matching the interpreter's `resolve_element_descriptor`. `AnyValue`, `Ref`,
/// and unresolvable ids map to `None` (dynamic).
unsafe fn jit_resolve_array_element_descriptor(
    module: &Module,
    type_index: u32,
) -> Option<crate::compiler::bytecode::RuntimeTypeDescriptor> {
    use crate::compiler::bytecode::types::COMPLEX_BASE;
    use crate::compiler::bytecode::RuntimeTypeDescriptor as D;
    let descriptor = if type_index < COMPLEX_BASE {
        match type_index {
            0 => D::I32,
            1 => D::F64,
            2 => D::Bool,
            3 => D::String,
            4 => D::Null,
            5 => D::Void,
            6 => return None, // AnyValue == dynamic
            7 => return None, // Ref == accept-all
            _ => return None,
        }
    } else {
        module
            .runtime_types
            .get((type_index - COMPLEX_BASE) as usize)
            .cloned()?
    };
    match descriptor {
        D::AnyValue | D::Ref => None,
        other => Some(other),
    }
}

/// Return an array pointer only when `value` is a GC allocation whose concrete
/// type is `Array`, verified through the GC header. Mirrors the interpreter's
/// `raya_array_ptr_checked` so native and interpreted array identity agree.
#[inline]
unsafe fn jit_array_ptr_checked(value: Value) -> Option<NonNull<crate::vm::object::Array>> {
    if !value.is_ptr() {
        return None;
    }
    let ptr = value.as_ptr::<u8>()?;
    let header = &*crate::vm::gc::header_ptr_from_value_ptr(ptr.as_ptr());
    (header.type_id() == std::any::TypeId::of::<crate::vm::object::Array>())
        .then(|| ptr.cast::<crate::vm::object::Array>())
}

/// Load `array[index]`. Returns the element, or the interpreter-fallback
/// sentinel when the receiver is not an array or the index is out of bounds.
/// Reads never validate element types, matching the interpreter.
unsafe extern "C" fn helper_array_load(array_raw: u64, index: i64, _shared_state: *mut ()) -> u64 {
    let array_value = Value::from_raw(array_raw);
    let Some(array_ptr) = jit_array_ptr_checked(array_value) else {
        return JIT_INTERPRETER_FALLBACK_SENTINEL;
    };
    if index < 0 {
        return JIT_INTERPRETER_FALLBACK_SENTINEL;
    }
    let array = &*array_ptr.as_ptr();
    match array.get(index as usize) {
        Some(value) => value.raw(),
        None => JIT_INTERPRETER_FALLBACK_SENTINEL,
    }
}

/// Store `array[index] = value`, honoring the resolved element constraint.
/// Returns [`JIT_STORE_SUCCESS`] on success and [`JIT_STORE_FALLBACK`] when the
/// receiver is not an array, the index is out of bounds, or the value violates
/// the element constraint (the interpreter then produces the exact error).
unsafe extern "C" fn helper_array_store(
    array_raw: u64,
    index: i64,
    value_raw: u64,
    _shared_state: *mut (),
) -> i8 {
    let array_value = Value::from_raw(array_raw);
    let Some(array_ptr) = jit_array_ptr_checked(array_value) else {
        return JIT_STORE_FALLBACK;
    };
    if index < 0 {
        return JIT_STORE_FALLBACK;
    }
    let array = &mut *array_ptr.as_ptr();
    match array.checked_set(index as usize, Value::from_raw(value_raw)) {
        Ok(()) => JIT_STORE_SUCCESS,
        Err(_) => JIT_STORE_FALLBACK,
    }
}

/// Push `value` onto `array`, honoring the resolved element constraint.
///
/// Growing the backing `Vec` may allocate, so both the array and the pushed
/// value are rooted for the mutation window. Returns [`JIT_STORE_SUCCESS`] or
/// [`JIT_STORE_FALLBACK`].
unsafe extern "C" fn helper_array_push(array_raw: u64, value_raw: u64, shared_state: *mut ()) -> i8 {
    if shared_state.is_null() {
        return JIT_STORE_FALLBACK;
    }
    let bridge = &*(shared_state.cast::<JitRuntimeBridgeContext>());
    let array_value = Value::from_raw(array_raw);
    let element = Value::from_raw(value_raw);
    let Some(array_ptr) = jit_array_ptr_checked(array_value) else {
        return JIT_STORE_FALLBACK;
    };
    // Root the receiver and the element across the push, which may reallocate.
    let Some(_scope) = EphemeralRootScope::open(bridge, &[array_value, element]) else {
        return JIT_STORE_FALLBACK;
    };
    let array = &mut *array_ptr.as_ptr();
    match array.checked_push(element) {
        Ok(_) => JIT_STORE_SUCCESS,
        Err(_) => JIT_STORE_FALLBACK,
    }
}

/// Pop the last element of `array`. Returns the popped value, `null` for an
/// empty array (matching the interpreter), or the fallback sentinel when the
/// receiver is not an array.
unsafe extern "C" fn helper_array_pop(array_raw: u64, _shared_state: *mut ()) -> u64 {
    let array_value = Value::from_raw(array_raw);
    let Some(array_ptr) = jit_array_ptr_checked(array_value) else {
        return JIT_INTERPRETER_FALLBACK_SENTINEL;
    };
    let array = &mut *array_ptr.as_ptr();
    array.pop().unwrap_or(Value::null()).raw()
}

/// Return `array.length`, or [`JIT_ARRAY_LEN_FALLBACK_SENTINEL`] when the
/// receiver is not an array.
unsafe extern "C" fn helper_array_len(array_raw: u64, _shared_state: *mut ()) -> i32 {
    let array_value = Value::from_raw(array_raw);
    let Some(array_ptr) = jit_array_ptr_checked(array_value) else {
        return JIT_ARRAY_LEN_FALLBACK_SENTINEL;
    };
    let array = &*array_ptr.as_ptr();
    i32::try_from(array.len()).unwrap_or(JIT_ARRAY_LEN_FALLBACK_SENTINEL)
}

unsafe extern "C" fn helper_alloc_string(
    data_ptr: *const u8,
    len: usize,
    shared_state: *mut (),
) -> *mut () {
    if shared_state.is_null() || (len != 0 && data_ptr.is_null()) {
        return std::ptr::null_mut();
    }
    let bridge = &*(shared_state.cast::<JitRuntimeBridgeContext>());
    if bridge.gc.is_null() {
        return std::ptr::null_mut();
    }

    let bytes = if len == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(data_ptr, len)
    };
    let text = String::from_utf8_lossy(bytes).into_owned();
    let mut gc = (&*bridge.gc).lock();
    let string_ptr = gc.allocate(RayaString::new(text));
    string_ptr.as_ptr().cast::<()>()
}

unsafe extern "C" fn helper_const_string(
    pool_index: u32,
    module_ptr: *const (),
    shared_state: *mut (),
) -> *mut () {
    if shared_state.is_null() || module_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let bridge = &*shared_state.cast::<JitRuntimeBridgeContext>();
    if bridge.gc.is_null()
        || bridge.constant_string_cache.is_null()
        || bridge.ephemeral_gc_roots.is_null()
    {
        return std::ptr::null_mut();
    }
    let module = &*module_ptr.cast::<Module>();
    let key = (module.checksum, pool_index as usize);
    if let Some(value) = (&*bridge.constant_string_cache).read().get(&key).copied() {
        return value
            .as_ptr::<u8>()
            .map_or(std::ptr::null_mut(), |ptr| ptr.as_ptr().cast::<()>());
    }
    let Some(text) = module.constants.get_string(pool_index).map(str::to_owned) else {
        return std::ptr::null_mut();
    };

    let allocated = {
        let mut gc = (&*bridge.gc).lock();
        let string = gc.allocate(RayaString::new(text));
        let value = Value::from_ptr(
            NonNull::new(string.as_ptr()).expect("GC returned a null string pointer"),
        );
        (&*bridge.ephemeral_gc_roots).write().push(value);
        value
    };
    let published = {
        let mut cache = (&*bridge.constant_string_cache).write();
        *cache.entry(key).or_insert(allocated)
    };
    jit_release_ephemeral_roots(bridge, &[allocated]);
    published
        .as_ptr::<u8>()
        .map_or(std::ptr::null_mut(), |ptr| ptr.as_ptr().cast::<()>())
}

unsafe extern "C" fn helper_safepoint_poll(shared_state: *const ()) {
    if shared_state.is_null() {
        return;
    }
    let bridge = &*(shared_state.cast::<JitRuntimeBridgeContext>());
    if bridge.safepoint.is_null() {
        return;
    }
    let safepoint = &*bridge.safepoint;
    safepoint.poll();
}

unsafe extern "C" fn helper_check_preemption(current_task: *const ()) -> bool {
    if current_task.is_null() {
        return false;
    }
    let task = &*(current_task.cast::<Task>());
    task.is_preempt_requested()
}

unsafe extern "C" fn helper_native_call_dispatch(
    native_id: u16,
    args_ptr: *const u64,
    arg_count: u8,
    shared_state: *mut (),
) -> u64 {
    if !shared_state.is_null() {
        let bridge = &*(shared_state.cast::<JitRuntimeBridgeContext>());
        if !bridge.gc.is_null()
            && !bridge.classes.is_null()
            && !bridge.layouts.is_null()
            && !bridge.class_metadata.is_null()
            && !bridge.resolved_natives.is_null()
        {
            let task_id = if !bridge.task.is_null() {
                (*bridge.task).id()
            } else {
                crate::vm::scheduler::TaskId::from_u64(0)
            };

            let ctx = EngineContext::new(
                &*bridge.gc,
                &*bridge.classes,
                &*bridge.layouts,
                task_id,
                &*bridge.class_metadata,
            );

            let value_args: Vec<Value> = if arg_count == 0 || args_ptr.is_null() {
                Vec::new()
            } else {
                std::slice::from_raw_parts(args_ptr, arg_count as usize)
                    .iter()
                    .copied()
                    .map(|raw| Value::from_raw(raw))
                    .collect()
            };
            let native_args: Vec<raya_sdk::NativeValue> =
                value_args.iter().map(|v| value_to_native(*v)).collect();

            let resolved = (&*bridge.resolved_natives).read();
            match resolved.call(native_id, &ctx, &native_args) {
                NativeCallResult::Value(v) => return native_to_value(v).raw(),
                NativeCallResult::Suspend(io_request) => {
                    if !bridge.io_submit_tx.is_null() {
                        let tx = &*bridge.io_submit_tx;
                        let _ = tx.send(IoSubmission {
                            task_id,
                            request: io_request,
                        });
                    }
                    return JIT_NATIVE_SUSPEND_SENTINEL;
                }
                NativeCallResult::Unhandled | NativeCallResult::Error(_) => {}
            }
        }
    }
    Value::null().raw()
}

unsafe extern "C" fn helper_interpreter_call(
    opcode_raw: u8,
    operand_u64: u64,
    operand_u32: u32,
    receiver_raw: u64,
    args_ptr: *const u64,
    arg_count: u16,
    module_ptr: *const (),
    shared_state: *mut (),
) -> u64 {
    if shared_state.is_null() || module_ptr.is_null() {
        return JIT_INTERPRETER_FALLBACK_SENTINEL;
    }
    let bridge = &*(shared_state.cast::<JitRuntimeBridgeContext>());
    let Some(opcode) = Opcode::from_u8(opcode_raw) else {
        return JIT_INTERPRETER_FALLBACK_SENTINEL;
    };
    let module = &*(module_ptr.cast::<Module>());
    let Some(task) = (!bridge.task_arc.is_null()).then(|| unsafe { &*bridge.task_arc }) else {
        return JIT_INTERPRETER_FALLBACK_SENTINEL;
    };
    let Some(mut interpreter) = jit_build_interpreter(bridge) else {
        return JIT_INTERPRETER_FALLBACK_SENTINEL;
    };

    let args: Vec<Value> = if arg_count == 0 || args_ptr.is_null() {
        Vec::new()
    } else {
        std::slice::from_raw_parts(args_ptr, arg_count as usize)
            .iter()
            .copied()
            .map(|raw| unsafe { Value::from_raw(raw) })
            .collect()
    };

    let mut stack = Stack::new();
    if std::env::var("RAYA_JIT_DEBUG_CALLS").is_ok() {
        eprintln!(
            "jit interpreter_call: opcode={opcode:?} operand_u32={operand_u32} operand_u64={operand_u64:#x} receiver=0x{receiver_raw:016x} argc={arg_count}"
        );
    }
    match opcode {
        Opcode::Call => {
            if operand_u32 == 0xFFFF_FFFF {
                if stack.push(Value::from_raw(receiver_raw)).is_err() {
                    return JIT_INTERPRETER_EXCEPTION_SENTINEL;
                }
            }
            for arg in &args {
                if stack.push(*arg).is_err() {
                    return JIT_INTERPRETER_EXCEPTION_SENTINEL;
                }
            }
        }
        Opcode::CallMethodExact
        | Opcode::OptionalCallMethodExact
        | Opcode::CallMethodShape
        | Opcode::OptionalCallMethodShape
        | Opcode::ConstructType
        | Opcode::CallSuper => {
            if stack.push(Value::from_raw(receiver_raw)).is_err() {
                return JIT_INTERPRETER_EXCEPTION_SENTINEL;
            }
            for arg in &args {
                if stack.push(*arg).is_err() {
                    return JIT_INTERPRETER_EXCEPTION_SENTINEL;
                }
            }
        }
        Opcode::CallConstructor | Opcode::CallStatic => {
            for arg in &args {
                if stack.push(*arg).is_err() {
                    return JIT_INTERPRETER_EXCEPTION_SENTINEL;
                }
            }
        }
        _ => {
            return JIT_INTERPRETER_FALLBACK_SENTINEL;
        }
    }

    let mut code = vec![opcode_raw];
    match opcode {
        Opcode::Call
        | Opcode::CallMethodExact
        | Opcode::OptionalCallMethodExact
        | Opcode::CallStatic => {
            code.extend_from_slice(&operand_u32.to_le_bytes());
            code.extend_from_slice(&arg_count.to_le_bytes());
        }
        Opcode::CallConstructor | Opcode::CallSuper => {
            code.extend_from_slice(&operand_u32.to_le_bytes());
            code.extend_from_slice(&arg_count.to_le_bytes());
        }
        Opcode::ConstructType => {
            code.extend_from_slice(&(operand_u32 as u16).to_le_bytes());
            code.push(arg_count as u8);
        }
        Opcode::CallMethodShape | Opcode::OptionalCallMethodShape => {
            code.extend_from_slice(&operand_u64.to_le_bytes());
            code.extend_from_slice(&(operand_u32 as u16).to_le_bytes());
            code.extend_from_slice(&arg_count.to_le_bytes());
        }
        _ => {}
    }

    let mut ip = 1usize;
    match interpreter.exec_call_ops(&mut stack, &mut ip, &code, module, task, opcode) {
        crate::vm::interpreter::OpcodeResult::Continue => {
            let value = stack.pop().unwrap_or_else(|_| Value::null());
            if std::env::var("RAYA_JIT_DEBUG_CALLS").is_ok() {
                eprintln!("jit interpreter_call continue: opcode={opcode:?} result={value:?}");
            }
            value.raw()
        }
        crate::vm::interpreter::OpcodeResult::Return(value) => {
            if std::env::var("RAYA_JIT_DEBUG_CALLS").is_ok() {
                eprintln!("jit interpreter_call return: opcode={opcode:?} result={value:?}");
            }
            value.raw()
        }
        crate::vm::interpreter::OpcodeResult::Suspend(_) => JIT_INTERPRETER_FALLBACK_SENTINEL,
        crate::vm::interpreter::OpcodeResult::Error(error) => {
            if std::env::var("RAYA_JIT_DEBUG_CALLS").is_ok() {
                eprintln!("jit interpreter_call error: opcode={opcode:?} error={error}");
            }
            jit_raise_vm_error(bridge, error);
            JIT_INTERPRETER_EXCEPTION_SENTINEL
        }
        crate::vm::interpreter::OpcodeResult::PushFrame {
            func_id,
            arg_count,
            is_closure,
            closure_val,
            module: callee_module,
            return_action,
        } => {
            let callee_module = callee_module.unwrap_or_else(|| Arc::new(module.clone()));
            if !jit_function_is_sync_safe(callee_module.as_ref(), func_id, &mut FxHashSet::default()) {
                return JIT_INTERPRETER_FALLBACK_SENTINEL;
            }
            match jit_execute_sync_frame(
                &mut interpreter,
                bridge,
                &mut stack,
                callee_module,
                func_id,
                arg_count,
                is_closure,
                closure_val,
                return_action,
            ) {
                JitNestedCallResult::Value(value) => {
                    if std::env::var("RAYA_JIT_DEBUG_CALLS").is_ok() {
                        eprintln!(
                            "jit interpreter_call nested: opcode={opcode:?} result={value:?}"
                        );
                    }
                    value.raw()
                }
                JitNestedCallResult::Fallback => {
                    if std::env::var("RAYA_JIT_DEBUG_CALLS").is_ok() {
                        eprintln!("jit interpreter_call nested fallback: opcode={opcode:?}");
                    }
                    JIT_INTERPRETER_FALLBACK_SENTINEL
                }
                JitNestedCallResult::Exception => {
                    if std::env::var("RAYA_JIT_DEBUG_CALLS").is_ok() {
                        eprintln!("jit interpreter_call nested exception: opcode={opcode:?}");
                    }
                    JIT_INTERPRETER_EXCEPTION_SENTINEL
                }
            }
        }
    }
}

fn jit_add_ephemeral_roots(bridge: &JitRuntimeBridgeContext, values: &[Value]) -> bool {
    if bridge.ephemeral_gc_roots.is_null() {
        return false;
    }
    let mut roots = unsafe { &*bridge.ephemeral_gc_roots }.write();
    roots.extend(values.iter().copied().filter(Value::is_heap_allocated));
    true
}

/// RAII guard that keeps a set of `Value`s reachable for the duration of an
/// allocation helper window.
///
/// JIT-compiled native frames publish empty stack maps, so a GC triggered by an
/// allocation inside a helper cannot see operands that live only in machine
/// registers. Every helper that holds a live `Value` across a `gc.allocate`
/// call must open a scope so those inputs are treated as roots until the helper
/// returns and the result is safely published. The guard releases exactly the
/// roots it added on drop, including on the error/early-return paths.
struct EphemeralRootScope<'a> {
    bridge: &'a JitRuntimeBridgeContext,
    rooted: Vec<Value>,
    active: bool,
}

impl<'a> EphemeralRootScope<'a> {
    /// Open a scope rooting `values`. Returns `None` when the root set is
    /// unavailable (a null bridge field), so callers fail closed rather than
    /// allocating with unprotected operands.
    fn open(bridge: &'a JitRuntimeBridgeContext, values: &[Value]) -> Option<Self> {
        if !jit_add_ephemeral_roots(bridge, values) {
            return None;
        }
        Some(Self {
            bridge,
            rooted: values.to_vec(),
            active: true,
        })
    }
}

impl Drop for EphemeralRootScope<'_> {
    fn drop(&mut self) {
        if self.active {
            jit_release_ephemeral_roots(self.bridge, &self.rooted);
            self.active = false;
        }
    }
}

fn jit_release_ephemeral_roots(bridge: &JitRuntimeBridgeContext, values: &[Value]) {
    if bridge.ephemeral_gc_roots.is_null() {
        return;
    }
    let mut roots = unsafe { &*bridge.ephemeral_gc_roots }.write();
    for value in values.iter().rev().filter(|value| value.is_heap_allocated()) {
        if let Some(index) = roots.iter().rposition(|candidate| candidate == value) {
            roots.swap_remove(index);
        }
    }
}

unsafe extern "C" fn helper_string_concat(
    left_raw: u64,
    right_raw: u64,
    shared_state: *mut (),
) -> u64 {
    if shared_state.is_null() {
        return JIT_INTERPRETER_FALLBACK_SENTINEL;
    }
    let bridge = &*shared_state.cast::<JitRuntimeBridgeContext>();
    if bridge.gc.is_null() {
        return JIT_INTERPRETER_FALLBACK_SENTINEL;
    }

    let values = [Value::from_raw(left_raw), Value::from_raw(right_raw)];
    if !jit_add_ephemeral_roots(bridge, &values) {
        return JIT_INTERPRETER_FALLBACK_SENTINEL;
    }
    let left = value_to_string(values[0]);
    let right = value_to_string(values[1]);
    let result = {
        let mut gc = (&*bridge.gc).lock();
        let string = gc.allocate(RayaString::new(format!("{}{}", left, right)));
        Value::from_ptr(NonNull::new(string.as_ptr()).expect("GC returned a null string pointer"))
    };
    jit_release_ephemeral_roots(bridge, &values);
    result.raw()
}

unsafe extern "C" fn helper_string_len(string_raw: u64, _shared_state: *mut ()) -> i32 {
    let value = Value::from_raw(string_raw);
    let Some(string_ptr) = raya_string_ptr_checked(value) else {
        if std::env::var("RAYA_JIT_DEBUG_CALLS").is_ok() {
            eprintln!("jit string_len fallback: raw=0x{string_raw:016x} value={value:?}");
        }
        return JIT_STRING_LEN_FALLBACK_SENTINEL;
    };
    let string = &*string_ptr.as_ptr();
    if std::env::var("RAYA_JIT_DEBUG_CALLS").is_ok() {
        eprintln!("jit string_len: len={} value={value:?}", string.len());
    }
    i32::try_from(string.len()).unwrap_or(JIT_STRING_LEN_FALLBACK_SENTINEL)
}

unsafe extern "C" fn helper_generic_equals(
    left_raw: u64,
    right_raw: u64,
    _shared_state: *mut (),
) -> bool {
    values_equal(Value::from_raw(left_raw), Value::from_raw(right_raw))
}

unsafe extern "C" fn helper_string_compare(
    left_raw: u64,
    right_raw: u64,
    _shared_state: *mut (),
) -> i8 {
    match compare_strings(Value::from_raw(left_raw), Value::from_raw(right_raw)) {
        Some(std::cmp::Ordering::Less) => -1,
        Some(std::cmp::Ordering::Equal) => 0,
        Some(std::cmp::Ordering::Greater) => 1,
        None => 2,
    }
}

unsafe extern "C" fn helper_value_to_string(value_raw: u64, shared_state: *mut ()) -> u64 {
    if shared_state.is_null() {
        return JIT_INTERPRETER_FALLBACK_SENTINEL;
    }
    let bridge = &*shared_state.cast::<JitRuntimeBridgeContext>();
    if bridge.gc.is_null() {
        return JIT_INTERPRETER_FALLBACK_SENTINEL;
    }

    let value = Value::from_raw(value_raw);
    if !jit_add_ephemeral_roots(bridge, &[value]) {
        return JIT_INTERPRETER_FALLBACK_SENTINEL;
    }
    let text = value_to_string(value);
    let result = {
        let mut gc = (&*bridge.gc).lock();
        let string = gc.allocate(RayaString::new(text));
        Value::from_ptr(NonNull::new(string.as_ptr()).expect("GC returned a null string pointer"))
    };
    jit_release_ephemeral_roots(bridge, &[value]);
    result.raw()
}

// ---------------------------------------------------------------------------
// RefCell helpers (D4.4)
//
// NOT YET REACHABLE, but for a narrower reason than `BindMethod`.
//
// The lifter already handles all three opcodes correctly (`lifter.rs:1604-1626`),
// emitting `JitInstr::NewRefCell` / `LoadRefCell` / `StoreRefCell` with the right
// stack effects — including `StoreRefCell`'s net -2, which pushes nothing. What is
// missing is the *native* half: `jit/backend/cranelift/lowering.rs` has no RefCell
// arm at all, so nothing reaches these helpers, and they are deliberately absent
// from the trampoline table as well. `BindMethod` is the opposite problem — the
// lifter emits nothing for it at all.
//
// So RefCell is lifter-ready and only needs a Cranelift arm plus a differential
// test. Do not read this as "the whole path is absent". That is stated here because unwired helpers are exactly what I
// misread during the D4.3 audit — I trusted a capability classification and
// assumed reachability, and three of the object helpers turned out to be wired
// while a fourth was referenced nowhere.
//
// They are written and tested now so the semantics are settled before anything can
// call them. Wiring is the next slice, and `NewRefCell`/`LoadRefCell`/`StoreRefCell`
// stay `Rejected` until it is done.
//
// **The receiver check is deliberately weak.** The interpreter's RefCell handlers
// test `is_ptr()` only — never the GC-header TypeId — and will reinterpret any heap
// value as a RefCell (ALY-54). These helpers reproduce that exactly rather than
// "fixing" it, because a helper that type-checks properly would diverge from the
// interpreter, which is the opposite of the goal. Do not strengthen these checks
// without fixing the interpreter in the same change.

// ---------------------------------------------------------------------------
// Task helpers (D4.6)
// ---------------------------------------------------------------------------

/// `Await` one of the paths that do not suspend.
///
/// Three paths, and this covers the two that can be leaf operations:
///
/// 1. **The value is not a task id** — pushed straight back, execution continues.
///    This is JS-like `await` normalisation. Note `Value::as_u64` is **tag-gated**
///    (`is_u64()` then `PAYLOAD_MASK`), so `await` on a boxed `i32 42` returns
///    `None` and is NOT read as task id 42. The test below pins that.
/// 2. **The awaited task is already `Completed`** — its result is returned.
/// 3. **Cancelled or still pending** — returns the interpreter-fallback sentinel, so
///    the interpreter raises `"Awaited task {:?} cancelled"` or suspends
///    respectively. A leaf helper cannot raise, and it must not invent a value for
///    either case.
///
/// Path 3 is not an oversight: `JitSuspendReason` has no `AwaitTask` variant, so the
/// JIT cannot express an await suspension at all. Its only suspension is
/// `InterpreterBoundary`. See the D4.6 spec.
///
/// NOT YET LOWERED. See the note on the RefCell helpers.
unsafe extern "C" fn helper_await_task(value_raw: u64, shared_state: *mut ()) -> u64 {
    let bridge = match NonNull::new(shared_state.cast::<JitRuntimeBridgeContext>()) {
        Some(ptr) => &*ptr.as_ptr(),
        None => return JIT_INTERPRETER_FALLBACK_SENTINEL,
    };
    let value = Value::from_raw(value_raw);

    // Path 1. Deliberately `Value::as_u64()` and not a payload test: the accessor is
    // tag-gated, and reimplementing it by payload would misread any value whose
    // payload looks like a plausible task id.
    let Some(task_id_u64) = value.as_u64() else {
        return value_raw;
    };

    if bridge.tasks.is_null() {
        return JIT_INTERPRETER_FALLBACK_SENTINEL;
    }
    let task_id = crate::vm::scheduler::TaskId::from_u64(task_id_u64);
    let tasks = (&*bridge.tasks).read();
    let Some(awaited) = tasks.get(&task_id).cloned() else {
        // Unknown task id: let the interpreter produce its own error rather than
        // guessing at one here.
        return JIT_INTERPRETER_FALLBACK_SENTINEL;
    };
    drop(tasks);

    if awaited.is_cancelled() {
        // The interpreter raises "Awaited task {:?} cancelled" after marking the
        // rejection observed. Marking it is a visible side effect we must not
        // duplicate, so hand back and let the interpreter do it exactly once.
        return JIT_INTERPRETER_FALLBACK_SENTINEL;
    }

    if awaited.state() == crate::vm::scheduler::TaskState::Completed {
        // Path 2. `result()` may be unset on a completed task; the interpreter uses
        // `unwrap_or(Value::null())`, and so must this.
        return awaited.result().unwrap_or(Value::null()).raw();
    }

    // Path 3: still pending. The interpreter suspends; we cannot, so exit.
    JIT_INTERPRETER_FALLBACK_SENTINEL
}

// ---------------------------------------------------------------------------
// Dynamic NodeCompat helpers (D4.7)
// ---------------------------------------------------------------------------

/// `DynGetKeyed` for the views that do not need field-index resolution.
///
/// `JSView::Str` and `JSView::Arr` are ordinary leaf reads and are handled here.
/// **`JSView::Struct` and everything else return the interpreter-fallback
/// sentinel**, deliberately: `Struct` field lookup goes through
/// `field_index_for_value`, which falls back to `structural_object_shapes` — a
/// registry `JitRuntimeBridgeContext` does not carry and `SharedVmState` does not
/// have either (it has `structural_layout_shapes` instead). Reproducing that
/// resolution here would diverge from the interpreter rather than match it, so the
/// interpreter keeps that part. See the D4.7 spec.
///
/// Key parsing calls the interpreter's own `dyn_key_parts`, and the view split
/// uses its own `js_classify`. Neither is reimplemented here: a hand-rolled key
/// parser or view dispatch is exactly the shape of divergence that ships.
///
/// NOT YET LOWERED. See the note on the RefCell helpers.
unsafe extern "C" fn helper_dyn_get_keyed(
    object_raw: u64,
    key_raw: u64,
    shared_state: *mut (),
) -> u64 {
    use crate::vm::json::view::{js_classify, JSView};

    let bridge = match NonNull::new(shared_state.cast::<JitRuntimeBridgeContext>()) {
        Some(ptr) => &*ptr.as_ptr(),
        None => return JIT_INTERPRETER_FALLBACK_SENTINEL,
    };
    if bridge.gc.is_null() {
        return JIT_INTERPRETER_FALLBACK_SENTINEL;
    }

    // The interpreter raises on a malformed key, so a helper cannot invent an answer
    // for one.
    let (key_str, array_index) =
        match crate::vm::interpreter::opcodes::types::dyn_key_parts(Value::from_raw(key_raw)) {
            Ok(parts) => parts,
            Err(_) => return JIT_INTERPRETER_FALLBACK_SENTINEL,
        };

    match js_classify(Value::from_raw(object_raw)) {
        JSView::Arr(ptr) => {
            let array = unsafe { &*ptr };
            match array_index {
                Some(index) => array.get(index).unwrap_or(Value::null()).raw(),
                None => Value::null().raw(),
            }
        }
        JSView::Str(ptr) => {
            let string = unsafe { &*ptr };
            if let Some(index) = array_index {
                // Each character is a freshly allocated `RayaString`, with `string`
                // held live across the allocation, so root it.
                let Some(character) = string.data.chars().nth(index) else {
                    // Out of range is a legitimate answer, not a failure.
                    return Value::null().raw();
                };
                let string_value = Value::from_raw(object_raw);
                let Some(_scope) = EphemeralRootScope::open(bridge, &[string_value]) else {
                    return JIT_INTERPRETER_FALLBACK_SENTINEL;
                };
                let mut gc = (&*bridge.gc).lock();
                let allocated = gc.allocate(crate::vm::object::RayaString::new(character.to_string()));
                let pointer = NonNull::new(allocated.as_ptr()).unwrap();
                Value::from_ptr(pointer).raw()
            } else if key_str.as_deref() == Some("length") {
                Value::i32(string.data.chars().count() as i32).raw()
            } else {
                Value::null().raw()
            }
        }
        // `Struct` needs the registry the bridge lacks; everything else is the
        // interpreter's business.
        _ => JIT_INTERPRETER_FALLBACK_SENTINEL,
    }
}

// ---------------------------------------------------------------------------
// Closure helpers (D4.4)
//
// NOT YET LOWERED. `lowering.rs` has no `MakeClosure` arm and the opcode stays
// `Rejected`. The helpers are written and tested first so the semantics are settled
// before anything can call them — the same order the RefCell helpers used.

/// Bind a method on a nominal object, producing a `BoundMethod`.
///
/// This is the body that `7d211cd` removed from `helper_object_get_field`: the
/// vtable lookup and `BoundMethod` allocation. It was dead there — that helper
/// constructed its binding as `Field(...)` and matched on it, so the `Method` arm
/// could never run. It is needed here, with the class-registry lookups actually
/// wired. The machinery was not wrong, only unreachable and in the wrong place.
///
/// The interpreter's handler (`vm/interpreter/opcodes/objects.rs`) does, in order:
/// `ensure_object_receiver`, then `nominal_type_id_usize`, then
/// `classes.get_class`, then `class.vtable.get_method`, then build and allocate.
/// All four failure points return `0` here rather than being raised, because a
/// leaf helper cannot raise a catchable error; the lowering turns a zero result
/// into the interpreter boundary exit, which produces the real diagnostic:
///   * receiver is not an object -> `TypeError("Expected Object receiver for method binding")`
///   * structural object          -> `TypeError("Cannot bind method on structural object value")`
///   * unknown nominal type id    -> `RuntimeError("Invalid nominal type id: N")`
///   * no method in that slot     -> `RuntimeError("Invalid method slot: N for class X")`
///
/// The receiver is a live value across the allocation, so it is rooted with
/// `EphemeralRootScope` — native stack maps are empty.
///
/// NOT YET LOWERED. See the note on the RefCell helpers.
unsafe extern "C" fn helper_bind_method(
    object_raw: u64,
    method_slot: u32,
    shared_state: *mut (),
) -> u64 {
    let bridge = match NonNull::new(shared_state.cast::<JitRuntimeBridgeContext>()) {
        Some(ptr) => &*ptr.as_ptr(),
        None => return 0,
    };
    if bridge.gc.is_null() || bridge.classes.is_null() {
        return 0;
    }

    let object_value = Value::from_raw(object_raw);
    // Deliberately the interpreter's weak `is_ptr()` check, not a TypeId
    // comparison -- see ALY-54. A stronger check here would diverge.
    if !object_value.is_ptr() {
        return 0;
    }
    let Some(object_ptr) = object_value.as_ptr::<crate::vm::object::Object>() else {
        return 0;
    };
    let object = &*object_ptr.as_ptr();

    let Some(nominal_type_id) = object.nominal_type_id_usize() else {
        // Structural object -- the interpreter's "Cannot bind method on structural
        // object value".
        return 0;
    };

    let (func_id, method_module) = {
        let classes = (&*bridge.classes).read();
        let Some(class) = classes.get_class(nominal_type_id) else {
            return 0;
        };
        let Some(func_id) = class.vtable.get_method(method_slot as usize) else {
            return 0;
        };
        (func_id, class.module.clone())
    };

    // Root the receiver across the allocation: it is a live heap value and native
    // stack maps are empty.
    let Some(_scope) = EphemeralRootScope::open(bridge, &[object_value]) else {
        return 0;
    };

    let bound = crate::vm::object::BoundMethod {
        receiver: object_value,
        func_id,
        module: method_module,
    };
    let mut gc = (&*bridge.gc).lock();
    let ptr = gc.allocate(bound);
    Value::from_ptr(NonNull::new(ptr.as_ptr()).unwrap()).raw()
}

/// Read one capture of the closure currently executing.
///
/// The active closure is NOT carried in the JIT frame. It is read from the task,
/// exactly as the interpreter does: `Task::current_closure()` returns
/// `closure_stack.last()`. That is the same path `helper_make_closure` uses for
/// the current module, and it is deliberate — reading the interpreter's own
/// accessor is what makes this faithful, including the edge cases. In particular
/// `closure_stack.last()` is the *innermost* closure, so a `LoadCaptured` in
/// natively-compiled code that is not a closure body would read whatever closure is
/// on top. The interpreter behaves identically, so reproducing it is the correct
/// outcome, not a bug to guard against here (ALY-52/ALY-54 are the places that
/// would want fixing, separately).
///
/// Both interpreter failure modes are returned as the fallback sentinel rather than
/// raised, because a leaf helper cannot raise a catchable error:
///   * no active closure      -> `RuntimeError("LoadCaptured without active closure")`
///   * capture index too high -> `RuntimeError("Capture index N out of bounds")`
/// The lowering turns either into the interpreter boundary exit, which raises them
/// for real.
///
/// NOT YET LOWERED. See the note on the RefCell helpers.
unsafe extern "C" fn helper_load_captured(index: u32, shared_state: *mut ()) -> u64 {
    let bridge = match NonNull::new(shared_state.cast::<JitRuntimeBridgeContext>()) {
        Some(ptr) => &*ptr.as_ptr(),
        None => return JIT_INTERPRETER_FALLBACK_SENTINEL,
    };
    if bridge.task_arc.is_null() {
        return JIT_INTERPRETER_FALLBACK_SENTINEL;
    }
    let Some(closure_val) = (&*bridge.task_arc).current_closure() else {
        // Matches the interpreter's "LoadCaptured without active closure".
        return JIT_INTERPRETER_FALLBACK_SENTINEL;
    };
    let Some(ptr) = closure_val.as_ptr::<crate::vm::object::Closure>() else {
        return JIT_INTERPRETER_FALLBACK_SENTINEL;
    };
    let closure = &*ptr.as_ptr();
    // `get_captured` is bounds-checked and returns None, which is the interpreter's
    // "Capture index N out of bounds".
    match closure.get_captured(index as usize) {
        Some(value) => value.raw(),
        None => JIT_INTERPRETER_FALLBACK_SENTINEL,
    }
}

/// Write one capture of the closure currently executing.
///
/// Reads the active closure from the task via `Task::current_closure()`, exactly as
/// the interpreter does and as `helper_load_captured` does — the innermost closure
/// on `closure_stack`, which is faithful to the handler including its edge cases.
///
/// Both interpreter failure modes return `JIT_STORE_FALLBACK` rather than being
/// raised, since a leaf helper cannot raise a catchable error:
///   * no active closure  -> `RuntimeError("StoreCaptured without active closure")`
///   * capture index high -> the `set_captured` error string, a `RuntimeError`
///
/// The check happens before the write, so a fallback return means the closure was
/// not modified and an interpreter fallback cannot double-apply the store.
///
/// NOT YET LOWERED. See the note on the RefCell helpers.
unsafe extern "C" fn helper_store_captured(
    index: u32,
    value_raw: u64,
    shared_state: *mut (),
) -> i8 {
    let bridge = match NonNull::new(shared_state.cast::<JitRuntimeBridgeContext>()) {
        Some(ptr) => &*ptr.as_ptr(),
        None => return JIT_STORE_FALLBACK,
    };
    if bridge.task_arc.is_null() {
        return JIT_STORE_FALLBACK;
    }
    let Some(closure_val) = (&*bridge.task_arc).current_closure() else {
        // Matches the interpreter's "StoreCaptured without active closure".
        return JIT_STORE_FALLBACK;
    };
    let Some(ptr) = closure_val.as_ptr::<crate::vm::object::Closure>() else {
        return JIT_STORE_FALLBACK;
    };
    let closure = &mut *ptr.as_ptr();
    match closure.set_captured(index as usize, Value::from_raw(value_raw)) {
        Ok(()) => JIT_STORE_SUCCESS,
        Err(_) => JIT_STORE_FALLBACK,
    }
}

/// Patch one capture slot of an existing closure.
///
/// This is how recursive closures are wired up: `MakeClosure` runs first, then
/// `SetClosureCapture` writes the closure into its own slot.
///
/// `JIT_STORE_FALLBACK` means nothing was mutated. That covers a non-pointer
/// receiver, which the interpreter reports as `TypeError("Expected closure")`, and
/// a capture index out of range, which `Closure::set_captured` reports as an error
/// string the interpreter turns into a `RuntimeError`. Both are handed back rather
/// than raised, because a leaf helper cannot raise a catchable error — the lowering
/// turns a fallback into the interpreter boundary exit.
///
/// NOT YET LOWERED. See the note on the RefCell helpers.
unsafe extern "C" fn helper_set_closure_capture(
    closure_raw: u64,
    index: u32,
    value_raw: u64,
    _shared_state: *mut (),
) -> i8 {
    let closure_value = Value::from_raw(closure_raw);
    // Deliberately the interpreter's weak `is_ptr()` check, not a TypeId
    // comparison. See ALY-54: strengthening it here would diverge from the
    // interpreter rather than fix anything.
    if !closure_value.is_ptr() {
        return JIT_STORE_FALLBACK;
    }
    let Some(ptr) = closure_value.as_ptr::<crate::vm::object::Closure>() else {
        return JIT_STORE_FALLBACK;
    };
    let closure = &mut *ptr.as_ptr();
    // `set_captured` bounds-checks, so an out-of-range index lands here rather than
    // writing past the capture vector.
    match closure.set_captured(index as usize, Value::from_raw(value_raw)) {
        Ok(()) => JIT_STORE_SUCCESS,
        Err(_) => JIT_STORE_FALLBACK,
    }
}

// Unused until the Cranelift lowering for MakeClosure exists; see the note on the
// RefCell helpers above for why that is marked rather than left to warn.
unsafe extern "C" fn helper_make_closure(
    func_id: u32,
    captures_ptr: *const u64,
    capture_count: u32,
    shared_state: *mut (),
) -> u64 {
    let bridge = match NonNull::new(shared_state.cast::<JitRuntimeBridgeContext>()) {
        Some(ptr) => &*ptr.as_ptr(),
        None => return 0,
    };
    if bridge.gc.is_null() || bridge.task_arc.is_null() {
        return 0;
    }

    // `Closure::with_module` needs an `Arc<Module>`, which cannot be reconstructed
    // from the raw `*const Module` the lowering has. The bridge carries the current
    // task, so the module comes from there — the same source the interpreter uses.
    let module = {
        let task_arc = &*bridge.task_arc;
        task_arc.current_module()
    };

    let captures: Vec<Value> = if captures_ptr.is_null() || capture_count == 0 {
        Vec::new()
    } else {
        (0..capture_count as usize)
            .map(|i| Value::from_raw(*captures_ptr.add(i)))
            .collect()
    };

    // Every capture is a live value across the allocation, and native stack maps are
    // empty, so root them all — `EphemeralRootScope::open` filters to heap values
    // itself. Fail closed to null, matching `helper_alloc_object`.
    let Some(_scope) = EphemeralRootScope::open(bridge, &captures) else {
        return 0;
    };

    let closure = crate::vm::object::Closure::with_module(func_id as usize, captures, module);
    let mut gc = (&*bridge.gc).lock();
    let ptr = gc.allocate(closure);
    Value::from_ptr(NonNull::new(ptr.as_ptr()).unwrap()).raw()
}

/// Allocate a RefCell holding `initial_raw`.
///
/// Returns null when the root set is unavailable, so the allocation cannot happen
/// with an unprotected operand: native stack maps are empty, so a collection during
/// this allocation would not see `initial_raw`.
///
/// Null is the codebase's convention for an allocating helper failing —
/// `helper_alloc_object` does the same, and the `NewObject` lowering tests
/// `icmp_imm(Equal, ptr, 0)`. This originally returned `u64::MAX`, a bespoke
/// sentinel sitting in tagged-pointer space; matching the convention means the
/// lowering can reuse the established `is_null` test and there is one fewer magic
/// value to reason about.
// Unused until the Cranelift lowering for RefCell exists. Marked rather than left
// to warn on every build: these are deliberately unreachable, and the alternative
// -- wiring them now without a differential test -- is the thing D4.3 got wrong.
unsafe extern "C" fn helper_new_refcell(initial_raw: u64, shared_state: *mut ()) -> u64 {
    let bridge = match NonNull::new(shared_state.cast::<JitRuntimeBridgeContext>()) {
        Some(ptr) => &*ptr.as_ptr(),
        None => return 0,
    };
    if bridge.gc.is_null() {
        return 0;
    }
    let initial = Value::from_raw(initial_raw);
    // Root the initial value across the allocation, exactly as `helper_alloc_array`
    // roots its operands, and fail closed if that is not possible.
    let Some(_scope) = EphemeralRootScope::open(bridge, &[initial]) else {
        return 0;
    };
    let mut gc = (&*bridge.gc).lock();
    let ptr = gc.allocate(crate::vm::object::RefCell::new(initial));
    Value::from_ptr(NonNull::new(ptr.as_ptr()).unwrap()).raw()
}

/// Read a RefCell's value.
///
/// Returns the interpreter-fallback sentinel for a non-pointer receiver, matching
/// the interpreter's `TypeError("Expected RefCell")` by handing the error back
/// rather than raising it: a leaf helper cannot raise a catchable error.
unsafe extern "C" fn helper_load_refcell(refcell_raw: u64, _shared_state: *mut ()) -> u64 {
    let refcell_value = Value::from_raw(refcell_raw);
    if !refcell_value.is_ptr() {
        return JIT_INTERPRETER_FALLBACK_SENTINEL;
    }
    let ptr = refcell_value.as_ptr::<crate::vm::object::RefCell>();
    match ptr {
        Some(ptr) => (&*ptr.as_ptr()).get().raw(),
        None => JIT_INTERPRETER_FALLBACK_SENTINEL,
    }
}

/// Write `value_raw` into a RefCell.
///
/// `JIT_STORE_FALLBACK` is returned for a non-pointer receiver. The check happens
/// before the write, so a fallback return means nothing was mutated.
// Unused until the Cranelift lowering for RefCell exists. Marked rather than left
// to warn on every build: these are deliberately unreachable, and the alternative
// -- wiring them now without a differential test -- is the thing D4.3 got wrong.
unsafe extern "C" fn helper_store_refcell(
    refcell_raw: u64,
    value_raw: u64,
    _shared_state: *mut (),
) -> i8 {
    let refcell_value = Value::from_raw(refcell_raw);
    if !refcell_value.is_ptr() {
        return JIT_STORE_FALLBACK;
    }
    let value = Value::from_raw(value_raw);
    let Some(ptr) = refcell_value.as_ptr::<crate::vm::object::RefCell>() else {
        return JIT_STORE_FALLBACK;
    };
    (&mut *ptr.as_ptr()).set(value);
    JIT_STORE_SUCCESS
}

/// Exact field load.
///
/// This is **not** an implementation of the interpreter's `LoadFieldExact`. It
/// omits two things the interpreter does, and the omission is why the opcode is
/// `Rejected` in `jit/capability.rs` rather than `HelperExact`:
///
///  * it does not consult `__node_compat_descriptor` accessors, so a field
///    installed with a `get` descriptor returns the raw slot instead of invoking
///    the getter;
///  * it does not unwrap a proxy receiver, so `jit_object_ptr_checked` returns
///    `None` and the load yields null where the interpreter reads the target.
///
/// It also does not pass through `helper_object_get_shape_field`, which resolves
/// slots through a shape adapter. Do not wire this helper to `LoadFieldExact`
/// without all three, and do not promote the opcode on the strength of this
/// comment — see /workspace/specs/2026-10-03-raya-d4-fixed-layout-objects.md.
unsafe extern "C" fn helper_object_get_field(
    object_raw: u64,
    expected_slot: u32,
    expected_layout_generation: u64,
    func_id: u32,
    module_ptr: *const (),
    shared_state: *mut (),
) -> u64 {
    if shared_state.is_null() || module_ptr.is_null() {
        return Value::null().raw();
    }
    let bridge = &*(shared_state.cast::<JitRuntimeBridgeContext>());
    if bridge.classes.is_null() || bridge.gc.is_null() || bridge.code_cache.is_null() {
        return Value::null().raw();
    }
    let code_cache = &*bridge.code_cache;
    if !code_cache.layout_generation_matches(
        crate::jit::runtime::code_cache::LayoutDependency::AnyLayout,
        expected_layout_generation,
    ) {
        return JIT_LAYOUT_GUARD_FALLBACK_SENTINEL;
    }

    let object_val = Value::from_raw(object_raw);
    let Some(object_ptr) = jit_object_ptr_checked(object_val) else {
        return Value::null().raw();
    };
    let object = &*object_ptr.as_ptr();
    let _ = bridge;
    let _ = module_ptr;
    let _ = func_id;
    let _ = object_val;

    // A raw field read, matching the interpreter's `LoadFieldExact`. See
    // `helper_object_set_field` for why there is no adapter resolution.
    //
    // The `Method`, `Dynamic` and `Missing` arms this used to carry were
    // unreachable, since `binding` was constructed as `Field`. The `Method` arm
    // also allocated a `BoundMethod` under the GC lock, which a plain field read
    // has no reason to do. Gone.
    object
        .get_field(expected_slot as usize)
        .unwrap_or(Value::null())
        .raw()
}

/// Exact field store.
///
/// Unwired: nothing in `jit/backend/cranelift/lowering.rs` references
/// `HELPER_OBJECT_SET_FIELD_OFFSET`, which is consistent with `StoreFieldExact`
/// being `InterpreterBoundary`.
///
/// It is also **not** a faithful implementation of that opcode. The interpreter's
/// `StoreFieldExact` consults `__node_compat_descriptor` and, for a setter-backed
/// field, invokes the setter as a callable frame plus the writability checks
/// (`vm/interpreter/opcodes/objects.rs:763-800`). That is not expressible as a
/// leaf helper returning a value, which is why the classification is correct and
/// why this helper must never be wired to close the gap. `StoreFieldExact` stays
/// `InterpreterBoundary` permanently.
unsafe extern "C" fn helper_object_set_field(
    object_raw: u64,
    expected_slot: u32,
    value_raw: u64,
    func_id: u32,
    module_ptr: *const (),
    shared_state: *mut (),
) -> bool {
    if shared_state.is_null() || module_ptr.is_null() {
        return false;
    }
    let bridge = &*(shared_state.cast::<JitRuntimeBridgeContext>());
    let object_val = Value::from_raw(object_raw);
    let Some(object_ptr) = jit_object_ptr_checked(object_val) else {
        return false;
    };
    let object = &mut *object_ptr.as_ptr();
    let _ = bridge;
    let _ = module_ptr;
    let _ = func_id;
    // A raw field write, matching the interpreter's `StoreFieldExact`, which also
    // constructs `StructuralSlotBinding::Field(field_offset)` directly. There is
    // deliberately no adapter resolution here: the opcode is
    // `InterpreterBoundary` because its handler can invoke a descriptor setter as
    // a callable frame, which a leaf helper cannot express, so a faithful
    // implementation is not available to write.
    //
    // This previously carried `Dynamic`, `Method` and `Missing` arms behind a
    // `match` on a value constructed as `Field` one line above. All three were
    // unreachable, and the `Dynamic` arm carried an allocation the helper had no
    // business performing. Removing them makes the helper honest about what it is.
    object
        .set_field(expected_slot as usize, Value::from_raw(value_raw))
        .is_ok()
}

unsafe extern "C" fn helper_object_implements_shape(
    object_raw: u64,
    required_shape: u64,
    shared_state: *mut (),
) -> bool {
    if shared_state.is_null() {
        return false;
    }
    let bridge = &*(shared_state.cast::<JitRuntimeBridgeContext>());
    let object_val = Value::from_raw(object_raw);
    let Some(object_ptr) = jit_object_ptr_checked(object_val) else {
        return false;
    };
    let object = &*object_ptr.as_ptr();
    let Some(adapter) = jit_ensure_shape_adapter_for_object(bridge, object, required_shape) else {
        return false;
    };
    (0..adapter.len()).all(|slot| {
        !matches!(adapter.binding_for_slot(slot), StructuralSlotBinding::Missing)
    })
}

unsafe extern "C" fn helper_object_is_nominal(
    object_raw: u64,
    local_nominal_type_index: u32,
    module_ptr: *const (),
    shared_state: *mut (),
) -> bool {
    if shared_state.is_null() || module_ptr.is_null() {
        return false;
    }
    let bridge = &*(shared_state.cast::<JitRuntimeBridgeContext>());
    let module = &*(module_ptr.cast::<Module>());
    let Some(target_nominal_type_id) =
        jit_resolve_nominal_type_id(bridge, module, local_nominal_type_index)
    else {
        return false;
    };
    let object_val = Value::from_raw(object_raw);
    let Some(object_ptr) = jit_object_ptr_checked(object_val) else {
        return false;
    };
    let object = &*object_ptr.as_ptr();
    jit_object_matches_nominal_type(bridge, object, target_nominal_type_id)
}

unsafe extern "C" fn helper_object_get_shape_field(
    object_raw: u64,
    required_shape: u64,
    expected_slot: u32,
    optional: u8,
    _func_id: u32,
    _module_ptr: *const (),
    shared_state: *mut (),
) -> u64 {
    if shared_state.is_null() {
        return JIT_SHAPE_FIELD_FALLBACK_SENTINEL;
    }
    let bridge = &*(shared_state.cast::<JitRuntimeBridgeContext>());
    let object_val = Value::from_raw(object_raw);
    if optional != 0 && object_val.is_null() {
        return Value::null().raw();
    }
    let Some(object_ptr) = jit_object_ptr_checked(object_val) else {
        if std::env::var("RAYA_JIT_DEBUG_SHAPES").is_ok() {
            eprintln!(
                "jit shape field: non-object receiver raw=0x{object_raw:016x} shape={required_shape:#x} slot={expected_slot}"
            );
        }
        return JIT_SHAPE_FIELD_FALLBACK_SENTINEL;
    };
    let object = &*object_ptr.as_ptr();
    let Some(adapter) = jit_ensure_shape_adapter_for_object(bridge, object, required_shape) else {
        if std::env::var("RAYA_JIT_DEBUG_SHAPES").is_ok() {
            eprintln!(
                "jit shape field: missing adapter nominal={:?} layout={} shape={required_shape:#x} slot={expected_slot}",
                object.nominal_type_id_usize(),
                object.layout_id(),
            );
        }
        return JIT_SHAPE_FIELD_FALLBACK_SENTINEL;
    };
    let binding = adapter.binding_for_slot(expected_slot as usize);
    if std::env::var("RAYA_JIT_DEBUG_SHAPES").is_ok() {
        eprintln!(
            "jit shape field: nominal={:?} layout={} shape={required_shape:#x} slot={} binding={:?}",
            object.nominal_type_id_usize(),
            object.layout_id(),
            expected_slot,
            binding,
        );
    }
    match binding {
        StructuralSlotBinding::Field(slot) => object.get_field(slot).unwrap_or(Value::null()).raw(),
        StructuralSlotBinding::Dynamic(key) => object
            .dyn_map()
            .and_then(|dyn_map| dyn_map.get(&key).copied())
            .unwrap_or(Value::null())
            .raw(),
        StructuralSlotBinding::Method(method_slot) => {
            let Some(nominal_type_id) = object.nominal_type_id_usize() else {
                return JIT_SHAPE_FIELD_FALLBACK_SENTINEL;
            };
            let (func_id, method_module) = {
                let classes = (&*bridge.classes).read();
                let Some(class) = classes.get_class(nominal_type_id) else {
                    return JIT_SHAPE_FIELD_FALLBACK_SENTINEL;
                };
                let Some(fid) = class.vtable.get_method(method_slot) else {
                    return JIT_SHAPE_FIELD_FALLBACK_SENTINEL;
                };
                (fid, class.module.clone())
            };
            let bound = BoundMethod {
                receiver: object_val,
                func_id,
                module: method_module,
            };
            let mut gc = (&*bridge.gc).lock();
            let bm_ptr = gc.allocate(bound);
            Value::from_ptr(NonNull::new(bm_ptr.as_ptr()).unwrap()).raw()
        }
        StructuralSlotBinding::Missing => Value::null().raw(),
    }
}

/// Structural shape field store.
///
/// Wired at `jit/backend/cranelift/lowering.rs:1325`, but the opcode is
/// `Rejected` in `jit/capability.rs`, so this arm is not reachable from normal
/// compilation. Kept wired so the helper stays covered by its lowering test.
///
/// Like `helper_object_get_field` it does not consult descriptor accessors or
/// unwrap a proxy, and unlike `helper_object_get_field` it takes **no layout
/// generation** — the store lowering bakes none, because the generation the load
/// path uses comes from `any_layout_generation` on the compiled function. Adding
/// one would change the helper ABI for a path that cannot currently be reached,
/// so it is recorded as a precondition for re-promotion rather than done blind.
/// See Gap 5 in
/// /workspace/specs/2026-10-03-raya-d4-fixed-layout-objects.md.
unsafe extern "C" fn helper_object_set_shape_field(
    object_raw: u64,
    required_shape: u64,
    expected_slot: u32,
    value_raw: u64,
    _func_id: u32,
    _module_ptr: *const (),
    shared_state: *mut (),
) -> i8 {
    if shared_state.is_null() {
        return JIT_STORE_FALLBACK;
    }
    let bridge = &*(shared_state.cast::<JitRuntimeBridgeContext>());
    let object_val = Value::from_raw(object_raw);
    let Some(object_ptr) = jit_object_ptr_checked(object_val) else {
        return JIT_STORE_FALLBACK;
    };
    let object = &mut *object_ptr.as_ptr();
    let Some(adapter) = jit_ensure_shape_adapter_for_object(bridge, object, required_shape) else {
        return JIT_STORE_FALLBACK;
    };
    match adapter.binding_for_slot(expected_slot as usize) {
        StructuralSlotBinding::Field(slot) => object
            .set_field(slot, Value::from_raw(value_raw))
            .map(|_| JIT_STORE_SUCCESS)
            .unwrap_or(JIT_STORE_FALLBACK),
        StructuralSlotBinding::Dynamic(key) => {
            // See `helper_object_set_field`: `ensure_dyn_map` allocates inside a
            // JIT helper with empty native stack maps. Root the operands across
            // the allocation window before touching the map. Opening the scope
            // first is what keeps the FALLBACK contract: if roots are
            // unavailable we return before any mutation, so a caller falling back
            // to the interpreter cannot double-apply the store.
            let value = Value::from_raw(value_raw);
            let Some(_scope) = EphemeralRootScope::open(bridge, &[object_val, value])
            else {
                return JIT_STORE_FALLBACK;
            };
            object.ensure_dyn_map().insert(key, value);
            JIT_STORE_SUCCESS
        }
        StructuralSlotBinding::Method(_) | StructuralSlotBinding::Missing => JIT_STORE_FALLBACK,
    }
}

/// Raya's exact i32 division semantics. The VM raises a catchable error for a
/// zero divisor and wraps the otherwise overflowing MIN/-1 case.
pub fn exact_i32_div(left: i32, right: i32) -> Result<i32, &'static str> {
    if right == 0 {
        Err("division by zero")
    } else {
        Ok(left.wrapping_div(right))
    }
}

/// Raya's exact i32 remainder semantics.
pub fn exact_i32_rem(left: i32, right: i32) -> Result<i32, &'static str> {
    if right == 0 {
        Err("division by zero")
    } else {
        Ok(left.wrapping_rem(right))
    }
}

#[cfg(test)]
mod tests {
    use super::{exact_i32_div, exact_i32_rem};

    #[test]
    fn exact_integer_division_matches_raya_edges() {
        let cases = [(7, 2, 3), (-7, 2, -3), (i32::MIN, -1, i32::MIN)];
        for (left, right, expected) in cases {
            assert_eq!(exact_i32_div(left, right), Ok(expected));
        }
        assert_eq!(exact_i32_div(1, 0), Err("division by zero"));
    }

    #[test]
    fn exact_integer_remainder_matches_raya_edges() {
        let cases = [(7, 2, 1), (-7, 2, -1), (i32::MIN, -1, 0)];
        for (left, right, expected) in cases {
            assert_eq!(exact_i32_rem(left, right), Ok(expected));
        }
        assert_eq!(exact_i32_rem(1, 0), Err("division by zero"));
    }

    use super::*;
    use crate::compiler::bytecode::ClassDef;
    use crossbeam::channel::unbounded;
    use crossbeam_deque::Injector;
    use parking_lot::RwLock;
    use rustc_hash::FxHashMap;
    use std::sync::Arc;

    #[test]
    fn layout_guard_sentinel_is_not_a_valid_value_encoding() {
        // 0xFFFE is the null tag; a non-zero payload is reserved and cannot be
        // emitted by any safe Value constructor.
        assert_eq!(JIT_LAYOUT_GUARD_FALLBACK_SENTINEL >> 48, 0xFFFE);
        assert_ne!(
            JIT_LAYOUT_GUARD_FALLBACK_SENTINEL & 0x0000_FFFF_FFFF_FFFF,
            0
        );
        assert_ne!(JIT_LAYOUT_GUARD_FALLBACK_SENTINEL, Value::null().raw());
        assert_ne!(
            JIT_LAYOUT_GUARD_FALLBACK_SENTINEL,
            Value::u64(0xDEAD_0000_0005).raw()
        );
    }

    #[test]
    fn exact_field_helper_rejects_stale_layout_generation() {
        let safepoint = Arc::new(SafepointCoordinator::new(1));
        let tasks = Arc::new(RwLock::new(FxHashMap::default()));
        let injector = Arc::new(Injector::new());
        let shared = Arc::new(crate::vm::interpreter::SharedVmState::new(
            safepoint.clone(),
            tasks,
            injector,
        ));
        let module = Arc::new(Module::new("jit-layout-guard-test".to_string()));
        let task = Arc::new(Task::new(0, module.clone(), None));
        let code_cache = crate::jit::runtime::code_cache::CodeCache::new(1024);
        let expected_generation = code_cache.layout_generation(
            crate::jit::runtime::code_cache::LayoutDependency::AnyLayout,
        );
        let object_value = {
            let mut object = Object::new_structural(42, 1);
            object.fields[0] = Value::i32(73);
            let mut gc = shared.gc.lock();
            let object_ptr = gc.allocate(object);
            unsafe {
                Value::from_ptr(
                    NonNull::new(object_ptr.as_ptr() as *mut Object).expect("allocated object"),
                )
            }
        };
        let bridge = build_runtime_bridge_context(
            safepoint.as_ref(),
            &task,
            &shared.gc,
            &shared.classes,
            &shared.layouts,
            &code_cache,
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
            &shared.resolved_natives,
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
        let bridge_ptr = (&bridge as *const JitRuntimeBridgeContext) as *mut ();
        let module_ptr = Arc::as_ptr(&module).cast::<()>();

        let current = unsafe {
            helper_object_get_field(
                object_value.raw(),
                0,
                expected_generation,
                0,
                module_ptr,
                bridge_ptr,
            )
        };
        assert_eq!(current, Value::i32(73).raw());

        code_cache.invalidate_layout(42);
        let stale = unsafe {
            helper_object_get_field(
                object_value.raw(),
                0,
                expected_generation,
                0,
                module_ptr,
                bridge_ptr,
            )
        };
        assert_eq!(stale, JIT_LAYOUT_GUARD_FALLBACK_SENTINEL);
    }

    #[test]
    fn jit_helper_native_dispatch_returns_resolved_value() {
        let safepoint = Arc::new(SafepointCoordinator::new(1));
        let tasks = Arc::new(RwLock::new(FxHashMap::default()));
        let injector = Arc::new(Injector::new());
        let shared = Arc::new(crate::vm::interpreter::SharedVmState::new(
            safepoint.clone(),
            tasks,
            injector,
        ));
        {
            let mut reg = shared.native_registry.write();
            reg.register("jit.native.value", |_ctx, _args| NativeCallResult::i32(88));
            let resolved =
                ResolvedNatives::link(&["jit.native.value".to_string()], &reg).expect("link");
            *shared.resolved_natives.write() = resolved;
        }

        let module = Arc::new(Module::new("jit-test".to_string()));
        let task = Arc::new(Task::new(0, module, None));
        let code_cache = crate::jit::runtime::code_cache::CodeCache::new(1024);
        let bridge = build_runtime_bridge_context(
            safepoint.as_ref(),
            &task,
            &shared.gc,
            &shared.classes,
            &shared.layouts,
            &code_cache,
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
            &shared.resolved_natives,
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

        let raw = unsafe {
            helper_native_call_dispatch(
                0,
                std::ptr::null(),
                0,
                (&bridge as *const JitRuntimeBridgeContext) as *mut (),
            )
        };
        assert_eq!(raw, Value::i32(88).raw());
    }

    #[test]
    fn jit_helper_native_dispatch_submits_io_on_suspend() {
        let safepoint = Arc::new(SafepointCoordinator::new(1));
        let tasks = Arc::new(RwLock::new(FxHashMap::default()));
        let injector = Arc::new(Injector::new());
        let shared = Arc::new(crate::vm::interpreter::SharedVmState::new(
            safepoint.clone(),
            tasks,
            injector,
        ));
        let (tx, rx) = unbounded();
        *shared.io_submit_tx.lock() = Some(tx.clone());
        {
            let mut reg = shared.native_registry.write();
            reg.register("jit.native.suspend", |_ctx, _args| {
                NativeCallResult::Suspend(raya_sdk::IoRequest::Sleep { duration_nanos: 1 })
            });
            let resolved =
                ResolvedNatives::link(&["jit.native.suspend".to_string()], &reg).expect("link");
            *shared.resolved_natives.write() = resolved;
        }

        let module = Arc::new(Module::new("jit-test".to_string()));
        let task = Arc::new(Task::new(0, module, None));
        let code_cache = crate::jit::runtime::code_cache::CodeCache::new(1024);
        let bridge = build_runtime_bridge_context(
            safepoint.as_ref(),
            &task,
            &shared.gc,
            &shared.classes,
            &shared.layouts,
            &code_cache,
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
            &shared.resolved_natives,
            &shared.structural_shape_names,
            &shared.structural_layout_shapes,
            &shared.structural_shape_adapters,
            &shared.aot_profile,
            &shared.type_handles,
            &shared.prop_keys,
            &shared.stack_pool,
            shared.max_preemptions,
            0,
            Some(&tx),
        );

        let raw = unsafe {
            helper_native_call_dispatch(
                0,
                std::ptr::null(),
                0,
                (&bridge as *const JitRuntimeBridgeContext) as *mut (),
            )
        };
        assert_eq!(raw, JIT_NATIVE_SUSPEND_SENTINEL);
        let submission = rx.try_recv().expect("expected io submission");
        assert_eq!(submission.task_id.as_u64(), task.id().as_u64());
        assert!(matches!(
            submission.request,
            raya_sdk::IoRequest::Sleep { duration_nanos: 1 }
        ));
    }

    #[test]
    fn jit_helper_alloc_object_resolves_module_local_nominal_type_index() {
        let safepoint = Arc::new(SafepointCoordinator::new(1));
        let tasks = Arc::new(RwLock::new(FxHashMap::default()));
        let injector = Arc::new(Injector::new());
        let shared = Arc::new(crate::vm::interpreter::SharedVmState::new(
            safepoint.clone(),
            tasks,
            injector,
        ));

        let mut seed_module = Module::new("jit-seed".to_string());
        seed_module.classes.push(ClassDef {
            name: "Seed".to_string(),
            field_count: 1,
            parent_id: None,
            methods: Vec::new(),
        });
        let seed_module = Arc::new(
            Module::decode(&seed_module.encode()).expect("finalize seed module checksum"),
        );
        shared
            .register_module(seed_module)
            .expect("register seed module");

        let mut target_module = Module::new("jit-target".to_string());
        target_module.classes.push(ClassDef {
            name: "Target".to_string(),
            field_count: 2,
            parent_id: None,
            methods: Vec::new(),
        });
        let target_module = Arc::new(
            Module::decode(&target_module.encode()).expect("finalize target module checksum"),
        );
        shared
            .register_module(target_module.clone())
            .expect("register target module");

        let expected_nominal_type_id = shared
            .resolve_nominal_type_id(&target_module, 0)
            .expect("module-local nominal type id");

        let task = Arc::new(Task::new(0, target_module.clone(), None));
        let code_cache = crate::jit::runtime::code_cache::CodeCache::new(1024);
        let bridge = build_runtime_bridge_context(
            safepoint.as_ref(),
            &task,
            &shared.gc,
            &shared.classes,
            &shared.layouts,
            &code_cache,
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
            &shared.resolved_natives,
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
            helper_alloc_object(
                0,
                Arc::as_ptr(&target_module) as *const (),
                (&bridge as *const JitRuntimeBridgeContext) as *mut (),
            )
        };
        assert!(!object_ptr.is_null());

        let obj = unsafe { &*(object_ptr.cast::<Object>()) };
        assert_eq!(obj.nominal_type_id_usize(), Some(expected_nominal_type_id));
        assert_eq!(obj.field_count(), 2);
    }

    /// Build a bridge context over a fresh shared VM state for array-helper
    /// tests. Returns the shared state, module, code cache, and task so their
    /// storage outlives the returned bridge.
    fn array_helper_fixture() -> (
        Arc<crate::vm::interpreter::SharedVmState>,
        Arc<Module>,
        Box<crate::jit::runtime::code_cache::CodeCache>,
        Arc<Task>,
    ) {
        let safepoint = Arc::new(SafepointCoordinator::new(1));
        let tasks = Arc::new(RwLock::new(FxHashMap::default()));
        let injector = Arc::new(Injector::new());
        let shared = Arc::new(crate::vm::interpreter::SharedVmState::new(
            safepoint, tasks, injector,
        ));
        let module = Arc::new(Module::new("jit-array-test".to_string()));
        let task = Arc::new(Task::new(0, module.clone(), None));
        let code_cache = Box::new(crate::jit::runtime::code_cache::CodeCache::new(1024));
        (shared, module, code_cache, task)
    }

    macro_rules! with_array_bridge {
        ($shared:expr, $module:expr, $code_cache:expr, $task:expr, $bridge:ident, $body:block) => {{
            let $bridge = build_runtime_bridge_context(
                $shared.safepoint.as_ref(),
                &$task,
                &$shared.gc,
                &$shared.classes,
                &$shared.layouts,
                &$code_cache,
                &$shared.mutex_registry,
                &$shared.semaphore_registry,
                &$shared.globals_by_index,
                &$shared.builtin_global_slots,
                &$shared.constant_string_cache,
                &$shared.ephemeral_gc_roots,
                &$shared.pinned_handles,
                &$shared.tasks,
                &$shared.injector,
                &$shared.module_layouts,
                &$shared.metadata,
                &$shared.class_metadata,
                &$shared.native_handler,
                &$shared.resolved_natives,
                &$shared.structural_shape_names,
                &$shared.structural_layout_shapes,
                &$shared.structural_shape_adapters,
                &$shared.aot_profile,
                &$shared.type_handles,
                &$shared.prop_keys,
                &$shared.stack_pool,
                $shared.max_preemptions,
                0,
                None,
            );
            $body
        }};
    }

    /// D4.2 recorded, and D4.3 inherited, that `EphemeralRootScope` was never
    /// asserted: helpers opened a scope, but nothing checked that the roots were
    /// actually installed during the window or released afterwards. These two
    /// tests close that, and both properties are safety-critical rather than
    /// cosmetic.
    ///
    /// The window matters because JIT-compiled native frames publish empty stack
    /// maps. A collection triggered by an allocation inside a helper cannot see
    /// operands living only in machine registers, so they must be published as
    /// ephemeral roots for the duration. A scope that failed to release would
    /// silently grow the root set across every call.

    #[test]
    fn ephemeral_root_scope_releases_roots_after_a_scoped_push() {
        let (shared, module, code_cache, task) = array_helper_fixture();
        with_array_bridge!(shared, module, code_cache, task, bridge, {
            let ss = (&bridge as *const JitRuntimeBridgeContext) as *mut ();
            let module_ptr = Arc::as_ptr(&module) as *const ();

            // Baseline: nothing rooted.
            assert!(shared.ephemeral_gc_roots.read().is_empty());

            let arr_ptr = unsafe { helper_alloc_array(6, 2, module_ptr, ss) };
            let arr_val = unsafe { Value::from_ptr(NonNull::new(arr_ptr.cast::<u8>()).unwrap()) };

            assert_eq!(
                unsafe { helper_array_push(arr_val.raw(), Value::i32(11).raw(), ss) },
                JIT_STORE_SUCCESS
            );

            // The scope must have released exactly what it added, leaving the
            // shared root list as it found it.
            assert!(
                shared.ephemeral_gc_roots.read().is_empty(),
                "EphemeralRootScope leaked roots: {:?}",
                shared.ephemeral_gc_roots.read()
            );
            // And the push really happened, so this is not passing vacuously.
            assert_eq!(unsafe { helper_array_len(arr_val.raw(), ss) }, 3);
        });
    }

    #[test]
    fn scoped_helper_fails_closed_and_mutates_nothing_without_a_root_set() {
        let (shared, module, code_cache, task) = array_helper_fixture();
        with_array_bridge!(shared, module, code_cache, task, bridge, {
            let mut bridge = bridge;
            let ss = (&bridge as *const JitRuntimeBridgeContext) as *mut ();
            let module_ptr = Arc::as_ptr(&module) as *const ();

            let arr_ptr = unsafe { helper_alloc_array(6, 2, module_ptr, ss) };
            let arr_val = unsafe { Value::from_ptr(NonNull::new(arr_ptr.cast::<u8>()).unwrap()) };
            let len_before = unsafe { helper_array_len(arr_val.raw(), ss) };

            // Deny the root set. `EphemeralRootScope::open` returns None here, and
            // the helper must bail out rather than allocate with unprotected
            // operands.
            bridge.ephemeral_gc_roots = std::ptr::null();

            assert_eq!(
                unsafe { helper_array_push(arr_val.raw(), Value::i32(13).raw(), ss) },
                JIT_STORE_FALLBACK,
                "push must fail closed when the root set is unavailable"
            );
            // A fallback return has to mean "nothing was mutated", or a caller
            // falling back to the interpreter could double-apply the operation.
            assert_eq!(
                unsafe { helper_array_len(arr_val.raw(), ss) },
                len_before,
                "failed-closed push must not mutate the array"
            );
        });
    }

    /// D4.4: the RefCell helpers exist and are tested before anything can call
    /// them. They are NOT wired into the lowering, and `NewRefCell`,
    /// `LoadRefCell` and `StoreRefCell` stay `Rejected` until it is.
    /// D4.4: `helper_make_closure`, tested before anything can call it. NOT lowered
    /// yet, and `MakeClosure` stays `Rejected`.
    #[test]
    fn make_closure_helper_builds_a_closure_over_the_given_captures() {
        let (shared, module, code_cache, task) = array_helper_fixture();
        with_array_bridge!(shared, module, code_cache, task, bridge, {
            let ss = (&bridge as *const JitRuntimeBridgeContext) as *mut ();

            // Captures must arrive in capture order: the interpreter pops them off
            // the stack and reverses, so the JIT passes them already ordered.
            let captures: Vec<u64> = vec![
                Value::i32(7).raw(),
                Value::i32(8).raw(),
                Value::i32(9).raw(),
            ];

            let raw = unsafe {
                helper_make_closure(42, captures.as_ptr(), captures.len() as u32, ss)
            };
            assert_ne!(raw, 0, "closure allocation must not fail closed here");

            // Read the closure back out of the GC and check what was captured.
            let closure = unsafe {
                let value = Value::from_raw(raw);
                let ptr = value.as_ptr::<crate::vm::object::Closure>().unwrap();
                &*ptr.as_ptr()
            };
            assert_eq!(closure.func_id, 42);
            let got: Vec<i32> = closure
                .captures
                .iter()
                .map(|value| value.as_i32().unwrap_or(i32::MIN))
                .collect();
            assert_eq!(
                got,
                vec![7, 8, 9],
                "captures must survive in the order supplied"
            );
        });
    }

    /// `helper_set_closure_capture`, tested before anything can call it.
    /// `helper_load_captured`, tested before anything can call it. NOT lowered yet.
    /// `helper_bind_method`, tested before anything can call it. NOT lowered yet.
    ///
    /// The failure cases matter more than the success case here: the interpreter
    /// has four distinct diagnostics for this opcode and every one of them has to
    /// collapse to a zero return so the boundary exit can raise the right thing.
    /// D4.6 `Await`, paths 1 and 2. NOT lowered yet.
    ///
    /// The first test is the one that matters: `Value::as_u64` is **tag-gated**, so
    /// `await` on a boxed `i32 42` must push that exact value back. A payload-based
    /// "is this a task id" test would read 42 as a task id, and a guard that rejected
    /// integer-looking values would break ordinary `await 42`.
    #[test]
    fn await_helper_pushes_back_a_non_task_value() {
        let (shared, module, code_cache, task) = array_helper_fixture();
        with_array_bridge!(shared, module, code_cache, task, bridge, {
            let ss = (&bridge as *const JitRuntimeBridgeContext) as *mut ();

            // Path 1, the tagged-integer case: unchanged, not read as task id 42.
            for value in [
                Value::i32(42),
                Value::i64(42),
                Value::bool(true),
                Value::null(),
            ] {
                assert_eq!(
                    unsafe { helper_await_task(value.raw(), ss) },
                    value.raw(),
                    "a non-task value must be pushed back unchanged: {value:?}"
                );
            }
        });
    }

    #[test]
    fn await_helper_returns_a_completed_task_result_and_refuses_otherwise() {
        let (shared, module, code_cache, task) = array_helper_fixture();
        with_array_bridge!(shared, module, code_cache, task, bridge, {
            let ss = (&bridge as *const JitRuntimeBridgeContext) as *mut ();

            // Path 2: a completed task yields its result.
            let completed = std::sync::Arc::new(
                crate::vm::scheduler::Task::new(1, module.clone(), None),
            );
            completed.complete(Value::i32(99));
            let id = completed.id();
            let as_u64 = {
                // Task ids are carried as tagged u64 Values, which is exactly what `Await`
                // reads: `TaskId::as_u64` -> `Value::u64` -> tag-gated `as_u64()` back.
                crate::vm::value::Value::u64(id.as_u64()).raw()
            };
            {
                let mut tasks = unsafe { (&*bridge.tasks).write() };
                tasks.insert(id, completed.clone());
            }
            assert_eq!(
                unsafe { helper_await_task(as_u64, ss) },
                Value::i32(99).raw(),
                "a completed task must yield its result"
            );

            // A pending task must fall back rather than inventing a value: the
            // interpreter suspends, and the JIT has no AwaitTask suspend reason.
            let pending = std::sync::Arc::new(
                crate::vm::scheduler::Task::new(2, module.clone(), None),
            );
            let pending_id = pending.id();
            let pending_as_u64 = {
                crate::vm::value::Value::u64(pending_id.as_u64()).raw()
            };
            {
                let mut tasks = unsafe { (&*bridge.tasks).write() };
                tasks.insert(pending_id, pending.clone());
            }
            assert_eq!(
                unsafe { helper_await_task(pending_as_u64, ss) },
                JIT_INTERPRETER_FALLBACK_SENTINEL,
                "a pending task must fall back so the interpreter suspends"
            );

            // An unknown id falls back too, so the interpreter raises its own error.
            let unknown = {
                crate::vm::value::Value::u64(u64::MAX >> 4).raw()
            };
            assert_eq!(
                unsafe { helper_await_task(unknown, ss) },
                JIT_INTERPRETER_FALLBACK_SENTINEL
            );
        });
    }

    /// D4.7 `DynGetKeyed`: the two views a helper can honestly reproduce, and the
    /// one it must decline.
    ///
    /// The `Struct` case is the important half. Its field lookup falls back to
    /// `structural_object_shapes`, a registry the bridge does not carry, so the
    /// helper must return the fallback sentinel rather than approximate the
    /// resolution. A test that only exercised `Arr` would pass while that
    /// fallback silently regressed into a wrong answer.
    #[test]
    fn dyn_get_keyed_helper_handles_arr_and_str_and_defers_the_rest() {
        use crate::vm::object::{Array, Object, RayaString};
        use std::sync::Arc;

        let (shared, module, code_cache, task) = array_helper_fixture();
        with_array_bridge!(shared, module, code_cache, task, bridge, {
            let ss = (&bridge as *const JitRuntimeBridgeContext) as *mut ();

            // Arrange an array and a string in the shared GC.
            let (arr_raw, str_raw, obj_raw) = {
                let mut gc = shared.gc.lock();

                let mut array = Array::new(0, 2);
                array.set(0, Value::i32(7)).unwrap();
                array.set(1, Value::i32(8)).unwrap();
                let array_ptr = gc.allocate(array);
                let arr_raw = unsafe {
                    Value::from_ptr(NonNull::new(array_ptr.as_ptr()).unwrap()).raw()
                };

                let string_ptr = gc.allocate(RayaString::new("hello".to_string()));
                let str_raw =
                    unsafe { Value::from_ptr(NonNull::new(string_ptr.as_ptr()).unwrap()).raw() };

                // A nominal object: a `Struct` view for the keyed path.
                let obj_ptr = gc.allocate(Object::new_nominal(1, 5, 1));
                let obj_raw =
                    unsafe { Value::from_ptr(NonNull::new(obj_ptr.as_ptr()).unwrap()).raw() };

                (arr_raw, str_raw, obj_raw)
            };
            let _ = Arc::strong_count(&shared);

            // Arr, in range.
            let arr_key = Value::i32(1);
            assert_eq!(
                unsafe { helper_dyn_get_keyed(arr_raw, arr_key.raw(), ss) },
                Value::i32(8).raw(),
                "an in-range array index must read the element"
            );
            // Arr, out of range: a legitimate null, not a fallback.
            let oob = Value::i32(99);
            assert_eq!(
                unsafe { helper_dyn_get_keyed(arr_raw, oob.raw(), ss) },
                Value::null().raw(),
                "an out-of-range array index is null, not a fallback"
            );

            // Str, char index — each char is a freshly allocated RayaString.
            let idx = Value::i32(1);
            let char_value =
                unsafe { Value::from_raw(helper_dyn_get_keyed(str_raw, idx.raw(), ss)) };
            let Some(char_ptr) = (unsafe { char_value.as_ptr::<RayaString>() }) else {
                panic!("expected an allocated RayaString for the character");
            };
            assert_eq!(unsafe { char_ptr.as_ref().data.as_str() }, "e");

            // Str, "length".
            let length_key = unsafe {
                let mut gc = shared.gc.lock();
                let k = gc.allocate(RayaString::new("length".to_string()));
                Value::from_raw(Value::from_ptr(NonNull::new(k.as_ptr()).unwrap()).raw())
            };
            assert_eq!(
                unsafe { helper_dyn_get_keyed(str_raw, length_key.raw(), ss) },
                Value::i32(5).raw()
            );

            // Struct: MUST decline. This is the whole reason the helper is shaped
            // this way.
            let zero = Value::i32(0);
            assert_eq!(
                unsafe { helper_dyn_get_keyed(obj_raw, zero.raw(), ss) },
                JIT_INTERPRETER_FALLBACK_SENTINEL,
                "a Struct lookup needs structural_object_shapes, which the bridge \
                 lacks, so the helper must defer to the interpreter"
            );

            // Non-node target also defers.
            assert_eq!(
                unsafe { helper_dyn_get_keyed(Value::i32(5).raw(), Value::i32(0).raw(), ss) },
                JIT_INTERPRETER_FALLBACK_SENTINEL
            );
        });
    }

    #[test]
    fn bind_method_helper_binds_and_rejects() {
        let (shared, module, code_cache, task) = array_helper_fixture();
        with_array_bridge!(shared, module, code_cache, task, bridge, {
            let ss = (&bridge as *const JitRuntimeBridgeContext) as *mut ();

            // Register the class first: `register_class` assigns the nominal type
            // id, and the object must carry that same id.
            let nominal_type_id = {
                let mut classes = unsafe { (&*bridge.classes).write() };
                let mut class = crate::vm::object::Class::new(0, "Point".to_string(), 2);
                class.module = Some(module.clone());
                class.vtable.add_method(42);
                classes.register_class(class)
            };

            let object_raw = {
                let mut gc = shared.gc.lock();
                let mut object =
                    crate::vm::object::Object::new_nominal(1, nominal_type_id as u32, 2);
                object.set_field(0, Value::i32(99)).unwrap();
                let ptr = gc.allocate(object);
                unsafe { Value::from_ptr(NonNull::new(ptr.as_ptr()).unwrap()).raw() }
            };
            {
                let mut classes = unsafe { (&*bridge.classes).write() };
                // `register_class` assigns the type id, so build a class carrying the
                // one we need and register it; the object below uses that id.
                let mut class = crate::vm::object::Class::new(0, "Point".to_string(), 2);
                class.module = Some(module.clone());
                class.vtable.add_method(42);
                let id = classes.register_class(class);
            }

            // Success: a BoundMethod carrying the receiver and the resolved func_id.
            let bound = unsafe { helper_bind_method(object_raw, 0, ss) };
            assert_ne!(bound, 0, "binding method slot 0 must succeed");
            let bm = unsafe {
                let value = Value::from_raw(bound);
                let ptr = value
                    .as_ptr::<crate::vm::object::BoundMethod>()
                    .expect("result must be a BoundMethod");
                &*ptr.as_ptr()
            };
            assert_eq!(bm.func_id, 42, "vtable slot must resolve to the func id");
            assert_eq!(bm.receiver.raw(), object_raw, "receiver must be carried");

            // Failure cases, each of which is a distinct interpreter diagnostic.
            assert_eq!(
                unsafe { helper_bind_method(Value::i32(1).raw(), 0, ss) },
                0,
                "a non-pointer receiver must be refused"
            );
            assert_eq!(
                unsafe { helper_bind_method(object_raw, 99, ss) },
                0,
                "an unknown method slot must be refused"
            );
        });
    }

    #[test]
    fn load_captured_helper_reads_the_active_closure() {
        let (shared, module, code_cache, task) = array_helper_fixture();
        with_array_bridge!(shared, module, code_cache, task, bridge, {
            let ss = (&bridge as *const JitRuntimeBridgeContext) as *mut ();

            // A closure capturing [7, 8], installed as the task's active closure.
            let closure_raw = {
                let mut gc = shared.gc.lock();
                let closure =
                    crate::vm::object::Closure::new(0, vec![Value::i32(7), Value::i32(8)]);
                let ptr = gc.allocate(closure);
                unsafe { Value::from_ptr(NonNull::new(ptr.as_ptr()).unwrap()).raw() }
            };
            task.push_closure(unsafe { Value::from_raw(closure_raw) });

            // The helper reads the active closure from the task, not from a frame.
            assert_eq!(
                unsafe { helper_load_captured(0, ss) },
                Value::i32(7).raw()
            );
            assert_eq!(
                unsafe { helper_load_captured(1, ss) },
                Value::i32(8).raw()
            );

            // Out of range: bounds-checked, so a fallback rather than a read past
            // the capture vector.
            assert_eq!(
                unsafe { helper_load_captured(9, ss) },
                JIT_INTERPRETER_FALLBACK_SENTINEL
            );

            // With no active closure the interpreter raises "LoadCaptured without
            // active closure", so the helper must refuse rather than invent a value.
            task.pop_closure();
            assert_eq!(
                unsafe { helper_load_captured(0, ss) },
                JIT_INTERPRETER_FALLBACK_SENTINEL
            );
        });
    }

    /// `helper_store_captured`, tested before anything can call it. NOT lowered yet.
    #[test]
    fn store_captured_helper_writes_through_the_active_closure() {
        let (shared, module, code_cache, task) = array_helper_fixture();
        with_array_bridge!(shared, module, code_cache, task, bridge, {
            let ss = (&bridge as *const JitRuntimeBridgeContext) as *mut ();

            let closure_raw = {
                let mut gc = shared.gc.lock();
                let closure =
                    crate::vm::object::Closure::new(0, vec![Value::i32(7), Value::i32(8)]);
                let ptr = gc.allocate(closure);
                unsafe { Value::from_ptr(NonNull::new(ptr.as_ptr()).unwrap()).raw() }
            };
            task.push_closure(unsafe { Value::from_raw(closure_raw) });

            assert_eq!(
                unsafe { helper_store_captured(0, Value::i32(42).raw(), ss) },
                JIT_STORE_SUCCESS
            );
            // The write must be visible through the load helper, which reads the
            // same active closure.
            assert_eq!(
                unsafe { helper_load_captured(0, ss) },
                Value::i32(42).raw()
            );

            // Out of range falls back without mutating.
            assert_eq!(
                unsafe { helper_store_captured(9, Value::i32(1).raw(), ss) },
                JIT_STORE_FALLBACK
            );
            assert_eq!(
                unsafe { helper_load_captured(1, ss) },
                Value::i32(8).raw(),
                "a rejected store must leave the capture untouched"
            );

            // With no active closure the interpreter raises, so the helper refuses
            // rather than writing to whatever happens to be on the stack.
            task.pop_closure();
            assert_eq!(
                unsafe { helper_store_captured(0, Value::i32(1).raw(), ss) },
                JIT_STORE_FALLBACK
            );
        });
    }

    #[test]
    fn set_closure_capture_helper_patches_a_slot() {
        let (shared, module, code_cache, task) = array_helper_fixture();
        with_array_bridge!(shared, module, code_cache, task, bridge, {
            let ss = (&bridge as *const JitRuntimeBridgeContext) as *mut ();

            // A closure with one capture, holding 7.
            let mut gc = shared.gc.lock();
            let closure = crate::vm::object::Closure::new(0, vec![Value::i32(7)]);
            let ptr = gc.allocate(closure);
            let closure_raw =
                unsafe { Value::from_ptr(NonNull::new(ptr.as_ptr()).unwrap()).raw() };
            drop(gc);

            // Patch the slot, then read it back.
            assert_eq!(
                unsafe { helper_set_closure_capture(closure_raw, 0, Value::i32(11).raw(), ss) },
                JIT_STORE_SUCCESS
            );
            assert_eq!(
                unsafe {
                    let value = Value::from_raw(closure_raw);
                    let c = &*value.as_ptr::<crate::vm::object::Closure>().unwrap().as_ptr();
                    c.get_captured(0).unwrap().as_i32()
                },
                Some(11)
            );

            // Out of range: bounds-checked, so a fallback and no mutation rather
            // than a write past the capture vector.
            assert_eq!(
                unsafe { helper_set_closure_capture(closure_raw, 9, Value::i32(1).raw(), ss) },
                JIT_STORE_FALLBACK
            );
            // Non-pointer receiver: the interpreter's weak check, refused.
            assert_eq!(
                unsafe { helper_set_closure_capture(Value::i32(5).raw(), 0, Value::i32(1).raw(), ss) },
                JIT_STORE_FALLBACK
            );
        });
    }

    #[test]
    fn make_closure_helper_fails_closed_without_a_root_set() {
        let (shared, module, code_cache, task) = array_helper_fixture();
        with_array_bridge!(shared, module, code_cache, task, bridge, {
            let mut bridge = bridge;
            let ss = (&bridge as *const JitRuntimeBridgeContext) as *mut ();

            let captures: Vec<u64> = vec![Value::i32(7).raw()];

            // Native stack maps are empty, so allocating with unprotected captures
            // could collect them.
            bridge.ephemeral_gc_roots = std::ptr::null();

            assert_eq!(
                unsafe { helper_make_closure(42, captures.as_ptr(), 1, ss) },
                0,
                "closure allocation must fail closed without a root set"
            );
        });
    }

    #[test]
    fn refcell_helpers_roundtrip_and_reject_non_pointers() {
        let (shared, module, code_cache, task) = array_helper_fixture();
        with_array_bridge!(shared, module, code_cache, task, bridge, {
            let ss = (&bridge as *const JitRuntimeBridgeContext) as *mut ();

            // allocate a RefCell holding 7, read it back, overwrite, read again.
            let cell = unsafe { helper_new_refcell(Value::i32(7).raw(), ss) };
            assert_ne!(cell, 0, "allocation must not fail closed here");
            // `cell` is the RefCell's address, not its contents; the 7 lives
            // inside it and comes back through the load helper.
            assert_eq!(
                unsafe { helper_load_refcell(cell, ss) },
                Value::i32(7).raw()
            );
            assert_eq!(
                unsafe { helper_store_refcell(cell, Value::i32(9).raw(), ss) },
                JIT_STORE_SUCCESS
            );
            assert_eq!(
                unsafe { Value::from_raw(helper_load_refcell(cell, ss)).as_i32() },
                Some(9)
            );
        });
    }

    #[test]
    fn refcell_helpers_reject_non_pointers() {
        let (shared, module, code_cache, task) = array_helper_fixture();
        with_array_bridge!(shared, module, code_cache, task, bridge, {
            let ss = (&bridge as *const JitRuntimeBridgeContext) as *mut ();

            // An immediate is not a pointer, so both must refuse. This is the
            // interpreter's weak `is_ptr()` check reproduced exactly -- a heap
            // value of the wrong type is still accepted, as it is in the
            // interpreter today (ALY-54).
            let immediate = Value::i32(5).raw();
            assert_eq!(
                unsafe { helper_load_refcell(immediate, ss) },
                JIT_INTERPRETER_FALLBACK_SENTINEL
            );
            assert_eq!(
                unsafe { helper_store_refcell(immediate, Value::i32(1).raw(), ss) },
                JIT_STORE_FALLBACK
            );
        });
    }

    #[test]
    fn new_refcell_fails_closed_without_a_root_set() {
        let (shared, module, code_cache, task) = array_helper_fixture();
        with_array_bridge!(shared, module, code_cache, task, bridge, {
            let mut bridge = bridge;
            let ss = (&bridge as *const JitRuntimeBridgeContext) as *mut ();

            // Deny the root set. Native stack maps are empty, so allocating here
            // with an unprotected operand could collect the initial value.
            bridge.ephemeral_gc_roots = std::ptr::null();

            assert_eq!(
                unsafe { helper_new_refcell(Value::i32(7).raw(), ss) },
                0,
                "RefCell allocation must fail closed without a root set"
            );
        });
    }

    #[test]
    fn jit_array_helpers_roundtrip_and_reject_non_arrays() {
        let (shared, module, code_cache, task) = array_helper_fixture();
        with_array_bridge!(shared, module, code_cache, task, bridge, {
            let ss = (&bridge as *const JitRuntimeBridgeContext) as *mut ();
            let module_ptr = Arc::as_ptr(&module) as *const ();

            // Dynamic array (AnyValue element id 6), capacity 2.
            let arr_ptr = unsafe { helper_alloc_array(6, 2, module_ptr, ss) };
            assert!(!arr_ptr.is_null());
            let arr_val = unsafe { Value::from_ptr(NonNull::new(arr_ptr.cast::<u8>()).unwrap()) };

            // len == 2, slots start null.
            assert_eq!(unsafe { helper_array_len(arr_val.raw(), ss) }, 2);
            assert!(unsafe { Value::from_raw(helper_array_load(arr_val.raw(), 0, ss)) }.is_null());

            // store then load.
            assert_eq!(
                unsafe { helper_array_store(arr_val.raw(), 0, Value::i32(7).raw(), ss) },
                JIT_STORE_SUCCESS
            );
            assert_eq!(
                unsafe { Value::from_raw(helper_array_load(arr_val.raw(), 0, ss)).as_i32() },
                Some(7)
            );

            // out-of-bounds store falls back; load falls back with the sentinel.
            assert_eq!(
                unsafe { helper_array_store(arr_val.raw(), 5, Value::i32(1).raw(), ss) },
                JIT_STORE_FALLBACK
            );
            assert_eq!(
                unsafe { helper_array_load(arr_val.raw(), 5, ss) },
                JIT_INTERPRETER_FALLBACK_SENTINEL
            );

            // push grows the array; pop returns it.
            assert_eq!(
                unsafe { helper_array_push(arr_val.raw(), Value::i32(9).raw(), ss) },
                JIT_STORE_SUCCESS
            );
            assert_eq!(unsafe { helper_array_len(arr_val.raw(), ss) }, 3);
            assert_eq!(
                unsafe { Value::from_raw(helper_array_pop(arr_val.raw(), ss)).as_i32() },
                Some(9)
            );

            // A non-array receiver (a string) is rejected by every helper.
            let string_val = {
                let mut gc = shared.gc.lock();
                let s = gc.allocate(RayaString::new("nope".to_string()));
                unsafe { Value::from_ptr(NonNull::new(s.as_ptr()).unwrap()) }
            };
            assert_eq!(
                unsafe { helper_array_len(string_val.raw(), ss) },
                JIT_ARRAY_LEN_FALLBACK_SENTINEL
            );
            assert_eq!(
                unsafe { helper_array_load(string_val.raw(), 0, ss) },
                JIT_INTERPRETER_FALLBACK_SENTINEL
            );
            assert_eq!(
                unsafe { helper_array_store(string_val.raw(), 0, Value::i32(1).raw(), ss) },
                JIT_STORE_FALLBACK
            );
            assert_eq!(
                unsafe { helper_array_pop(string_val.raw(), ss) },
                JIT_INTERPRETER_FALLBACK_SENTINEL
            );
        });
    }

    #[test]
    fn jit_typed_array_store_enforces_element_constraint() {
        let (shared, module, code_cache, task) = array_helper_fixture();
        with_array_bridge!(shared, module, code_cache, task, bridge, {
            let ss = (&bridge as *const JitRuntimeBridgeContext) as *mut ();
            let module_ptr = Arc::as_ptr(&module) as *const ();

            // Bool-typed array (element id 2): storing a bool succeeds, an i32 falls back.
            let arr_ptr = unsafe { helper_alloc_array(2, 1, module_ptr, ss) };
            let arr_val = unsafe { Value::from_ptr(NonNull::new(arr_ptr.cast::<u8>()).unwrap()) };
            assert_eq!(
                unsafe { helper_array_store(arr_val.raw(), 0, Value::bool(true).raw(), ss) },
                JIT_STORE_SUCCESS
            );
            assert_eq!(
                unsafe { helper_array_store(arr_val.raw(), 0, Value::i32(1).raw(), ss) },
                JIT_STORE_FALLBACK
            );
            // null is always accepted, even into a typed slot.
            assert_eq!(
                unsafe { helper_array_store(arr_val.raw(), 0, Value::null().raw(), ss) },
                JIT_STORE_SUCCESS
            );
        });
    }
}
