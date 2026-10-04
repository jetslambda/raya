//! Object opcode handlers: nominal allocation, field access, structural field access,
//! object literals, and method binding

use crate::compiler::Module;
use crate::compiler::Opcode;
use crate::vm::gc::header_ptr_from_value_ptr;
use crate::vm::interpreter::execution::{OpcodeResult, ReturnAction};
use crate::vm::interpreter::shared_state::{
    ShapeAdapter, StructuralAdapterKey, StructuralSlotBinding,
};
use crate::vm::interpreter::Interpreter;
use crate::vm::object::{Array, BoundMethod, Closure, Object, RayaString};
use crate::vm::stack::Stack;
use crate::vm::value::Value;
use crate::vm::VmError;
use std::sync::Arc;

/// The metadata key under which a Node-compat property descriptor is stored.
///
/// Previously duplicated verbatim in `native.rs`. One definition now: two private
/// copies of the same magic string is exactly the drift this milestone exists to
/// remove, and a rename that updated one and not the other would be silent.
pub(crate) const NODE_DESCRIPTOR_METADATA_KEY: &str = "__node_compat_descriptor";

impl<'a> Interpreter<'a> {
    fn load_shape_field_on_non_object(
        &mut self,
        receiver: Value,
        shape_id: u64,
        field_offset: usize,
    ) -> Option<Value> {
        use crate::vm::json::view::{js_classify, JSView};

        let member_name = {
            let names = self.structural_shape_names.read();
            names.get(&shape_id)?.get(field_offset)?.clone()
        };

        let bound_native = |this: &mut Self, native_id: u16| {
            let method = crate::vm::object::BoundNativeMethod {
                receiver,
                native_id,
            };
            let method_ptr = this.gc.lock().allocate(method);
            unsafe { Value::from_ptr(std::ptr::NonNull::new(method_ptr.as_ptr()).unwrap()) }
        };

        match js_classify(receiver) {
            JSView::Arr(ptr) => {
                let arr = unsafe { &*ptr };
                if member_name == "length" {
                    Some(Value::i32(arr.len() as i32))
                } else {
                    super::types::builtin_handle_native_method_id(receiver, &member_name)
                        .map(|native_id| bound_native(self, native_id))
                        .or(Some(Value::null()))
                }
            }
            JSView::Str(ptr) => {
                let s = unsafe { &*ptr };
                if member_name == "length" {
                    Some(Value::i32(s.len() as i32))
                } else {
                    super::types::builtin_handle_native_method_id(receiver, &member_name)
                        .map(|native_id| bound_native(self, native_id))
                        .or(Some(Value::null()))
                }
            }
            _ => None,
        }
    }

    fn nominal_method_slot_by_name(&self, nominal_type_id: usize, method_name: &str) -> Option<usize> {
        let classes = self.classes.read();
        let class = classes.get_class(nominal_type_id)?;
        let module = class.module.as_ref()?;
        for (slot, function_id) in class.vtable.methods.iter().copied().enumerate() {
            let function = module.functions.get(function_id)?;
            if function.name == method_name || function.name.ends_with(&format!("::{method_name}")) {
                return Some(slot);
            }
        }
        None
    }

    pub(in crate::vm::interpreter) fn bound_method_value_for_slot(
        &mut self,
        receiver: Value,
        method_slot: usize,
    ) -> Result<Value, VmError> {
        let receiver = Self::ensure_object_receiver(receiver, "method binding")?;
        let obj = unsafe { &*receiver.as_ptr::<Object>().unwrap().as_ptr() };
        let nominal_type_id = obj.nominal_type_id_usize().ok_or_else(|| {
            VmError::TypeError("Cannot bind method on structural object value".to_string())
        })?;
        let classes = self.classes.read();
        let class = classes
            .get_class(nominal_type_id)
            .ok_or_else(|| {
                VmError::RuntimeError(format!("Invalid nominal type id: {}", nominal_type_id))
            })?;
        let func_id = class.vtable.get_method(method_slot).ok_or_else(|| {
            VmError::RuntimeError(format!(
                "Invalid method slot: {} for class {}",
                method_slot, class.name
            ))
        })?;
        let method_module = class.module.clone();
        drop(classes);

        let bm = BoundMethod {
            receiver,
            func_id,
            module: method_module,
        };
        let gc_ptr = self.gc.lock().allocate(bm);
        Ok(unsafe { Value::from_ptr(std::ptr::NonNull::new(gc_ptr.as_ptr()).unwrap()) })
    }

    pub(in crate::vm::interpreter) fn callable_frame_for_value(
        &self,
        callable: Value,
        stack: &mut Stack,
        args: &[Value],
        return_action: ReturnAction,
    ) -> Result<Option<OpcodeResult>, VmError> {
        if !callable.is_ptr() {
            return Ok(None);
        }
        let header =
            unsafe { &*header_ptr_from_value_ptr(callable.as_ptr::<u8>().unwrap().as_ptr()) };
        if header.type_id() == std::any::TypeId::of::<BoundMethod>() {
            let bm = unsafe { &*callable.as_ptr::<BoundMethod>().unwrap().as_ptr() };
            stack.push(bm.receiver)?;
            for arg in args {
                stack.push(*arg)?;
            }
            return Ok(Some(OpcodeResult::PushFrame {
                func_id: bm.func_id,
                arg_count: args.len() + 1,
                is_closure: false,
                closure_val: None,
                module: bm.module.clone(),
                return_action,
            }));
        }
        if header.type_id() == std::any::TypeId::of::<Closure>() {
            let closure_module =
                unsafe { &*callable.as_ptr::<Closure>().unwrap().as_ptr() }.module();
            for arg in args {
                stack.push(*arg)?;
            }
            return Ok(Some(OpcodeResult::PushFrame {
                func_id: unsafe { &*callable.as_ptr::<Closure>().unwrap().as_ptr() }.func_id(),
                arg_count: args.len(),
                is_closure: true,
                closure_val: Some(callable),
                module: closure_module,
                return_action,
            }));
        }
        Ok(None)
    }

    fn legacy_field_name_for_layout(field_offset: usize, field_count: usize) -> Option<String> {
        let name = match field_offset {
            0 => "message",
            1 => "name",
            2 => "stack",
            3 => "cause",
            4 => "code",
            5 => "errno",
            6 => "syscall",
            7 => "path",
            8 => "errors",
            _ => return None,
        };
        (field_offset < field_count).then(|| name.to_string())
    }

    fn field_name_for_offset(&self, obj: &Object, field_offset: usize) -> Option<String> {
        let nominal_type_id = obj.nominal_type_id_usize();
        let class_metadata = self.class_metadata.read();
        let from_metadata = nominal_type_id.and_then(|nominal_type_id| {
            class_metadata
                .get(nominal_type_id)
                .and_then(|meta| meta.field_names.get(field_offset))
                .cloned()
                .filter(|name| !name.is_empty())
        });
        if from_metadata.is_some() {
            return from_metadata;
        }
        if let Some(name) = self
            .layout_field_names_for_object(obj)
            .and_then(|names| names.get(field_offset).cloned())
        {
            return Some(name);
        }
        Self::legacy_field_name_for_layout(field_offset, obj.field_count())
    }

    fn field_index_for_value(&self, obj_val: Value, field_name: &str) -> Option<usize> {
        // Delegates to the shared resolver. This body was a SECOND, inline copy of the
        // same three-step resolution that `get_field_index_for_value` had -- 24 callers
        // on this one, 20 on that -- so the interpreter had two code paths that could
        // drift. It is now one.
        //
        // It also used a differently-NAMED backstop, `legacy_field_index_for_layout`,
        // which read as though it must behave differently from
        // `legacy_object_literal_field_index`. Having compared both bodies they are
        // the same name->index table and the same `(idx < field_count)` bound, so
        // routing through the shared function is behaviour-preserving. **The name
        // difference was the only warning sign, and it pointed the wrong way** -- a
        // plausible-looking difference that would have made consolidating these two
        // look like a P0.
        let obj_ptr = unsafe { obj_val.as_ptr::<Object>() }?;
        let obj = unsafe { &*obj_ptr.as_ptr() };
        crate::vm::interpreter::opcodes::native::object_field_index(
            obj,
            field_name,
            &self.class_metadata,
            &self.layouts,
            self.structural_object_shapes,
        )
    }

    pub(in crate::vm::interpreter) fn build_shape_slot_map_for_object(
        &self,
        obj: &Object,
        required_names: &[String],
    ) -> Option<Vec<StructuralSlotBinding>> {
        let dynamic_binding_for = |name: &str| -> Option<StructuralSlotBinding> {
            let key = self.intern_prop_key(name);
            obj.dyn_map().and_then(|dyn_map| {
                dyn_map
                    .contains_key(&key)
                    .then_some(StructuralSlotBinding::Dynamic(key))
            })
        };
        let layout_names = self.layout_field_names_for_object(obj);

        if let Some(nominal_type_id) = obj.nominal_type_id_usize() {
            let class_metadata = self.class_metadata.read();
            let class_meta = class_metadata.get(nominal_type_id).cloned();
            drop(class_metadata);
            return Some(
                required_names
                    .iter()
                    .map(|name| {
                        class_meta
                            .as_ref()
                            .and_then(|meta| meta.get_field_index(name))
                            .and_then(|index| {
                                (index < obj.field_count())
                                    .then_some(StructuralSlotBinding::Field(index))
                            })
                            .or_else(|| {
                                layout_names
                                    .as_ref()
                                    .and_then(|names| {
                                        names.iter().position(|actual| actual == name)
                                    })
                                    .map(StructuralSlotBinding::Field)
                            })
                            .or_else(|| {
                                class_meta
                                    .as_ref()
                                    .and_then(|meta| meta.get_method_index(name))
                                    .map(StructuralSlotBinding::Method)
                            })
                            .or_else(|| {
                                self.nominal_method_slot_by_name(nominal_type_id, name)
                                    .map(StructuralSlotBinding::Method)
                            })
                            .or_else(|| dynamic_binding_for(name))
                            .unwrap_or(StructuralSlotBinding::Missing)
                    })
                    .collect(),
            );
        }

        let actual_names = layout_names;
        Some(
            required_names
                .iter()
                .map(|name| {
                    actual_names
                        .as_ref()
                        .and_then(|names| names.iter().position(|actual| actual == name))
                        .map(StructuralSlotBinding::Field)
                        .or_else(|| dynamic_binding_for(name))
                        .unwrap_or(StructuralSlotBinding::Missing)
                })
                .collect(),
        )
    }

    pub(in crate::vm::interpreter) fn ensure_shape_adapter_for_object(
        &self,
        obj: &Object,
        required_shape: crate::vm::object::ShapeId,
    ) -> Option<Arc<ShapeAdapter>> {
        let debug_structural = std::env::var("RAYA_DEBUG_STRUCTURAL_VIEW").is_ok();
        let adapter_key = StructuralAdapterKey {
            provider_layout: obj.layout_id(),
            required_shape,
        };
        let current_epoch = self
            .layouts
            .read()
            .layout_epoch(obj.layout_id())
            .unwrap_or(0);
        if let Some(adapter) = self
            .structural_shape_adapters
            .read()
            .get(&adapter_key)
            .cloned()
        {
            if adapter.epoch == current_epoch {
                return Some(adapter);
            }
        }

        let required_names = self
            .structural_shape_names
            .read()
            .get(&required_shape)
            .cloned();
        let Some(required_names) = required_names else {
            if debug_structural {
                eprintln!(
                    "[structural-shape] missing shape names layout={} shape={}",
                    obj.layout_id(),
                    required_shape
                );
            }
            return None;
        };
        let slot_map = self.build_shape_slot_map_for_object(obj, &required_names);
        let Some(slot_map) = slot_map else {
            if debug_structural {
                eprintln!(
                    "[structural-shape] cannot build slot map layout={} shape={} names=[{}]",
                    obj.layout_id(),
                    required_shape,
                    required_names.join(",")
                );
            }
            return None;
        };
        if debug_structural {
            let rendered = slot_map
                .iter()
                .enumerate()
                .map(|(idx, binding)| match binding {
                    StructuralSlotBinding::Field(slot) => format!("{idx}->f{slot}"),
                    StructuralSlotBinding::Method(slot) => format!("{idx}->m{slot}"),
                    StructuralSlotBinding::Dynamic(key) => format!("{idx}->d{key}"),
                    StructuralSlotBinding::Missing => format!("{idx}->missing"),
                })
                .collect::<Vec<_>>()
                .join(",");
            eprintln!(
                "[structural-shape] build layout={} shape={} names=[{}] map=[{}]",
                obj.layout_id(),
                required_shape,
                required_names.join(","),
                rendered
            );
        }
        let adapter = Arc::new(ShapeAdapter::from_slot_map(
            obj.layout_id(),
            required_shape,
            &slot_map,
            current_epoch,
        ));
        let mut adapters = self.structural_shape_adapters.write();
        Some(
            adapters
                .entry(adapter_key)
                .or_insert_with(|| adapter.clone())
                .clone(),
        )
    }

    fn get_value_field_by_name(&self, obj_val: Value, field_name: &str) -> Option<Value> {
        let index = self.field_index_for_value(obj_val, field_name)?;
        let obj_ptr = unsafe { obj_val.as_ptr::<Object>() }?;
        let obj = unsafe { &*obj_ptr.as_ptr() };
        obj.get_field(index)
    }

    pub(crate) fn is_field_writable(&self, obj_val: Value, field_name: &str) -> bool {
        // Delegates to the shared function so a JIT helper asks the identical
        // question. Its three permissive defaults are part of the contract -- see the
        // doc comment there.
        crate::vm::interpreter::opcodes::native::is_field_writable_for(
            obj_val,
            field_name,
            &self.metadata,
            &self.class_metadata,
            &self.layouts,
            self.structural_object_shapes,
        )
    }

    pub(crate) fn sync_descriptor_value(&self, obj_val: Value, field_name: &str, value: Value) {
        // Delegates to the shared function; its middle step is already the shared
        // `object_field_index`, so this extraction adds no new resolution logic.
        crate::vm::interpreter::opcodes::native::sync_descriptor_value_for(
            obj_val,
            field_name,
            value,
            &self.metadata,
            &self.class_metadata,
            &self.layouts,
            self.structural_object_shapes,
        );
    }

    pub(crate) fn descriptor_accessor(
        &self,
        obj_val: Value,
        field_name: &str,
        accessor_name: &str,
    ) -> Option<Value> {
        // Delegates to the shared function, which the JIT keyed-access helpers will
        // call. A JIT helper CANNOT run an accessor -- the interpreter calls it as a
        // frame -- so for that caller the useful part is only `.is_some()`, meaning
        // "decline and let the interpreter run the frame".
        crate::vm::interpreter::opcodes::native::descriptor_accessor_for(
            obj_val,
            field_name,
            accessor_name,
            &self.metadata,
            &self.class_metadata,
            &self.layouts,
            self.structural_object_shapes,
        )
    }

    /// Narrow a receiver to something the field accessors can work on.
    ///
    /// **This rejects proxies, and that makes every `unwrap_proxy_target` call in
    /// this file unreachable for them.** The field handlers call this *before*
    /// their `unwrap_proxy_target`, so a proxy receiver raises
    /// `TypeError: Expected Object receiver for <context>, got UnknownGcType` and
    /// the unwrap never runs. Proxy field access does not work in either engine —
    /// the JIT helpers do not unwrap either, and return null instead of raising.
    ///
    /// So treat those eight unwrap sites as aspirational until a proxy receiver is
    /// admitted here. Whether it should be admitted at all is an open design
    /// question, not a missing `if`: unwrapping silently bypasses the proxy
    /// handler, and the TODO at the first unwrap site says full trap support would
    /// call `handler.get(target, fieldName)`. Adding a Proxy arm here without
    /// settling that would make a proxy read the target's field and never consult
    /// the handler, which may be worse than the current honest error.
    ///
    /// See /workspace/specs/2026-10-03-raya-d4-fixed-layout-objects.md and the
    /// characterization test
    /// `field_access_through_a_proxy_currently_raises_and_that_is_a_defect`.
    pub(in crate::vm::interpreter) fn ensure_object_receiver(
        value: Value,
        context: &'static str,
    ) -> Result<Value, VmError> {
        if !value.is_ptr() {
            return Err(VmError::TypeError(format!(
                "Expected object for {}",
                context
            )));
        }

        let header = unsafe {
            &*header_ptr_from_value_ptr(value.as_ptr::<u8>().unwrap().as_ptr())
        };
        if header.type_id() == std::any::TypeId::of::<Object>() {
            return Ok(value);
        }

        let kind = if header.type_id() == std::any::TypeId::of::<Array>() {
            "Array"
        } else if header.type_id() == std::any::TypeId::of::<RayaString>() {
            "RayaString"
        } else if header.type_id() == std::any::TypeId::of::<Closure>() {
            "Closure"
        } else if header.type_id() == std::any::TypeId::of::<BoundMethod>() {
            "BoundMethod"
        } else {
            "UnknownGcType"
        };

        Err(VmError::TypeError(format!(
            "Expected Object receiver for {}, got {}",
            context, kind
        )))
    }

    pub(in crate::vm::interpreter) fn exec_object_ops(
        &mut self,
        stack: &mut Stack,
        ip: &mut usize,
        code: &[u8],
        module: &Module,
        opcode: Opcode,
    ) -> OpcodeResult {
        match opcode {
            Opcode::NewType => {
                self.safepoint.poll();
                let local_class_index = match Self::read_u16(code, ip) {
                    Ok(v) => v as usize,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let nominal_type_id = match self.resolve_nominal_type_id(module, local_class_index)
                {
                    Ok(id) => id,
                    Err(error) => return OpcodeResult::Error(error),
                };

                let classes = self.classes.read();
                let (layout_id, field_count) = match self.nominal_allocation(nominal_type_id) {
                    Some(allocation) => allocation,
                    None => {
                        return OpcodeResult::Error(VmError::RuntimeError(format!(
                            "Invalid nominal type id: {}",
                            nominal_type_id
                        )));
                    }
                };
                drop(classes);

                let obj = Object::new_nominal(layout_id, nominal_type_id as u32, field_count);
                let gc_ptr = self.gc.lock().allocate(obj);
                let value =
                    unsafe { Value::from_ptr(std::ptr::NonNull::new(gc_ptr.as_ptr()).unwrap()) };
                if let Err(e) = stack.push(value) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::LoadFieldExact => {
                let field_offset = match Self::read_u16(code, ip) {
                    Ok(v) => v as usize,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let obj_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                let obj_val = match Self::ensure_object_receiver(obj_val, "field access") {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                // Check if the object is a proxy - if so, unwrap to target
                // TODO: Full trap support would call handler.get(target, fieldName)
                let actual_obj = crate::vm::reflect::unwrap_proxy_target(obj_val);

                let obj_ptr = unsafe { actual_obj.as_ptr::<Object>() };
                let obj = unsafe { &*obj_ptr.unwrap().as_ptr() };
                let slot_binding = StructuralSlotBinding::Field(field_offset);
                if let StructuralSlotBinding::Missing = slot_binding {
                    if let Err(e) = stack.push(Value::null()) {
                        return OpcodeResult::Error(e);
                    }
                    return OpcodeResult::Continue;
                }
                if let StructuralSlotBinding::Method(method_slot) = slot_binding {
                    let bound = match self.bound_method_value_for_slot(actual_obj, method_slot) {
                        Ok(value) => value,
                        Err(error) => return OpcodeResult::Error(error),
                    };
                    if let Err(e) = stack.push(bound) {
                        return OpcodeResult::Error(e);
                    }
                    return OpcodeResult::Continue;
                }
                let field_offset = match slot_binding {
                    StructuralSlotBinding::Field(offset) => offset,
                    StructuralSlotBinding::Method(_)
                    | StructuralSlotBinding::Dynamic(_)
                    | StructuralSlotBinding::Missing => {
                        unreachable!()
                    }
                };
                if let Some(field_name) = self.field_name_for_offset(obj, field_offset) {
                    if let Some(getter) = self.descriptor_accessor(actual_obj, &field_name, "get") {
                        match self.callable_frame_for_value(
                            getter,
                            stack,
                            &[],
                            ReturnAction::PushReturnValue,
                        ) {
                            Ok(Some(frame)) => return frame,
                            Ok(None) => {
                                return OpcodeResult::Error(VmError::TypeError(format!(
                                    "Property '{}' getter is not callable",
                                    field_name
                                )));
                            }
                            Err(e) => return OpcodeResult::Error(e),
                        }
                    }
                }
                // Missing fields resolve to null. This matches object destructuring defaults
                // and allows optional object properties to be absent at runtime.
                let value = obj.get_field(field_offset).unwrap_or(Value::null());
                if std::env::var("RAYA_DEBUG_FIELD_TRACE").is_ok() {
                    let class_debug = obj
                        .nominal_type_id_usize()
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "structural".to_string());
                    eprintln!(
                        "[field-trace] LoadFieldExact[{}] nominal_type_id={} field_count={} => {:?} (is_ptr={})",
                        field_offset,
                        class_debug,
                        obj.field_count(),
                        value,
                        value.is_ptr()
                    );
                }
                if let Err(e) = stack.push(value) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::LoadFieldShape => {
                let shape_id = match Self::read_u64(code, ip) {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let field_offset = match Self::read_u16(code, ip) {
                    Ok(v) => v as usize,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let obj_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                if let Some(value) = self.load_shape_field_on_non_object(obj_val, shape_id, field_offset)
                {
                    if let Err(e) = stack.push(value) {
                        return OpcodeResult::Error(e);
                    }
                    return OpcodeResult::Continue;
                }

                let obj_val = match Self::ensure_object_receiver(obj_val, "shape field access") {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                let actual_obj = crate::vm::reflect::unwrap_proxy_target(obj_val);
                let obj_ptr = unsafe { actual_obj.as_ptr::<Object>() };
                let obj = unsafe { &*obj_ptr.unwrap().as_ptr() };
                self.record_aot_shape_site(
                    crate::aot_profile::AotSiteKind::LoadFieldShape,
                    obj.layout_id(),
                );
                let slot_binding = self.remap_shape_slot_binding(obj, shape_id, field_offset);
                if let StructuralSlotBinding::Missing = slot_binding {
                    if let Err(e) = stack.push(Value::null()) {
                        return OpcodeResult::Error(e);
                    }
                    return OpcodeResult::Continue;
                }
                if let StructuralSlotBinding::Dynamic(key) = slot_binding {
                    let value = obj
                        .dyn_map()
                        .and_then(|dyn_map| dyn_map.get(&key).copied())
                        .unwrap_or(Value::null());
                    if let Err(e) = stack.push(value) {
                        return OpcodeResult::Error(e);
                    }
                    return OpcodeResult::Continue;
                }
                if let StructuralSlotBinding::Method(method_slot) = slot_binding {
                    let bound = match self.bound_method_value_for_slot(actual_obj, method_slot) {
                        Ok(value) => value,
                        Err(error) => return OpcodeResult::Error(error),
                    };
                    if let Err(e) = stack.push(bound) {
                        return OpcodeResult::Error(e);
                    }
                    return OpcodeResult::Continue;
                }
                let field_offset = match slot_binding {
                    StructuralSlotBinding::Field(offset) => offset,
                    StructuralSlotBinding::Method(_)
                    | StructuralSlotBinding::Dynamic(_)
                    | StructuralSlotBinding::Missing => {
                        unreachable!()
                    }
                };
                if let Some(field_name) = self.field_name_for_offset(obj, field_offset) {
                    if let Some(getter) = self.descriptor_accessor(actual_obj, &field_name, "get") {
                        match self.callable_frame_for_value(
                            getter,
                            stack,
                            &[],
                            ReturnAction::PushReturnValue,
                        ) {
                            Ok(Some(frame)) => return frame,
                            Ok(None) => {
                                return OpcodeResult::Error(VmError::TypeError(format!(
                                    "Property '{}' getter is not callable",
                                    field_name
                                )));
                            }
                            Err(e) => return OpcodeResult::Error(e),
                        }
                    }
                }
                let value = obj.get_field(field_offset).unwrap_or(Value::null());
                if let Err(e) = stack.push(value) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::StoreFieldExact => {
                let field_offset = match Self::read_u16(code, ip) {
                    Ok(v) => v as usize,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let value = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let obj_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                let obj_val = match Self::ensure_object_receiver(obj_val, "field access") {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                // Check if the object is a proxy - if so, unwrap to target
                // TODO: Full trap support would call handler.set(target, fieldName, value)
                let actual_obj = crate::vm::reflect::unwrap_proxy_target(obj_val);

                let obj_ptr = unsafe { actual_obj.as_ptr::<Object>() };
                let obj = unsafe { &mut *obj_ptr.unwrap().as_ptr() };
                let slot_binding = StructuralSlotBinding::Field(field_offset);
                let field_offset = match slot_binding {
                    StructuralSlotBinding::Field(offset) => offset,
                    StructuralSlotBinding::Dynamic(_) => {
                        return OpcodeResult::Error(VmError::TypeError(
                            "Cannot assign to dynamic binding through fixed field store"
                                .to_string(),
                        ));
                    }
                    StructuralSlotBinding::Method(_) => {
                        return OpcodeResult::Error(VmError::TypeError(
                            "Cannot assign to structural method slot".to_string(),
                        ));
                    }
                    StructuralSlotBinding::Missing => {
                        return OpcodeResult::Error(VmError::TypeError(
                            "Cannot write field not present in structural slot view".to_string(),
                        ));
                    }
                };
                if let Some(field_name) = self.field_name_for_offset(obj, field_offset) {
                    if let Some(setter) = self.descriptor_accessor(actual_obj, &field_name, "set") {
                        match self.callable_frame_for_value(
                            setter,
                            stack,
                            &[value],
                            ReturnAction::Discard,
                        ) {
                            Ok(Some(frame)) => return frame,
                            Ok(None) => {
                                return OpcodeResult::Error(VmError::TypeError(format!(
                                    "Property '{}' setter is not callable",
                                    field_name
                                )));
                            }
                            Err(e) => return OpcodeResult::Error(e),
                        }
                    }
                    if self
                        .descriptor_accessor(actual_obj, &field_name, "get")
                        .is_some()
                        && !self.is_field_writable(actual_obj, &field_name)
                    {
                        return OpcodeResult::Error(VmError::TypeError(format!(
                            "Cannot set property '{}' which has only a getter",
                            field_name
                        )));
                    }
                    if !self.is_field_writable(actual_obj, &field_name) {
                        return OpcodeResult::Error(VmError::TypeError(format!(
                            "Cannot assign to non-writable property '{}'",
                            field_name
                        )));
                    }
                }
                if let Err(e) = obj.checked_set_field(field_offset, value) {
                    return OpcodeResult::Error(VmError::RuntimeError(e.to_string()));
                }
                if let Some(field_name) = self.field_name_for_offset(obj, field_offset) {
                    self.sync_descriptor_value(actual_obj, &field_name, value);
                }
                OpcodeResult::Continue
            }

            Opcode::StoreFieldShape => {
                let shape_id = match Self::read_u64(code, ip) {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let field_offset = match Self::read_u16(code, ip) {
                    Ok(v) => v as usize,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let value = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let obj_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                let obj_val = match Self::ensure_object_receiver(obj_val, "shape field access") {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                let actual_obj = crate::vm::reflect::unwrap_proxy_target(obj_val);
                let obj_ptr = unsafe { actual_obj.as_ptr::<Object>() };
                let obj = unsafe { &mut *obj_ptr.unwrap().as_ptr() };
                self.record_aot_shape_site(
                    crate::aot_profile::AotSiteKind::StoreFieldShape,
                    obj.layout_id(),
                );
                let slot_binding = self.remap_shape_slot_binding(obj, shape_id, field_offset);
                let field_offset = match slot_binding {
                    StructuralSlotBinding::Field(offset) => offset,
                    StructuralSlotBinding::Dynamic(key) => {
                        obj.ensure_dyn_map().insert(key, value);
                        return OpcodeResult::Continue;
                    }
                    StructuralSlotBinding::Method(_) => {
                        return OpcodeResult::Error(VmError::TypeError(
                            "Cannot assign to structural method slot".to_string(),
                        ));
                    }
                    StructuralSlotBinding::Missing => {
                        return OpcodeResult::Error(VmError::TypeError(
                            "Cannot write field not present in structural shape view".to_string(),
                        ));
                    }
                };
                if let Some(field_name) = self.field_name_for_offset(obj, field_offset) {
                    if let Some(setter) = self.descriptor_accessor(actual_obj, &field_name, "set") {
                        match self.callable_frame_for_value(
                            setter,
                            stack,
                            &[value],
                            ReturnAction::Discard,
                        ) {
                            Ok(Some(frame)) => return frame,
                            Ok(None) => {
                                return OpcodeResult::Error(VmError::TypeError(format!(
                                    "Property '{}' setter is not callable",
                                    field_name
                                )));
                            }
                            Err(e) => return OpcodeResult::Error(e),
                        }
                    }
                    if self
                        .descriptor_accessor(actual_obj, &field_name, "get")
                        .is_some()
                        && !self.is_field_writable(actual_obj, &field_name)
                    {
                        return OpcodeResult::Error(VmError::TypeError(format!(
                            "Cannot set property '{}' which has only a getter",
                            field_name
                        )));
                    }
                    if !self.is_field_writable(actual_obj, &field_name) {
                        return OpcodeResult::Error(VmError::TypeError(format!(
                            "Cannot assign to non-writable property '{}'",
                            field_name
                        )));
                    }
                }
                if let Err(e) = obj.checked_set_field(field_offset, value) {
                    return OpcodeResult::Error(VmError::RuntimeError(e.to_string()));
                }
                if let Some(field_name) = self.field_name_for_offset(obj, field_offset) {
                    self.sync_descriptor_value(actual_obj, &field_name, value);
                }
                OpcodeResult::Continue
            }

            Opcode::OptionalFieldExact => {
                let field_offset = match Self::read_u16(code, ip) {
                    Ok(v) => v as usize,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let obj_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                // If null, return null (optional chaining semantics)
                if obj_val.is_null() {
                    if let Err(e) = stack.push(Value::null()) {
                        return OpcodeResult::Error(e);
                    }
                    return OpcodeResult::Continue;
                }

                let obj_val = match Self::ensure_object_receiver(obj_val, "optional field access") {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                // Check if the object is a proxy - if so, unwrap to target
                let actual_obj = crate::vm::reflect::unwrap_proxy_target(obj_val);

                let obj_ptr = unsafe { actual_obj.as_ptr::<Object>() };
                let obj = unsafe { &*obj_ptr.unwrap().as_ptr() };
                let slot_binding = StructuralSlotBinding::Field(field_offset);
                if let StructuralSlotBinding::Missing = slot_binding {
                    if let Err(e) = stack.push(Value::null()) {
                        return OpcodeResult::Error(e);
                    }
                    return OpcodeResult::Continue;
                }
                if let StructuralSlotBinding::Method(method_slot) = slot_binding {
                    let bound = match self.bound_method_value_for_slot(actual_obj, method_slot) {
                        Ok(value) => value,
                        Err(error) => return OpcodeResult::Error(error),
                    };
                    if let Err(e) = stack.push(bound) {
                        return OpcodeResult::Error(e);
                    }
                    return OpcodeResult::Continue;
                }
                let field_offset = match slot_binding {
                    StructuralSlotBinding::Field(offset) => offset,
                    StructuralSlotBinding::Method(_)
                    | StructuralSlotBinding::Dynamic(_)
                    | StructuralSlotBinding::Missing => {
                        unreachable!()
                    }
                };
                let value = obj.get_field(field_offset).unwrap_or(Value::null());
                if let Err(e) = stack.push(value) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::OptionalFieldShape => {
                let shape_id = match Self::read_u64(code, ip) {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let field_offset = match Self::read_u16(code, ip) {
                    Ok(v) => v as usize,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let obj_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                if obj_val.is_null() {
                    if let Err(e) = stack.push(Value::null()) {
                        return OpcodeResult::Error(e);
                    }
                    return OpcodeResult::Continue;
                }

                if let Some(value) = self.load_shape_field_on_non_object(obj_val, shape_id, field_offset)
                {
                    if let Err(e) = stack.push(value) {
                        return OpcodeResult::Error(e);
                    }
                    return OpcodeResult::Continue;
                }

                let obj_val =
                    match Self::ensure_object_receiver(obj_val, "optional shape field access") {
                        Ok(v) => v,
                        Err(e) => return OpcodeResult::Error(e),
                    };

                let actual_obj = crate::vm::reflect::unwrap_proxy_target(obj_val);
                let obj_ptr = unsafe { actual_obj.as_ptr::<Object>() };
                let obj = unsafe { &*obj_ptr.unwrap().as_ptr() };
                let slot_binding = self.remap_shape_slot_binding(obj, shape_id, field_offset);
                if let StructuralSlotBinding::Missing = slot_binding {
                    if let Err(e) = stack.push(Value::null()) {
                        return OpcodeResult::Error(e);
                    }
                    return OpcodeResult::Continue;
                }
                if let StructuralSlotBinding::Dynamic(key) = slot_binding {
                    let value = obj
                        .dyn_map()
                        .and_then(|dyn_map| dyn_map.get(&key).copied())
                        .unwrap_or(Value::null());
                    if let Err(e) = stack.push(value) {
                        return OpcodeResult::Error(e);
                    }
                    return OpcodeResult::Continue;
                }
                if let StructuralSlotBinding::Method(method_slot) = slot_binding {
                    let bound = match self.bound_method_value_for_slot(actual_obj, method_slot) {
                        Ok(value) => value,
                        Err(error) => return OpcodeResult::Error(error),
                    };
                    if let Err(e) = stack.push(bound) {
                        return OpcodeResult::Error(e);
                    }
                    return OpcodeResult::Continue;
                }
                let field_offset = match slot_binding {
                    StructuralSlotBinding::Field(offset) => offset,
                    StructuralSlotBinding::Method(_)
                    | StructuralSlotBinding::Dynamic(_)
                    | StructuralSlotBinding::Missing => {
                        unreachable!()
                    }
                };
                let value = obj.get_field(field_offset).unwrap_or(Value::null());
                if let Err(e) = stack.push(value) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::ObjectLiteral => {
                self.safepoint.poll();
                let layout_id = match Self::read_u32(code, ip) {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let field_count = match Self::read_u16(code, ip) {
                    Ok(v) => v as usize,
                    Err(e) => return OpcodeResult::Error(e),
                };
                if layout_id == 0 {
                    return OpcodeResult::Error(VmError::RuntimeError(
                        "object literal is missing structural layout id".to_string(),
                    ));
                }

                let obj = Object::new_structural(layout_id, field_count);
                let gc_ptr = self.gc.lock().allocate(obj);
                let value =
                    unsafe { Value::from_ptr(std::ptr::NonNull::new(gc_ptr.as_ptr()).unwrap()) };
                if let Err(e) = stack.push(value) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::InitObject => {
                let field_offset = match Self::read_u16(code, ip) {
                    Ok(v) => v as usize,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let value = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let obj_val = match stack.peek() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                let obj_val = match Self::ensure_object_receiver(obj_val, "field initialization") {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                let obj_ptr = unsafe { obj_val.as_ptr::<Object>() };
                let obj = unsafe { &mut *obj_ptr.unwrap().as_ptr() };
                if let Err(e) = obj.checked_set_field(field_offset, value) {
                    return OpcodeResult::Error(VmError::RuntimeError(e.to_string()));
                }
                OpcodeResult::Continue
            }

            Opcode::BindMethod => {
                let method_slot = match Self::read_u16(code, ip) {
                    Ok(v) => v as usize,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let obj_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                let obj_val = match Self::ensure_object_receiver(obj_val, "method binding") {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                let obj = unsafe { &*obj_val.as_ptr::<Object>().unwrap().as_ptr() };
                let nominal_type_id = obj.nominal_type_id_usize().ok_or_else(|| {
                    VmError::TypeError("Cannot bind method on structural object value".to_string())
                });
                let nominal_type_id = match nominal_type_id {
                    Ok(id) => id,
                    Err(error) => return OpcodeResult::Error(error),
                };
                let classes = self.classes.read();
                let class = match classes.get_class(nominal_type_id) {
                    Some(c) => c,
                    None => {
                        return OpcodeResult::Error(VmError::RuntimeError(format!(
                            "Invalid nominal type id: {}",
                            nominal_type_id
                        )));
                    }
                };
                let func_id = match class.vtable.get_method(method_slot) {
                    Some(fid) => fid,
                    None => {
                        return OpcodeResult::Error(VmError::RuntimeError(format!(
                            "Invalid method slot: {} for class {}",
                            method_slot, class.name
                        )));
                    }
                };
                let method_module = class.module.clone();
                drop(classes);

                let bm = BoundMethod {
                    receiver: obj_val,
                    func_id,
                    module: method_module,
                };
                let gc_ptr = self.gc.lock().allocate(bm);
                let value =
                    unsafe { Value::from_ptr(std::ptr::NonNull::new(gc_ptr.as_ptr()).unwrap()) };
                if let Err(e) = stack.push(value) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            _ => OpcodeResult::Error(VmError::RuntimeError(format!(
                "Unexpected opcode in exec_object_ops: {:?}",
                opcode
            ))),
        }
    }
}
