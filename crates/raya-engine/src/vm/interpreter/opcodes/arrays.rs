//! Array opcode handlers: NewArray, LoadElem, StoreElem, ArrayLen, ArrayPush, ArrayPop, ArrayLiteral, InitArray
//!
//! Array receivers are resolved through [`raya_array_ptr_checked`], which
//! verifies the GC-header type identity so a string, object, or any other
//! allocation is never misread as an array. Mutations route through the
//! centralized checked setters on [`Array`] so element-type constraints cannot
//! be bypassed. Reads never validate element types: freshly allocated arrays
//! carry null-filled slots and rejecting those would be unsound.

use crate::compiler::bytecode::RuntimeTypeDescriptor;
use crate::compiler::{Module, Opcode};
use crate::vm::interpreter::execution::OpcodeResult;
use crate::vm::interpreter::Interpreter;
use crate::vm::object::{Array, ArrayStoreError};
use crate::vm::stack::Stack;
use crate::vm::value::Value;
use crate::vm::value_semantics::raya_array_ptr_checked;
use crate::vm::VmError;

/// Map a checked-store failure to the interpreter error taxonomy: bounds
/// violations are runtime errors, element-type violations are type errors.
#[inline]
fn store_error_to_vm(err: ArrayStoreError) -> VmError {
    match err {
        ArrayStoreError::OutOfBounds { .. } => VmError::RuntimeError(err.to_string()),
        ArrayStoreError::ElementType { .. } => VmError::TypeError(err.to_string()),
    }
}

impl<'a> Interpreter<'a> {
    pub(in crate::vm::interpreter) fn exec_array_ops(
        &mut self,
        stack: &mut Stack,
        ip: &mut usize,
        code: &[u8],
        module: &Module,
        opcode: Opcode,
    ) -> OpcodeResult {
        match opcode {
            Opcode::NewArray => {
                self.safepoint.poll();
                let type_index = match Self::read_u32(code, ip) {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let len = match stack.pop() {
                    Ok(v) => {
                        if let Some(i) = v.as_i32() {
                            i as usize
                        } else if let Some(f) = v.as_f64() {
                            f as usize
                        } else {
                            0
                        }
                    }
                    Err(e) => return OpcodeResult::Error(e),
                };

                let arr = Self::build_array(module, type_index, len);
                let gc_ptr = self.gc.lock().allocate(arr);
                let value =
                    unsafe { Value::from_ptr(std::ptr::NonNull::new(gc_ptr.as_ptr()).unwrap()) };
                if let Err(e) = stack.push(value) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::LoadElem => {
                let index = match stack.pop() {
                    Ok(v) => {
                        if let Some(i) = v.as_i32() {
                            i as usize
                        } else if let Some(f) = v.as_f64() {
                            f as usize
                        } else {
                            0
                        }
                    }
                    Err(e) => return OpcodeResult::Error(e),
                };
                let arr_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                let arr = match unsafe { raya_array_ptr_checked(arr_val) } {
                    Some(ptr) => unsafe { &*ptr.as_ptr() },
                    None => {
                        return OpcodeResult::Error(VmError::TypeError("Expected array".to_string()))
                    }
                };
                // Reads are never element-type checked; null-filled slots are
                // legitimate and must load as null.
                let value = match arr.get(index) {
                    Some(v) => v,
                    None => {
                        return OpcodeResult::Error(VmError::RuntimeError(format!(
                            "Array index {} out of bounds",
                            index
                        )));
                    }
                };
                if let Err(e) = stack.push(value) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::StoreElem => {
                let value = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let index = match stack.pop() {
                    Ok(v) => {
                        if let Some(i) = v.as_i32() {
                            i as usize
                        } else if let Some(f) = v.as_f64() {
                            f as usize
                        } else {
                            0
                        }
                    }
                    Err(e) => return OpcodeResult::Error(e),
                };
                let arr_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                let arr = match unsafe { raya_array_ptr_checked(arr_val) } {
                    Some(ptr) => unsafe { &mut *ptr.as_ptr() },
                    None => {
                        return OpcodeResult::Error(VmError::TypeError("Expected array".to_string()))
                    }
                };
                if let Err(e) = arr.checked_set(index, value) {
                    return OpcodeResult::Error(store_error_to_vm(e));
                }
                OpcodeResult::Continue
            }

            Opcode::ArrayLen => {
                let arr_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                let arr = match unsafe { raya_array_ptr_checked(arr_val) } {
                    Some(ptr) => unsafe { &*ptr.as_ptr() },
                    None => {
                        return OpcodeResult::Error(VmError::TypeError("Expected array".to_string()))
                    }
                };
                if let Err(e) = stack.push(Value::i32(arr.len() as i32)) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::ArrayPush => {
                // Stack: [array, value] -> [] (mutates array in-place)
                let element = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let arr_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                let arr = match unsafe { raya_array_ptr_checked(arr_val) } {
                    Some(ptr) => unsafe { &mut *ptr.as_ptr() },
                    None => {
                        return OpcodeResult::Error(VmError::TypeError("Expected array".to_string()))
                    }
                };
                if let Err(e) = arr.checked_push(element) {
                    return OpcodeResult::Error(store_error_to_vm(e));
                }
                OpcodeResult::Continue
            }

            Opcode::ArrayPop => {
                // Stack: [array] -> [popped_element]
                let arr_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                let arr = match unsafe { raya_array_ptr_checked(arr_val) } {
                    Some(ptr) => unsafe { &mut *ptr.as_ptr() },
                    None => {
                        return OpcodeResult::Error(VmError::TypeError("Expected array".to_string()))
                    }
                };
                let value = arr.pop().unwrap_or(Value::null());
                if let Err(e) = stack.push(value) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::ArrayLiteral => {
                self.safepoint.poll();
                let type_index = match Self::read_u32(code, ip) {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let length = match Self::read_u32(code, ip) {
                    Ok(v) => v as usize,
                    Err(e) => return OpcodeResult::Error(e),
                };

                // Pop elements from stack in reverse order (last pushed = last element)
                let mut elements = Vec::with_capacity(length);
                for _ in 0..length {
                    match stack.pop() {
                        Ok(v) => elements.push(v),
                        Err(e) => return OpcodeResult::Error(e),
                    }
                }
                // Reverse to get correct order (first pushed = first element)
                elements.reverse();

                // Create array with the elements, honoring the element constraint.
                let mut arr = Self::build_array(module, type_index, length);
                for (i, elem) in elements.into_iter().enumerate() {
                    if let Err(e) = arr.checked_set(i, elem) {
                        return OpcodeResult::Error(store_error_to_vm(e));
                    }
                }

                let gc_ptr = self.gc.lock().allocate(arr);
                let value =
                    unsafe { Value::from_ptr(std::ptr::NonNull::new(gc_ptr.as_ptr()).unwrap()) };
                if let Err(e) = stack.push(value) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::InitArray => {
                // Single-element initializer, mirroring InitObject:
                //   operand: u16 index
                //   stack:   [.., array, value] -> [.., array]
                // The array stays on the stack so a run of InitArray ops can
                // populate it slot by slot.
                let index = match Self::read_u16(code, ip) {
                    Ok(v) => v as usize,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let value = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let arr_val = match stack.peek() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                let arr = match unsafe { raya_array_ptr_checked(arr_val) } {
                    Some(ptr) => unsafe { &mut *ptr.as_ptr() },
                    None => {
                        return OpcodeResult::Error(VmError::TypeError("Expected array".to_string()))
                    }
                };
                if let Err(e) = arr.checked_set(index, value) {
                    return OpcodeResult::Error(store_error_to_vm(e));
                }
                OpcodeResult::Continue
            }

            _ => OpcodeResult::Error(VmError::RuntimeError(format!(
                "Unexpected opcode in exec_array_ops: {:?}",
                opcode
            ))),
        }
    }

    /// Construct an array of `length` null-filled slots, resolving `type_index`
    /// (the emitted **element** descriptor id) into a self-contained element
    /// constraint.
    ///
    /// The dynamic id (`AnyValue`) and any descriptor that cannot be resolved
    /// leave the constraint unset (`None`), so the array behaves dynamically.
    /// Concrete primitive and complex element descriptors are stored as the
    /// resolved constraint.
    fn build_array(module: &Module, type_index: u32, length: usize) -> Array {
        let element_type = resolve_element_descriptor(module, type_index);
        Array::with_element_type(type_index as usize, element_type, length)
    }
}

/// Decode an emitted element descriptor id into a resolved constraint.
///
/// Ids `0..=7` are primitives; `>= COMPLEX_BASE` index the module's
/// `runtime_types` table. `AnyValue` maps to `None` (dynamic, accept-all) so
/// dynamic arrays never carry a spurious constraint. An id that does not
/// resolve also maps to `None` rather than fabricating a type.
fn resolve_element_descriptor(module: &Module, type_index: u32) -> Option<RuntimeTypeDescriptor> {
    use crate::compiler::bytecode::types::COMPLEX_BASE;
    let descriptor = if type_index < COMPLEX_BASE {
        match type_index {
            0 => RuntimeTypeDescriptor::I32,
            1 => RuntimeTypeDescriptor::F64,
            2 => RuntimeTypeDescriptor::Bool,
            3 => RuntimeTypeDescriptor::String,
            4 => RuntimeTypeDescriptor::Null,
            5 => RuntimeTypeDescriptor::Void,
            6 => return None, // AnyValue == dynamic
            7 => RuntimeTypeDescriptor::Ref,
            _ => return None,
        }
    } else {
        module
            .runtime_types
            .get((type_index - COMPLEX_BASE) as usize)
            .cloned()?
    };
    // A boxed/opaque resolved descriptor is equivalent to dynamic; keep it unset.
    match descriptor {
        RuntimeTypeDescriptor::AnyValue | RuntimeTypeDescriptor::Ref => None,
        other => Some(other),
    }
}

#[cfg(test)]
mod tests {
    //! End-to-end array opcode semantics through the interpreter. These pin the
    //! normalized `InitArray`, the GC-header receiver checks, and the
    //! null-slot read behavior; they are the reference for future JIT
    //! differential tests when array capability is promoted.

    use crate::compiler::bytecode::{Function, Module, Opcode};
    use crate::vm::value::Value;
    use crate::vm::Vm;
    use crate::vm::VmError;

    fn run(code: Vec<u8>) -> Result<Value, VmError> {
        let mut module = Module::new("arrays".to_string());
        module.functions.push(Function {
            signature_id: 0,
            local_types: Vec::new(),
            abi_version: 1,
            name: "main".to_string(),
            param_count: 0,
            local_count: 0,
            code,
        });
        Vm::new().execute(&module)
    }

    fn const_i32(v: i32) -> Vec<u8> {
        let mut c = vec![Opcode::ConstI32 as u8];
        c.extend_from_slice(&v.to_le_bytes());
        c
    }
    fn new_array(element_type_id: u32) -> Vec<u8> {
        // Consumes a length already on the stack.
        let mut c = vec![Opcode::NewArray as u8];
        c.extend_from_slice(&element_type_id.to_le_bytes());
        c
    }

    #[test]
    fn new_array_store_load_roundtrip() {
        // len=3 dynamic array; store 42 at index 1; load index 1.
        // StoreElem pops value, index, array, so keep a copy of the array via Dup.
        let mut code = const_i32(3);
        code.extend(new_array(6)); // stack: [arr]  (AnyValue element type)
        code.push(Opcode::Dup as u8); // [arr, arr]
        code.extend(const_i32(1)); // [arr, arr, 1]
        code.extend(const_i32(42)); // [arr, arr, 1, 42]
        code.push(Opcode::StoreElem as u8); // consumes arr,1,42 -> [arr]
        code.extend(const_i32(1)); // [arr, 1]
        code.push(Opcode::LoadElem as u8); // [42]
        code.push(Opcode::Return as u8);
        assert_eq!(run(code).unwrap().as_i32(), Some(42));
    }

    #[test]
    fn load_of_null_filled_slot_is_null_not_type_error() {
        // Fresh array slots are null; reading one must yield null, never a type error.
        let mut code = const_i32(2);
        code.extend(new_array(6));
        code.extend(const_i32(0));
        code.push(Opcode::LoadElem as u8);
        code.push(Opcode::Return as u8);
        assert!(run(code).unwrap().is_null());
    }

    #[test]
    fn array_len_reports_length() {
        let mut code = const_i32(5);
        code.extend(new_array(6));
        code.push(Opcode::ArrayLen as u8);
        code.push(Opcode::Return as u8);
        assert_eq!(run(code).unwrap().as_i32(), Some(5));
    }

    #[test]
    fn push_then_pop_and_len() {
        let mut code = const_i32(0);
        code.extend(new_array(6)); // [arr] len 0
        code.push(Opcode::Dup as u8); // [arr, arr]
        code.extend(const_i32(7)); // [arr, arr, 7]
        code.push(Opcode::ArrayPush as u8); // pushes 7 onto arr -> [arr]
        code.push(Opcode::ArrayPop as u8); // pops -> [7]
        code.push(Opcode::Return as u8);
        assert_eq!(run(code).unwrap().as_i32(), Some(7));
    }

    #[test]
    fn store_elem_on_non_array_is_type_error() {
        // A bare integer is not an array; StoreElem must raise a TypeError
        // rather than blindly casting the pointer.
        let mut code = const_i32(99); // "array" (actually an int)
        code.extend(const_i32(0)); // index
        code.extend(const_i32(1)); // value
        code.push(Opcode::StoreElem as u8);
        code.push(Opcode::Return as u8);
        let err = run(code).unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("array")
                || err.to_string().to_lowercase().contains("type"),
            "expected a type error, got: {err}"
        );
    }

    #[test]
    fn array_len_on_non_array_is_type_error() {
        let mut code = const_i32(3); // not an array
        code.push(Opcode::ArrayLen as u8);
        code.push(Opcode::Return as u8);
        assert!(run(code).is_err());
    }

    #[test]
    fn init_array_is_single_element_u16_index_initializer() {
        // InitArray operand is a u16 index; it pops one value, peeks the array,
        // stores value at index, and leaves the array on the stack. Two
        // InitArray ops populate two slots.
        let mut code = const_i32(2);
        code.extend(new_array(6)); // [arr]
        code.extend(const_i32(10)); // [arr, 10]
        code.push(Opcode::InitArray as u8); // operand u16 index 0 -> [arr]
        code.extend_from_slice(&0u16.to_le_bytes());
        code.extend(const_i32(20)); // [arr, 20]
        code.push(Opcode::InitArray as u8); // index 1 -> [arr]
        code.extend_from_slice(&1u16.to_le_bytes());
        // load index 1 -> expect 20
        code.extend(const_i32(1));
        code.push(Opcode::LoadElem as u8);
        code.push(Opcode::Return as u8);
        assert_eq!(run(code).unwrap().as_i32(), Some(20));
    }

    #[test]
    fn array_literal_builds_populated_array() {
        // ArrayLiteral: u32 element_type_id, u32 length; pops `length` elements.
        let mut code = const_i32(11);
        code.extend(const_i32(22));
        code.push(Opcode::ArrayLiteral as u8);
        code.extend_from_slice(&6u32.to_le_bytes()); // AnyValue
        code.extend_from_slice(&2u32.to_le_bytes()); // length 2
        // [arr]; load index 0 -> 11
        code.extend(const_i32(0));
        code.push(Opcode::LoadElem as u8);
        code.push(Opcode::Return as u8);
        assert_eq!(run(code).unwrap().as_i32(), Some(11));
    }
}
