//! Array built-in method handlers
//!
//! Every receiver is resolved through [`raya_array_ptr_checked`], which verifies
//! the GC-header type identity, so a string, object, or any other allocation is
//! never reinterpreted as an `Array`. Mutating built-ins route through the
//! centralized checked mutation APIs on [`Array`] (`checked_push`,
//! `checked_unshift`, `checked_fill`, `checked_splice`) so element-type
//! constraints cannot be bypassed. Result arrays preserve the source element
//! constraint where the operation preserves element types.

use crate::compiler::Module;
use crate::vm::interpreter::Interpreter;
use crate::vm::object::{Array, ArrayStoreError, RayaString};
use crate::vm::scheduler::Task;
use crate::vm::stack::Stack;
use crate::vm::value::Value;
use crate::vm::value_semantics::{raya_array_ptr_checked, raya_string_ptr_checked};
use crate::vm::VmError;
use std::sync::Arc;

/// Resolve an array receiver `Value` to a shared reference, rejecting any
/// non-array with a `TypeError`.
#[inline]
unsafe fn array_ref<'r>(value: Value) -> Result<&'r Array, VmError> {
    match raya_array_ptr_checked(value) {
        Some(ptr) => Ok(&*ptr.as_ptr()),
        None => Err(VmError::TypeError("Expected array".to_string())),
    }
}

/// Resolve an array receiver `Value` to a mutable reference, rejecting any
/// non-array with a `TypeError`.
#[inline]
unsafe fn array_ref_mut<'r>(value: Value) -> Result<&'r mut Array, VmError> {
    match raya_array_ptr_checked(value) {
        Some(ptr) => Ok(&mut *ptr.as_ptr()),
        None => Err(VmError::TypeError("Expected array".to_string())),
    }
}

/// Map a checked-store failure to the interpreter error taxonomy.
#[inline]
fn store_error_to_vm(err: ArrayStoreError) -> VmError {
    match err {
        ArrayStoreError::OutOfBounds { .. } => VmError::RuntimeError(err.to_string()),
        ArrayStoreError::ElementType { .. } => VmError::TypeError(err.to_string()),
    }
}

/// Value equality used by `indexOf`/`lastIndexOf`/`includes`: string pointers
/// compare by content (checked via the GC header), everything else by identity.
fn values_equal(a: &Value, b: &Value) -> bool {
    if a == b {
        return true;
    }
    let (Some(a_ptr), Some(b_ptr)) =
        (unsafe { raya_string_ptr_checked(*a) }, unsafe {
            raya_string_ptr_checked(*b)
        })
    else {
        return false;
    };
    let a_str = unsafe { &*a_ptr.as_ptr() };
    let b_str = unsafe { &*b_ptr.as_ptr() };
    a_str.data == b_str.data
}

impl<'a> Interpreter<'a> {
    /// Handle built-in array methods
    pub(in crate::vm::interpreter) fn call_array_method(
        &mut self,
        _task: &Arc<Task>,
        stack: &mut Stack,
        method_id: u16,
        arg_count: usize,
        _module: &Module,
    ) -> Result<(), VmError> {
        use crate::vm::builtin::array;

        match method_id {
            array::PUSH => {
                if arg_count != 1 {
                    return Err(VmError::RuntimeError(format!(
                        "Array.push expects 1 argument, got {}",
                        arg_count
                    )));
                }
                let value = stack.pop()?;
                let array_val = stack.pop()?;
                let arr = unsafe { array_ref_mut(array_val)? };
                let new_len = arr.checked_push(value).map_err(store_error_to_vm)?;
                stack.push(Value::i32(new_len as i32))?;
                Ok(())
            }
            array::POP => {
                if arg_count != 0 {
                    return Err(VmError::RuntimeError(format!(
                        "Array.pop expects 0 arguments, got {}",
                        arg_count
                    )));
                }
                let array_val = stack.pop()?;
                let arr = unsafe { array_ref_mut(array_val)? };
                let result = arr.pop().unwrap_or(Value::null());
                stack.push(result)?;
                Ok(())
            }
            array::SHIFT => {
                if arg_count != 0 {
                    return Err(VmError::RuntimeError(format!(
                        "Array.shift expects 0 arguments, got {}",
                        arg_count
                    )));
                }
                let array_val = stack.pop()?;
                let arr = unsafe { array_ref_mut(array_val)? };
                let result = arr.shift().unwrap_or(Value::null());
                stack.push(result)?;
                Ok(())
            }
            array::UNSHIFT => {
                if arg_count != 1 {
                    return Err(VmError::RuntimeError(format!(
                        "Array.unshift expects 1 argument, got {}",
                        arg_count
                    )));
                }
                let value = stack.pop()?;
                let array_val = stack.pop()?;
                let arr = unsafe { array_ref_mut(array_val)? };
                let new_len = arr.checked_unshift(value).map_err(store_error_to_vm)?;
                stack.push(Value::i32(new_len as i32))?;
                Ok(())
            }
            array::INDEX_OF => {
                if !(1..=2).contains(&arg_count) {
                    return Err(VmError::RuntimeError(format!(
                        "Array.indexOf expects 1-2 arguments, got {}",
                        arg_count
                    )));
                }
                let from_index = if arg_count == 2 {
                    let v = stack.pop()?;
                    v.as_i32().unwrap_or(0).max(0) as usize
                } else {
                    0
                };
                let value = stack.pop()?;
                let array_val = stack.pop()?;
                let arr = unsafe { array_ref(array_val)? };
                let mut result: i32 = -1;
                for (i, elem) in arr.elements.iter().enumerate().skip(from_index) {
                    if values_equal(elem, &value) {
                        result = i as i32;
                        break;
                    }
                }
                stack.push(Value::i32(result))?;
                Ok(())
            }
            array::INCLUDES => {
                if arg_count != 1 {
                    return Err(VmError::RuntimeError(format!(
                        "Array.includes expects 1 argument, got {}",
                        arg_count
                    )));
                }
                let value = stack.pop()?;
                let array_val = stack.pop()?;
                let arr = unsafe { array_ref(array_val)? };
                let result = arr.elements.iter().any(|elem| values_equal(elem, &value));
                stack.push(Value::bool(result))?;
                Ok(())
            }
            array::SLICE => {
                // slice(start, end?) - arg_count is 1 or 2
                // Supports negative indices: -1 = last element, etc.
                let end_val = if arg_count >= 2 {
                    Some(stack.pop()?)
                } else {
                    None
                };
                let start_val = if arg_count >= 1 {
                    stack.pop()?
                } else {
                    Value::i32(0)
                };
                let array_val = stack.pop()?;
                let arr = unsafe { array_ref(array_val)? };

                let len = arr.len();
                let start_raw = start_val.as_i32().unwrap_or(0);
                let start = if start_raw < 0 {
                    ((len as i32 + start_raw).max(0) as usize).min(len)
                } else {
                    (start_raw as usize).min(len)
                };
                let end = end_val
                    .and_then(|v| v.as_i32())
                    .map(|e| {
                        if e < 0 {
                            ((len as i32 + e).max(0) as usize).min(len)
                        } else {
                            (e as usize).min(len)
                        }
                    })
                    .unwrap_or(len);

                // slice preserves element type: same elements, same constraint.
                let mut new_arr =
                    Array::with_element_type(arr.type_id, arr.element_type.clone(), 0);
                if start < end {
                    for i in start..end {
                        if let Some(v) = arr.get(i) {
                            new_arr.push(v);
                        }
                    }
                }
                let gc_ptr = self.gc.lock().allocate(new_arr);
                let value =
                    unsafe { Value::from_ptr(std::ptr::NonNull::new(gc_ptr.as_ptr()).unwrap()) };
                stack.push(value)?;
                Ok(())
            }
            array::SPLICE => {
                // splice(start, deleteCount?, ...items): remove and optionally insert.
                if arg_count < 1 {
                    return Err(VmError::RuntimeError(format!(
                        "Array.splice expects at least 1 argument, got {}",
                        arg_count
                    )));
                }
                let mut items = Vec::new();
                for _ in 2..arg_count {
                    items.push(stack.pop()?);
                }
                items.reverse();
                let delete_count_val = if arg_count >= 2 {
                    Some(stack.pop()?)
                } else {
                    None
                };
                let start_val = stack.pop()?;
                let array_val = stack.pop()?;
                let arr = unsafe { array_ref_mut(array_val)? };

                let len = arr.len();
                let start = (start_val.as_i32().unwrap_or(0).max(0) as usize).min(len);
                let delete_count = if let Some(dc_val) = delete_count_val {
                    dc_val.as_i32().unwrap_or(0).max(0) as usize
                } else {
                    len.saturating_sub(start)
                };
                let end = (start + delete_count).min(len);

                // Inserted items are constraint-checked; removed elements share
                // the source element type.
                let element_type = arr.element_type.clone();
                let removed_vals = arr
                    .checked_splice(start, end, items)
                    .map_err(store_error_to_vm)?;

                let mut removed =
                    Array::with_element_type(arr.type_id, element_type, removed_vals.len());
                for (i, v) in removed_vals.iter().enumerate() {
                    removed.elements[i] = *v;
                }
                let gc_ptr = self.gc.lock().allocate(removed);
                let value =
                    unsafe { Value::from_ptr(std::ptr::NonNull::new(gc_ptr.as_ptr()).unwrap()) };
                stack.push(value)?;
                Ok(())
            }
            array::REVERSE => {
                if arg_count != 0 {
                    return Err(VmError::RuntimeError(format!(
                        "Array.reverse expects 0 arguments, got {}",
                        arg_count
                    )));
                }
                let array_val = stack.pop()?;
                let arr = unsafe { array_ref_mut(array_val)? };
                arr.elements.reverse();
                stack.push(array_val)?;
                Ok(())
            }
            array::CONCAT => {
                // concat(other): merge two arrays.
                if arg_count != 1 {
                    return Err(VmError::RuntimeError(format!(
                        "Array.concat expects 1 argument, got {}",
                        arg_count
                    )));
                }
                let other_val = stack.pop()?;
                let array_val = stack.pop()?;
                let arr = unsafe { array_ref(array_val)? };
                let other = unsafe { array_ref(other_val)? };

                // The result is dynamic unless both operands share the same
                // resolved element constraint.
                let element_type = if arr.element_type == other.element_type {
                    arr.element_type.clone()
                } else {
                    None
                };
                let mut new_arr = Array::with_element_type(arr.type_id, element_type, 0);
                for elem in arr.elements.iter() {
                    new_arr.push(*elem);
                }
                for elem in other.elements.iter() {
                    new_arr.push(*elem);
                }
                let gc_ptr = self.gc.lock().allocate(new_arr);
                let value =
                    unsafe { Value::from_ptr(std::ptr::NonNull::new(gc_ptr.as_ptr()).unwrap()) };
                stack.push(value)?;
                Ok(())
            }
            array::LAST_INDEX_OF => {
                if !(1..=2).contains(&arg_count) {
                    return Err(VmError::RuntimeError(format!(
                        "Array.lastIndexOf expects 1-2 arguments, got {}",
                        arg_count
                    )));
                }
                let from_index = if arg_count == 2 {
                    let v = stack.pop()?;
                    Some(v.as_i32().unwrap_or(0).max(0) as usize)
                } else {
                    None
                };
                let search_val = stack.pop()?;
                let array_val = stack.pop()?;
                let arr = unsafe { array_ref(array_val)? };

                let end = from_index.unwrap_or(arr.elements.len().saturating_sub(1));
                let mut found_index: i32 = -1;
                for i in (0..=end.min(arr.elements.len().saturating_sub(1))).rev() {
                    if values_equal(&arr.elements[i], &search_val) {
                        found_index = i as i32;
                        break;
                    }
                }
                stack.push(Value::i32(found_index))?;
                Ok(())
            }
            array::FILL => {
                // fill(value, start?, end?): fill with value.
                if !(1..=3).contains(&arg_count) {
                    return Err(VmError::RuntimeError(format!(
                        "Array.fill expects 1-3 arguments, got {}",
                        arg_count
                    )));
                }
                let mut args = Vec::with_capacity(arg_count);
                for _ in 0..arg_count {
                    args.push(stack.pop()?);
                }
                args.reverse();
                let array_val = stack.pop()?;
                let arr = unsafe { array_ref_mut(array_val)? };

                let fill_value = args[0];
                let start = if arg_count >= 2 {
                    args[1].as_i32().unwrap_or(0).max(0) as usize
                } else {
                    0
                };
                let end = if arg_count >= 3 {
                    args[2].as_i32().unwrap_or(arr.len() as i32).max(0) as usize
                } else {
                    arr.len()
                };
                arr.checked_fill(fill_value, start, end)
                    .map_err(store_error_to_vm)?;
                stack.push(array_val)?;
                Ok(())
            }
            array::FLAT => {
                // flat(depth?): flatten nested arrays.
                let depth = if arg_count >= 1 {
                    let d = stack.pop()?.as_i32().unwrap_or(1);
                    d.max(0) as usize
                } else {
                    1
                };
                let array_val = stack.pop()?;
                let arr = unsafe { array_ref(array_val)? };

                // Flattening can mix element types, so the result is dynamic.
                fn flatten(arr: &Array, depth: usize, out: &mut Array) {
                    for elem in arr.elements.iter() {
                        if depth > 0 {
                            if let Some(ptr) = unsafe { raya_array_ptr_checked(*elem) } {
                                let inner = unsafe { &*ptr.as_ptr() };
                                flatten(inner, depth - 1, out);
                                continue;
                            }
                        }
                        out.push(*elem);
                    }
                }
                let mut result = Array::new(0, 0);
                flatten(arr, depth, &mut result);
                let gc_ptr = self.gc.lock().allocate(result);
                let value =
                    unsafe { Value::from_ptr(std::ptr::NonNull::new(gc_ptr.as_ptr()).unwrap()) };
                stack.push(value)?;
                Ok(())
            }
            // NOTE: SORT, REDUCE, FILTER, MAP, FIND, FIND_INDEX, FOR_EACH, EVERY, SOME
            // are now compiled as inline loops by the compiler (see lower_array_intrinsic in expr.rs)
            // and never reach this handler at runtime.
            array::JOIN => {
                // join(separator?) - arg_count is 0 or 1
                let sep = if arg_count >= 1 {
                    let sep_val = stack.pop()?;
                    if let Some(ptr) = unsafe { raya_string_ptr_checked(sep_val) } {
                        let s = unsafe { &*ptr.as_ptr() };
                        s.data.clone()
                    } else {
                        ",".to_string()
                    }
                } else {
                    ",".to_string()
                };
                let array_val = stack.pop()?;
                let arr = unsafe { array_ref(array_val)? };

                let parts: Vec<String> = arr
                    .elements
                    .iter()
                    .map(|v| {
                        if let Some(ptr) = unsafe { raya_string_ptr_checked(*v) } {
                            unsafe { &*ptr.as_ptr() }.data.clone()
                        } else if let Some(i) = v.as_i32() {
                            i.to_string()
                        } else if let Some(f) = v.as_f64() {
                            f.to_string()
                        } else if v.is_null() {
                            String::new()
                        } else if let Some(b) = v.as_bool() {
                            b.to_string()
                        } else {
                            String::new()
                        }
                    })
                    .collect();
                let result = parts.join(&sep);
                let raya_string = RayaString::new(result);
                let gc_ptr = self.gc.lock().allocate(raya_string);
                let value =
                    unsafe { Value::from_ptr(std::ptr::NonNull::new(gc_ptr.as_ptr()).unwrap()) };
                stack.push(value)?;
                Ok(())
            }
            // NOTE: FILTER, MAP, FIND, FIND_INDEX, FOR_EACH, EVERY, SOME, SORT, REDUCE
            // are now compiled as inline loops by the compiler (see lower_array_intrinsic in expr.rs)
            // and never reach this handler at runtime.
            _ => Err(VmError::RuntimeError(format!(
                "Array method {:#06x} not yet implemented in Interpreter",
                method_id
            ))),
        }
    }
}
