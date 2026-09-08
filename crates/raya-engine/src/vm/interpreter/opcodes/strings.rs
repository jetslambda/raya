//! String opcode handlers: Sconcat, Slen, Seq, Sne, Slt, Sle, Sgt, Sge, ToString

use crate::compiler::Opcode;
use crate::vm::interpreter::execution::OpcodeResult;
use crate::vm::interpreter::Interpreter;
use crate::vm::object::RayaString;
use crate::vm::stack::Stack;
use crate::vm::value::Value;
use crate::vm::value_semantics::{compare_strings, raya_string_ptr_checked, value_to_string};
use crate::vm::VmError;
use std::cmp::Ordering;

impl<'a> Interpreter<'a> {
    pub(in crate::vm::interpreter) fn exec_string_ops(
        &mut self,
        stack: &mut Stack,
        opcode: Opcode,
    ) -> OpcodeResult {
        match opcode {
            Opcode::Sconcat => {
                let right = match stack.pop() {
                    Ok(value) => value,
                    Err(error) => return OpcodeResult::Error(error),
                };
                let left = match stack.pop() {
                    Ok(value) => value,
                    Err(error) => return OpcodeResult::Error(error),
                };

                let left = unsafe { value_to_string(left) };
                let right = unsafe { value_to_string(right) };
                let result = RayaString::new(format!("{}{}", left, right));
                let gc_ptr = self.gc.lock().allocate(result);
                let value =
                    unsafe { Value::from_ptr(std::ptr::NonNull::new(gc_ptr.as_ptr()).unwrap()) };
                if let Err(error) = stack.push(value) {
                    return OpcodeResult::Error(error);
                }
                OpcodeResult::Continue
            }

            Opcode::Slen => {
                let value = match stack.pop() {
                    Ok(value) => value,
                    Err(error) => return OpcodeResult::Error(error),
                };
                let Some(string) = (unsafe { raya_string_ptr_checked(value) }) else {
                    return OpcodeResult::Error(VmError::TypeError("Expected string".to_string()));
                };
                let len = unsafe { &*string.as_ptr() }.len();
                if let Err(error) = stack.push(Value::i32(len as i32)) {
                    return OpcodeResult::Error(error);
                }
                OpcodeResult::Continue
            }

            Opcode::Seq | Opcode::Sne => {
                let right = match stack.pop() {
                    Ok(value) => value,
                    Err(error) => return OpcodeResult::Error(error),
                };
                let left = match stack.pop() {
                    Ok(value) => value,
                    Err(error) => return OpcodeResult::Error(error),
                };
                let equal = unsafe { compare_strings(left, right) }
                    .is_some_and(|ordering| ordering == Ordering::Equal);
                let result = if opcode == Opcode::Seq { equal } else { !equal };
                if let Err(error) = stack.push(Value::bool(result)) {
                    return OpcodeResult::Error(error);
                }
                OpcodeResult::Continue
            }

            Opcode::Slt | Opcode::Sle | Opcode::Sgt | Opcode::Sge => {
                let right = match stack.pop() {
                    Ok(value) => value,
                    Err(error) => return OpcodeResult::Error(error),
                };
                let left = match stack.pop() {
                    Ok(value) => value,
                    Err(error) => return OpcodeResult::Error(error),
                };
                let result = unsafe { compare_strings(left, right) }.is_some_and(|ordering| {
                    match opcode {
                        Opcode::Slt => ordering == Ordering::Less,
                        Opcode::Sle => ordering != Ordering::Greater,
                        Opcode::Sgt => ordering == Ordering::Greater,
                        Opcode::Sge => ordering != Ordering::Less,
                        _ => unreachable!(),
                    }
                });
                if let Err(error) = stack.push(Value::bool(result)) {
                    return OpcodeResult::Error(error);
                }
                OpcodeResult::Continue
            }

            Opcode::ToString => {
                let value = match stack.pop() {
                    Ok(value) => value,
                    Err(error) => return OpcodeResult::Error(error),
                };
                let result = RayaString::new(unsafe { value_to_string(value) });
                let gc_ptr = self.gc.lock().allocate(result);
                let value =
                    unsafe { Value::from_ptr(std::ptr::NonNull::new(gc_ptr.as_ptr()).unwrap()) };
                if let Err(error) = stack.push(value) {
                    return OpcodeResult::Error(error);
                }
                OpcodeResult::Continue
            }

            _ => OpcodeResult::Error(VmError::RuntimeError(format!(
                "Unexpected opcode in exec_string_ops: {:?}",
                opcode
            ))),
        }
    }
}
