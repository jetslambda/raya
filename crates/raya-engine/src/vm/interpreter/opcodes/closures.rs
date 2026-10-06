//! Closure opcode handlers: MakeClosure, LoadCaptured, StoreCaptured, SetClosureCapture, NewRefCell, LoadRefCell, StoreRefCell

use crate::compiler::Opcode;
use crate::vm::interpreter::execution::OpcodeResult;
use crate::vm::interpreter::Interpreter;
use crate::vm::object::Closure;
use crate::vm::scheduler::Task;
use crate::vm::stack::Stack;
use crate::vm::value::Value;
use crate::vm::VmError;
use std::sync::Arc;


/// ALY-54: does `value` point at a heap object whose GC header says `expected`?
///
/// ONE implementation, used by BOTH the interpreter handlers and the JIT helpers.
/// The D4.4 note recorded that both sides deliberately reproduced a weak `is_ptr()`
/// check so they would not diverge — which works right up until the day one side is
/// changed and the other is not. Sharing the function removes that possibility rather
/// than managing it.
///
/// `is_ptr()` alone asks only "is this a heap pointer?"; it does not ask "is this the
/// RIGHT KIND?". Reinterpreting the answer to one as the answer to the other is type
/// confusion whose only guard is that two layouts happen to line up.
pub(crate) fn typed_ptr_matches(value: Value, expected: std::any::TypeId) -> bool {
    if !value.is_ptr() {
        return false;
    }
    let header = unsafe {
        &*crate::vm::gc::header_ptr_from_value_ptr(value.as_ptr::<u8>().unwrap().as_ptr())
    };
    header.type_id() == expected
}

impl<'a> Interpreter<'a> {
    pub(in crate::vm::interpreter) fn exec_closure_ops(
        &mut self,
        stack: &mut Stack,
        ip: &mut usize,
        code: &[u8],
        task: &Arc<Task>,
        opcode: Opcode,
    ) -> OpcodeResult {
        match opcode {
            Opcode::MakeClosure => {
                self.safepoint.poll();
                let func_index = match Self::read_u32(code, ip) {
                    Ok(v) => v as usize,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let capture_count = match Self::read_u16(code, ip) {
                    Ok(v) => v as usize,
                    Err(e) => return OpcodeResult::Error(e),
                };

                let mut captures = Vec::with_capacity(capture_count);
                for _ in 0..capture_count {
                    match stack.pop() {
                        Ok(v) => captures.push(v),
                        Err(e) => return OpcodeResult::Error(e),
                    }
                }
                captures.reverse();

                let closure = Closure::with_module(func_index, captures, task.current_module());
                let gc_ptr = self.gc.lock().allocate(closure);
                let value =
                    unsafe { Value::from_ptr(std::ptr::NonNull::new(gc_ptr.as_ptr()).unwrap()) };
                if let Err(e) = stack.push(value) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::LoadCaptured => {
                let capture_index = match Self::read_u16(code, ip) {
                    Ok(v) => v as usize,
                    Err(e) => return OpcodeResult::Error(e),
                };

                let closure_val = match task.current_closure() {
                    Some(v) => v,
                    None => {
                        return OpcodeResult::Error(VmError::RuntimeError(
                            "LoadCaptured without active closure".to_string(),
                        ));
                    }
                };

                let closure_ptr = unsafe { closure_val.as_ptr::<Closure>() };
                let closure = unsafe { &*closure_ptr.unwrap().as_ptr() };
                let value = match closure.get_captured(capture_index) {
                    Some(v) => v,
                    None => {
                        return OpcodeResult::Error(VmError::RuntimeError(format!(
                            "Capture index {} out of bounds",
                            capture_index
                        )));
                    }
                };
                if std::env::var("RAYA_DEBUG_FIELD_TRACE").is_ok() {
                    eprintln!(
                        "[field-trace] LoadCaptured[{}] => {:?} (is_ptr={})",
                        capture_index,
                        value,
                        value.is_ptr()
                    );
                }
                if let Err(e) = stack.push(value) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::StoreCaptured => {
                let capture_index = match Self::read_u16(code, ip) {
                    Ok(v) => v as usize,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let value = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                let closure_val = match task.current_closure() {
                    Some(v) => v,
                    None => {
                        return OpcodeResult::Error(VmError::RuntimeError(
                            "StoreCaptured without active closure".to_string(),
                        ));
                    }
                };

                let closure_ptr = unsafe { closure_val.as_ptr::<Closure>() };
                let closure = unsafe { &mut *closure_ptr.unwrap().as_ptr() };
                if let Err(e) = closure.set_captured(capture_index, value) {
                    return OpcodeResult::Error(VmError::RuntimeError(e));
                }
                OpcodeResult::Continue
            }

            Opcode::SetClosureCapture => {
                let capture_index = match Self::read_u16(code, ip) {
                    Ok(v) => v as usize,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let value = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let closure_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                // ALY-54: check the GC-header TypeId, not just `is_ptr()`.
                let closure_val = match crate::vm::interpreter::Interpreter::ensure_typed_ptr(
                    closure_val,
                    std::any::TypeId::of::<Closure>(),
                    "Closure",
                    "SetClosureCapture",
                ) {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                let closure_ptr = unsafe { closure_val.as_ptr::<Closure>() };
                let closure = unsafe { &mut *closure_ptr.unwrap().as_ptr() };
                if let Err(e) = closure.set_captured(capture_index, value) {
                    return OpcodeResult::Error(VmError::RuntimeError(e));
                }
                if let Err(e) = stack.push(closure_val) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::NewRefCell => {
                use crate::vm::object::RefCell;
                let initial_value = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let refcell = RefCell::new(initial_value);
                let gc_ptr = self.gc.lock().allocate(refcell);
                let value =
                    unsafe { Value::from_ptr(std::ptr::NonNull::new(gc_ptr.as_ptr()).unwrap()) };
                if let Err(e) = stack.push(value) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::LoadRefCell => {
                use crate::vm::object::RefCell;
                let refcell_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                // ALY-54: check the GC-header TypeId, not just `is_ptr()`.
                let refcell_val = match crate::vm::interpreter::Interpreter::ensure_typed_ptr(
                    refcell_val,
                    std::any::TypeId::of::<RefCell>(),
                    "RefCell",
                    "RefCell access",
                ) {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                let refcell_ptr = unsafe { refcell_val.as_ptr::<RefCell>() };
                let refcell = unsafe { &*refcell_ptr.unwrap().as_ptr() };
                if let Err(e) = stack.push(refcell.get()) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::StoreRefCell => {
                use crate::vm::object::RefCell;
                let value = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let refcell_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                // ALY-54: check the GC-header TypeId, not just `is_ptr()`.
                let refcell_val = match crate::vm::interpreter::Interpreter::ensure_typed_ptr(
                    refcell_val,
                    std::any::TypeId::of::<RefCell>(),
                    "RefCell",
                    "RefCell access",
                ) {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };

                let refcell_ptr = unsafe { refcell_val.as_ptr::<RefCell>() };
                let refcell = unsafe { &mut *refcell_ptr.unwrap().as_ptr() };
                refcell.set(value);
                OpcodeResult::Continue
            }

            _ => OpcodeResult::Error(VmError::RuntimeError(format!(
                "Unexpected opcode in exec_closure_ops: {:?}",
                opcode
            ))),
        }
    }
}
