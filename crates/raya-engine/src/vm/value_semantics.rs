//! Shared interpreter and JIT semantics for boxed values.

use crate::vm::gc::header_ptr_from_value_ptr;
use crate::vm::object::{Array, RayaString};
use crate::vm::value::Value;
use std::any::TypeId;
use std::cmp::Ordering;
use std::ptr::NonNull;

/// Return a string pointer only when `value` names a GC allocation whose
/// concrete type is `RayaString`.
///
/// # Safety
///
/// Pointer-valued `Value`s must satisfy the normal VM invariant: they point to
/// a live allocation managed by this VM's collector.
#[inline]
pub(crate) unsafe fn raya_string_ptr_checked(value: Value) -> Option<NonNull<RayaString>> {
    let ptr = value.as_ptr::<u8>()?;
    let header = &*header_ptr_from_value_ptr(ptr.as_ptr());
    (header.type_id() == TypeId::of::<RayaString>()).then(|| ptr.cast::<RayaString>())
}

/// Return an array pointer only when `value` names a GC allocation whose
/// concrete type is `Array`.
///
/// This is the single sanctioned way to obtain an `Array` from a `Value`: it
/// verifies the GC-header Rust `TypeId` so a `RayaString`, `Object`, or any
/// other heap allocation is never misread as an array. Callers must not
/// short-circuit with a bare `is_ptr()` + `as_ptr::<Array>()` cast.
///
/// # Safety
///
/// Pointer-valued `Value`s must satisfy the normal VM invariant: they point to
/// a live allocation managed by this VM's collector.
#[inline]
pub(crate) unsafe fn raya_array_ptr_checked(value: Value) -> Option<NonNull<Array>> {
    let ptr = value.as_ptr::<u8>()?;
    let header = &*header_ptr_from_value_ptr(ptr.as_ptr());
    (header.type_id() == TypeId::of::<Array>()).then(|| ptr.cast::<Array>())
}

/// True when `value` is a GC allocation whose concrete type is `Array`.
#[inline]
pub(crate) unsafe fn value_is_array(value: Value) -> bool {
    raya_array_ptr_checked(value).is_some()
}

/// Convert a value with the interpreter's existing `ToString` rules.
///
/// # Safety
///
/// Pointer-valued inputs must satisfy the VM pointer invariant documented by
/// `raya_string_ptr_checked`.
pub(crate) unsafe fn value_to_string(value: Value) -> String {
    if value.is_null() {
        "null".to_string()
    } else if let Some(boolean) = value.as_bool() {
        boolean.to_string()
    } else if let Some(integer) = value.as_i32() {
        integer.to_string()
    } else if let Some(float) = value.as_f64() {
        if float.fract() == 0.0 && float.abs() < 1e15 {
            (float as i64).to_string()
        } else {
            float.to_string()
        }
    } else if let Some(string) = raya_string_ptr_checked(value) {
        (&*string.as_ptr()).data.clone()
    } else if value.is_ptr() {
        "[object]".to_string()
    } else {
        "undefined".to_string()
    }
}

/// Compare two concrete string values.
///
/// Returns `None` when either operand is not a `RayaString` allocation.
///
/// # Safety
///
/// Pointer-valued inputs must satisfy the VM pointer invariant documented by
/// `raya_string_ptr_checked`.
pub(crate) unsafe fn compare_strings(left: Value, right: Value) -> Option<Ordering> {
    let left = raya_string_ptr_checked(left)?;
    let right = raya_string_ptr_checked(right)?;
    Some((&*left.as_ptr()).data.cmp(&(&*right.as_ptr()).data))
}

/// Generic equality as implemented by the interpreter today.
///
/// # Safety
///
/// Pointer-valued inputs must satisfy the VM pointer invariant documented by
/// `raya_string_ptr_checked`.
pub(crate) unsafe fn values_equal(left: Value, right: Value) -> bool {
    if left.is_f64() || right.is_f64() {
        let left = left
            .as_f64()
            .unwrap_or(left.as_i32().map(|value| value as f64).unwrap_or(0.0));
        let right = right
            .as_f64()
            .unwrap_or(right.as_i32().map(|value| value as f64).unwrap_or(0.0));
        left == right
    } else if let Some(ordering) = compare_strings(left, right) {
        ordering == Ordering::Equal
    } else {
        left == right
    }
}

#[cfg(test)]
mod tests {
    use super::{compare_strings, value_to_string, values_equal};
    use crate::vm::interpreter::{SafepointCoordinator, SharedVmState};
    use crate::vm::object::{Object, RayaString};
    use crate::vm::scheduler::Task;
    use crate::vm::value::Value;
    use crossbeam_deque::Injector;
    use rustc_hash::FxHashMap;
    use std::cmp::Ordering;
    use std::ptr::NonNull;
    use std::sync::Arc;

    #[test]
    fn generic_numeric_equality_preserves_ieee_behavior() {
        unsafe {
            assert!(values_equal(Value::i32(7), Value::f64(7.0)));
            assert!(values_equal(Value::f64(0.0), Value::f64(-0.0)));
            assert!(!values_equal(Value::f64(f64::NAN), Value::f64(f64::NAN)));
        }
    }

    #[test]
    fn generic_non_numeric_equality_uses_value_identity() {
        unsafe {
            assert!(values_equal(Value::null(), Value::null()));
            assert!(values_equal(Value::bool(true), Value::bool(true)));
            assert!(!values_equal(Value::bool(true), Value::bool(false)));
        }
    }

    #[test]
    fn checked_string_semantics_distinguish_strings_from_other_allocations() {
        let safepoint = Arc::new(SafepointCoordinator::new(1));
        let tasks = Arc::new(parking_lot::RwLock::new(FxHashMap::<
            crate::vm::scheduler::TaskId,
            Arc<Task>,
        >::default()));
        let shared = SharedVmState::new(safepoint, tasks, Arc::new(Injector::new()));
        let (first, second, object) = {
            let mut gc = shared.gc.lock();
            let first = gc.allocate(RayaString::new("same".to_string()));
            let second = gc.allocate(RayaString::new("same".to_string()));
            let object = gc.allocate(Object::new_structural(7, 0));
            unsafe {
                (
                    Value::from_ptr(NonNull::new(first.as_ptr()).unwrap()),
                    Value::from_ptr(NonNull::new(second.as_ptr()).unwrap()),
                    Value::from_ptr(NonNull::new(object.as_ptr()).unwrap()),
                )
            }
        };

        unsafe {
            assert_eq!(compare_strings(first, second), Some(Ordering::Equal));
            assert!(values_equal(first, second));
            assert_eq!(compare_strings(first, object), None);
            assert!(!values_equal(first, object));
            assert_eq!(value_to_string(object), "[object]");
        }
    }

    #[test]
    fn checked_array_identity_rejects_strings_and_objects() {
        use super::{raya_array_ptr_checked, value_is_array};
        use crate::vm::object::Array;

        let safepoint = Arc::new(SafepointCoordinator::new(1));
        let tasks = Arc::new(parking_lot::RwLock::new(FxHashMap::<
            crate::vm::scheduler::TaskId,
            Arc<Task>,
        >::default()));
        let shared = SharedVmState::new(safepoint, tasks, Arc::new(Injector::new()));
        let (array_val, string_val, object_val) = {
            let mut gc = shared.gc.lock();
            let array = gc.allocate(Array::new(0, 2));
            let string = gc.allocate(RayaString::new("s".to_string()));
            let object = gc.allocate(Object::new_structural(7, 0));
            unsafe {
                (
                    Value::from_ptr(NonNull::new(array.as_ptr()).unwrap()),
                    Value::from_ptr(NonNull::new(string.as_ptr()).unwrap()),
                    Value::from_ptr(NonNull::new(object.as_ptr()).unwrap()),
                )
            }
        };

        unsafe {
            // Only the real array resolves; a string, an object, and a
            // non-pointer are all rejected.
            assert!(value_is_array(array_val));
            assert!(raya_array_ptr_checked(array_val).is_some());
            assert!(!value_is_array(string_val));
            assert!(raya_array_ptr_checked(string_val).is_none());
            assert!(!value_is_array(object_val));
            assert!(raya_array_ptr_checked(object_val).is_none());
            assert!(!value_is_array(Value::i32(3)));
        }
    }

    #[test]
    fn string_typed_array_rejects_object_pointer() {
        use crate::compiler::bytecode::RuntimeTypeDescriptor as D;
        use crate::vm::object::{Array, ArrayStoreError};

        let safepoint = Arc::new(SafepointCoordinator::new(1));
        let tasks = Arc::new(parking_lot::RwLock::new(FxHashMap::<
            crate::vm::scheduler::TaskId,
            Arc<Task>,
        >::default()));
        let shared = SharedVmState::new(safepoint, tasks, Arc::new(Injector::new()));
        let (string_val, object_val) = {
            let mut gc = shared.gc.lock();
            let string = gc.allocate(RayaString::new("hello".to_string()));
            let object = gc.allocate(Object::new_structural(7, 0));
            unsafe {
                (
                    Value::from_ptr(NonNull::new(string.as_ptr()).unwrap()),
                    Value::from_ptr(NonNull::new(object.as_ptr()).unwrap()),
                )
            }
        };

        let mut arr = Array::with_element_type(0, Some(D::String), 1);
        // A real RayaString is accepted; an Object pointer (also is_ptr) is
        // rejected by the GC-header identity check, not accepted as before.
        assert!(arr.element_matches(string_val));
        assert!(!arr.element_matches(object_val));
        assert!(arr.checked_set(0, string_val).is_ok());
        assert!(matches!(
            arr.checked_set(0, object_val),
            Err(ArrayStoreError::ElementType { .. })
        ));
        // A non-pointer is never a string.
        assert!(!arr.element_matches(Value::i32(1)));
    }
}
