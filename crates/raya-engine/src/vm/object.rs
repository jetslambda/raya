//! Object model and class system

use crate::vm::value::Value;
use rustc_hash::FxHashMap;
use std::any::TypeId;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, RwLock};

/// Physical object-layout identity used for slot dispatch and structural adapters.
pub type LayoutId = u32;
/// Nominal runtime type identity used for class semantics.
pub type NominalTypeId = u32;
/// Structural compatibility identity.
pub type ShapeId = u64;
/// Interned dynamic property key identity.
pub type PropKeyId = u32;
/// Runtime handle identity for imported/exported type constructors.
pub type TypeHandleId = u32;
/// High-bit tag reserved for deterministic structural layout IDs.
pub const STRUCTURAL_LAYOUT_ID_TAG: LayoutId = 0x8000_0000;
/// Object flag: this object has an active dynamic property lane.
pub const OBJECT_FLAG_HAS_DYN_MAP: u32 = 1 << 0;
/// Object flag: object is frozen against mutation.
pub const OBJECT_FLAG_FROZEN: u32 = 1 << 1;
/// Object flag: object is sealed against extension.
pub const OBJECT_FLAG_SEALED: u32 = 1 << 2;

/// Runtime handle for nominal type constructors crossing module boundaries.
///
/// This is the canonical runtime representation used for imported/exported class
/// constructors in binary linking mode. It avoids exposing raw class IDs through
/// module hydration globals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeHandle {
    /// Runtime-owned type handle ID in the VM registry.
    pub handle_id: TypeHandleId,
    /// Optional structural shape hash for diagnostics/contract checks.
    pub shape_id: Option<ShapeId>,
}

/// Global counter for generating unique object IDs
static NEXT_OBJECT_ID: AtomicU64 = AtomicU64::new(1);
/// Process-local fallback registry for structural layout names used outside a live VM.
static GLOBAL_LAYOUT_NAMES: LazyLock<RwLock<FxHashMap<LayoutId, Vec<String>>>> =
    LazyLock::new(|| RwLock::new(FxHashMap::default()));
pub const OBJECT_DESCRIPTOR_LAYOUT_FIELDS: &[&str] =
    &["value", "writable", "configurable", "enumerable", "get", "set"];
pub const BUFFER_LAYOUT_FIELDS: &[&str] = &["bufferPtr", "length"];

/// Generate a new unique object ID
fn generate_object_id() -> u64 {
    NEXT_OBJECT_ID.fetch_add(1, Ordering::Relaxed)
}

/// Derive a deterministic physical layout ID from ordered member names.
pub fn layout_id_from_ordered_names(names: &[String]) -> LayoutId {
    let payload = format!("layout{{{}}}", names.join(","));
    let raw = crate::parser::types::signature_hash(&payload) as LayoutId;
    (raw | STRUCTURAL_LAYOUT_ID_TAG).max(STRUCTURAL_LAYOUT_ID_TAG | 1)
}

/// Derive a deterministic structural shape ID from member names.
pub fn shape_id_from_member_names(names: &[String]) -> ShapeId {
    let mut canonical = names.to_vec();
    canonical.sort_unstable();
    canonical.dedup();
    let payload = format!("shape{{{}}}", canonical.join(","));
    crate::parser::types::signature_hash(&payload)
}

/// Register structural layout names in the process-local fallback registry.
pub fn register_global_layout_names(layout_id: LayoutId, names: &[String]) {
    if layout_id == 0 {
        return;
    }
    let mut registry = GLOBAL_LAYOUT_NAMES
        .write()
        .expect("global layout registry poisoned");
    registry.entry(layout_id).or_insert_with(|| names.to_vec());
}

/// Resolve structural layout names from the process-local fallback registry.
pub fn global_layout_names(layout_id: LayoutId) -> Option<Vec<String>> {
    GLOBAL_LAYOUT_NAMES
        .read()
        .expect("global layout registry poisoned")
        .get(&layout_id)
        .cloned()
}

/// Immutable identity header for runtime objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectHeader {
    /// Unique object identity used for hashing/structural adapter caches.
    pub object_id: u64,
    /// Physical layout identity used for fixed-slot dispatch.
    pub layout_id: LayoutId,
    /// Optional nominal runtime type identity used for class semantics.
    pub nominal_type_id: Option<NominalTypeId>,
    /// Runtime object flags.
    pub flags: u32,
}

impl ObjectHeader {
    #[inline]
    pub fn nominal(layout_id: LayoutId, nominal_type_id: NominalTypeId) -> Self {
        assert_ne!(layout_id, 0, "runtime object layout id must be nonzero");
        Self {
            object_id: generate_object_id(),
            layout_id,
            nominal_type_id: Some(nominal_type_id),
            flags: 0,
        }
    }

    #[inline]
    pub fn structural(layout_id: LayoutId) -> Self {
        assert_ne!(layout_id, 0, "runtime object layout id must be nonzero");
        Self {
            object_id: generate_object_id(),
            layout_id,
            nominal_type_id: None,
            flags: 0,
        }
    }
}

/// Object instance (heap-allocated)
#[derive(Debug, Clone)]
pub struct Object {
    /// Runtime identity header.
    pub header: ObjectHeader,
    /// Field values
    pub fields: Vec<Value>,
    /// Dynamic property lane for JS-style keyed properties.
    pub dyn_map: Option<FxHashMap<PropKeyId, Value>>,
}

impl Object {
    /// Create a nominal object with explicit layout and nominal type IDs.
    pub fn new_nominal(
        layout_id: LayoutId,
        nominal_type_id: NominalTypeId,
        field_count: usize,
    ) -> Self {
        Self {
            header: ObjectHeader::nominal(layout_id, nominal_type_id),
            fields: vec![Value::null(); field_count],
            dyn_map: None,
        }
    }

    /// Create a structural/dynamic object with explicit layout ID and no nominal identity.
    pub fn new_structural(layout_id: LayoutId, field_count: usize) -> Self {
        Self {
            header: ObjectHeader::structural(layout_id),
            fields: vec![Value::null(); field_count],
            dyn_map: None,
        }
    }

    /// Create a structural object with the dynamic property lane enabled.
    pub fn new_dynamic(layout_id: LayoutId, field_count: usize) -> Self {
        let mut object = Self::new_structural(layout_id, field_count);
        object.ensure_dyn_map();
        object
    }

    fn synthetic_nominal_layout_id(nominal_type_id: NominalTypeId, field_count: usize) -> LayoutId {
        let payload = format!("test_nominal_layout:{nominal_type_id}:{field_count}");
        let raw = crate::parser::types::signature_hash(&payload) as LayoutId;
        (raw & !STRUCTURAL_LAYOUT_ID_TAG).max(1)
    }

    fn synthetic_structural_layout_id(field_count: usize) -> LayoutId {
        let names = (0..field_count)
            .map(|index| format!("__slot{index}"))
            .collect::<Vec<_>>();
        layout_id_from_ordered_names(&names)
    }

    /// Create a synthetic nominal object with a deterministic nonzero layout.
    ///
    /// This is intended for internal tests and helper code that need a nominal
    /// object without first registering a full runtime class definition.
    pub fn new_synthetic_nominal(nominal_type_id: usize, field_count: usize) -> Self {
        let nominal_type_id = nominal_type_id as NominalTypeId;
        let layout_id = Self::synthetic_nominal_layout_id(nominal_type_id, field_count);
        Self::new_nominal(layout_id, nominal_type_id, field_count)
    }

    /// Create a synthetic structural object with a deterministic nonzero layout.
    ///
    /// This is intended for internal tests and helper code that need a
    /// structural object without going through compiler-produced object literals.
    pub fn new_synthetic_structural(field_count: usize) -> Self {
        let layout_id = Self::synthetic_structural_layout_id(field_count);
        Self::new_structural(layout_id, field_count)
    }

    #[inline]
    pub fn object_id(&self) -> u64 {
        self.header.object_id
    }

    #[inline]
    pub fn layout_id(&self) -> LayoutId {
        self.header.layout_id
    }

    #[inline]
    pub fn set_layout_id(&mut self, layout_id: LayoutId) {
        self.header.layout_id = layout_id;
    }

    #[inline]
    pub fn nominal_type_id(&self) -> Option<NominalTypeId> {
        self.header.nominal_type_id
    }

    #[inline]
    pub fn set_nominal_type_id(&mut self, nominal_type_id: Option<NominalTypeId>) {
        self.header.nominal_type_id = nominal_type_id;
    }

    #[inline]
    pub fn flags(&self) -> u32 {
        self.header.flags
    }

    #[inline]
    pub fn has_flag(&self, flag: u32) -> bool {
        self.header.flags & flag != 0
    }

    #[inline]
    pub fn set_flag(&mut self, flag: u32) {
        self.header.flags |= flag;
    }

    #[inline]
    pub fn clear_flag(&mut self, flag: u32) {
        self.header.flags &= !flag;
    }

    #[inline]
    pub fn dyn_map(&self) -> Option<&FxHashMap<PropKeyId, Value>> {
        self.dyn_map.as_ref()
    }

    #[inline]
    pub fn dyn_map_mut(&mut self) -> Option<&mut FxHashMap<PropKeyId, Value>> {
        self.dyn_map.as_mut()
    }

    pub fn ensure_dyn_map(&mut self) -> &mut FxHashMap<PropKeyId, Value> {
        self.set_flag(OBJECT_FLAG_HAS_DYN_MAP);
        self.dyn_map.get_or_insert_with(FxHashMap::default)
    }

    /// Get a field value by index
    pub fn get_field(&self, index: usize) -> Option<Value> {
        self.fields.get(index).copied()
    }

    /// Set a field value by index.
    ///
    /// A thin wrapper over [`Object::checked_set_field`] that renders the error as
    /// a `String`. It exists so the many existing call sites keep their current
    /// behaviour and message; new callers, and anything that needs to tell a
    /// bounds failure apart from a binding failure, should use the checked form.
    pub fn set_field(&mut self, index: usize, value: Value) -> Result<(), String> {
        self.checked_set_field(index, value)
            .map_err(|error| error.to_string())
    }

    /// The single checked path for writing an object field.
    ///
    /// Mirrors the `checked_set` / `checked_push` / `checked_splice` family that
    /// D4.2 introduced for arrays: opcode handlers and built-in mutators go
    /// through here so an out-of-range offset is a typed, inspectable value rather
    /// than a formatted string that has to be matched on. Binding failures —
    /// a non-object receiver, a missing field, a descriptor setter — are decided
    /// one level up in the opcode handlers, so a caller of this method can tell
    /// "the offset was wrong" from "this field is not storable" without parsing.
    pub fn checked_set_field(
        &mut self,
        index: usize,
        value: Value,
    ) -> Result<(), ObjectFieldStoreError> {
        if index < self.fields.len() {
            self.fields[index] = value;
            Ok(())
        } else {
            Err(ObjectFieldStoreError::OutOfBounds {
                index,
                field_count: self.fields.len(),
            })
        }
    }

    /// Get number of fields
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Return the nominal type ID when this object participates in nominal dispatch.
    #[inline]
    pub fn nominal_type_id_usize(&self) -> Option<usize> {
        self.nominal_type_id().map(|id| id as usize)
    }

    /// Return a stable runtime identity for diagnostics/debug paths.
    #[inline]
    pub fn runtime_identity_id(&self) -> usize {
        self.nominal_type_id_usize()
            .unwrap_or_else(|| self.layout_id() as usize)
    }

    /// Return true when this object is a structural/dynamic carrier (no nominal identity).
    #[inline]
    pub fn is_structural(&self) -> bool {
        self.nominal_type_id().is_none()
    }
}

/// Class definition metadata
#[derive(Debug, Clone)]
pub struct Class {
    /// Class ID (unique identifier)
    pub id: usize,
    /// Class name
    pub name: String,
    /// Number of instance fields (including inherited)
    pub field_count: usize,
    /// Parent class ID (None for root classes)
    pub parent_id: Option<usize>,
    /// Virtual method table
    pub vtable: VTable,
    /// Static fields (class-level, shared across all instances)
    pub static_fields: Vec<Value>,
    /// Constructor function ID (None if no explicit constructor)
    pub constructor_id: Option<usize>,
    /// Optional defining module for method/constructor function IDs.
    pub module: Option<Arc<crate::compiler::Module>>,
}

impl Class {
    /// Create a new class
    pub fn new(id: usize, name: String, field_count: usize) -> Self {
        Self {
            id,
            name,
            field_count,
            parent_id: None,
            vtable: VTable::new(),
            static_fields: Vec::new(),
            constructor_id: None,
            module: None,
        }
    }

    /// Create a new class with parent
    pub fn with_parent(id: usize, name: String, field_count: usize, parent_id: usize) -> Self {
        Self {
            id,
            name,
            field_count,
            parent_id: Some(parent_id),
            vtable: VTable::new(),
            static_fields: Vec::new(),
            constructor_id: None,
            module: None,
        }
    }

    /// Create a new class with static fields
    pub fn with_static_fields(
        id: usize,
        name: String,
        field_count: usize,
        static_field_count: usize,
    ) -> Self {
        Self {
            id,
            name,
            field_count,
            parent_id: None,
            vtable: VTable::new(),
            static_fields: vec![Value::null(); static_field_count],
            constructor_id: None,
            module: None,
        }
    }

    /// Set the constructor function ID
    pub fn set_constructor(&mut self, function_id: usize) {
        self.constructor_id = Some(function_id);
    }

    /// Get the constructor function ID
    pub fn get_constructor(&self) -> Option<usize> {
        self.constructor_id
    }

    /// Get a static field value by index
    pub fn get_static_field(&self, index: usize) -> Option<Value> {
        self.static_fields.get(index).copied()
    }

    /// Set a static field value by index
    pub fn set_static_field(&mut self, index: usize, value: Value) -> Result<(), String> {
        if index < self.static_fields.len() {
            self.static_fields[index] = value;
            Ok(())
        } else {
            Err(format!(
                "Static field index {} out of bounds (class has {} static fields)",
                index,
                self.static_fields.len()
            ))
        }
    }

    /// Get number of static fields
    pub fn static_field_count(&self) -> usize {
        self.static_fields.len()
    }

    /// Add a method to the vtable
    pub fn add_method(&mut self, function_id: usize) {
        self.vtable.add_method(function_id);
    }

    /// Get method from vtable
    pub fn get_method(&self, method_index: usize) -> Option<usize> {
        self.vtable.get_method(method_index)
    }
}

/// Virtual method table for dynamic dispatch
#[derive(Debug, Clone)]
pub struct VTable {
    /// Method function IDs (indexed by method slot)
    pub methods: Vec<usize>,
}

impl VTable {
    /// Create a new empty vtable
    pub fn new() -> Self {
        Self {
            methods: Vec::new(),
        }
    }

    /// Add a method to the vtable (appends to end)
    pub fn add_method(&mut self, function_id: usize) {
        self.methods.push(function_id);
    }

    /// Get method function ID by index
    pub fn get_method(&self, index: usize) -> Option<usize> {
        self.methods.get(index).copied()
    }

    /// Get number of methods
    pub fn method_count(&self) -> usize {
        self.methods.len()
    }

    /// Override a method at specific index
    pub fn override_method(&mut self, index: usize, function_id: usize) -> Result<(), String> {
        if index < self.methods.len() {
            self.methods[index] = function_id;
            Ok(())
        } else {
            Err(format!("Method index {} out of bounds", index))
        }
    }
}

impl Default for VTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Why an object field store was rejected.
///
/// Deliberately parallel to [`ArrayStoreError`], which D4.2 introduced for array
/// stores, so the two families read the same way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjectFieldStoreError {
    /// The offset was outside the object's field range.
    OutOfBounds {
        /// Requested field offset.
        index: usize,
        /// Field count at the time of the attempted store.
        field_count: usize,
    },
}

impl std::fmt::Display for ObjectFieldStoreError {
    /// Renders byte-for-byte the message `Object::set_field` produced before this
    /// type existed, so the delegating wrapper is a behaviour-preserving refactor.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ObjectFieldStoreError::OutOfBounds { index, field_count } => write!(
                f,
                "Field index {index} out of bounds (object has {field_count} fields)"
            ),
        }
    }
}

impl std::error::Error for ObjectFieldStoreError {}

/// Failure modes of a checked array store.
///
/// Distinguishing these lets callers map bounds violations to a runtime error
/// and element-type violations to a `TypeError`, without re-deriving intent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArrayStoreError {
    /// The index was outside the current element range.
    OutOfBounds {
        /// Requested array index.
        index: usize,
        /// Array length at the time of the attempted store.
        len: usize,
    },
    /// The value did not satisfy the array's resolved element constraint.
    ElementType {
        /// Resolved element descriptor that rejected the value, if one exists.
        expected: Option<crate::compiler::bytecode::RuntimeTypeDescriptor>,
    },
}

impl std::fmt::Display for ArrayStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArrayStoreError::OutOfBounds { index, len } => {
                write!(f, "Array index {index} out of bounds (length: {len})")
            }
            ArrayStoreError::ElementType { expected } => match expected {
                Some(descriptor) => write!(
                    f,
                    "Cannot store value: array element type is {}",
                    descriptor.atom()
                ),
                None => write!(f, "Cannot store value: array element type mismatch"),
            },
        }
    }
}

/// Array object (heap-allocated)
///
/// `type_id` is a legacy, module-local element hint retained for backward
/// compatibility with the many builtin call sites that construct dynamic
/// arrays. It is **not** a global runtime descriptor and must not be treated
/// as one (descriptor ids are only meaningful relative to the producing
/// module's `runtime_types` table).
///
/// `element_type` is the resolved, self-contained element constraint. It is
/// `None` for dynamic/legacy arrays (equivalent to `AnyValue`) and `Some(desc)`
/// for arrays whose element type was resolved at their construction site. The
/// resolved descriptor is safe to read across module boundaries because it
/// carries only runtime-global identity (primitives, or nominal/layout ids that
/// are already VM-global). It holds no GC references, so array tracing and
/// snapshotting are unaffected by its presence.
#[derive(Debug, Clone)]
pub struct Array {
    /// Legacy module-local element type hint (see struct docs). Do not treat as
    /// a global runtime descriptor.
    pub type_id: usize,
    /// Resolved element constraint, self-contained across module boundaries.
    /// `None` means dynamic (`AnyValue`) semantics.
    pub element_type: Option<crate::compiler::bytecode::RuntimeTypeDescriptor>,
    /// Array elements
    pub elements: Vec<Value>,
}

/// True when `value` is a GC allocation whose concrete type is `RayaString`,
/// verified via the GC-header Rust `TypeId`. A bare `is_ptr()` is insufficient
/// because objects and arrays are also pointers.
#[inline]
fn value_is_raya_string(value: Value) -> bool {
    if !value.is_ptr() {
        return false;
    }
    let Some(raw_ptr) = (unsafe { value.as_ptr::<u8>() }) else {
        return false;
    };
    let header = unsafe { &*crate::vm::gc::header_ptr_from_value_ptr(raw_ptr.as_ptr()) };
    header.type_id() == TypeId::of::<RayaString>()
}

impl Array {
    /// Create a new dynamic array with given length and legacy element hint.
    ///
    /// The resolved `element_type` constraint is left unset (`None` = dynamic);
    /// callers with a resolved descriptor should use
    /// [`Array::with_element_type`].
    pub fn new(type_id: usize, length: usize) -> Self {
        Self {
            type_id,
            element_type: None,
            elements: vec![Value::null(); length],
        }
    }

    /// Create a new array carrying a resolved element-type constraint.
    pub fn with_element_type(
        type_id: usize,
        element_type: Option<crate::compiler::bytecode::RuntimeTypeDescriptor>,
        length: usize,
    ) -> Self {
        Self {
            type_id,
            element_type,
            elements: vec![Value::null(); length],
        }
    }

    /// Resolved element constraint, if any. `None` means dynamic (`AnyValue`).
    #[inline]
    pub fn element_type(&self) -> Option<&crate::compiler::bytecode::RuntimeTypeDescriptor> {
        self.element_type.as_ref()
    }

    /// Get array length
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Check if array is empty
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Get element at index
    pub fn get(&self, index: usize) -> Option<Value> {
        self.elements.get(index).copied()
    }

    /// Set element at index (unchecked element type; bounds-checked only).
    ///
    /// Prefer [`Array::checked_set`] on paths that must honor a resolved
    /// element constraint. This raw setter is retained for internal builtin
    /// paths that construct arrays element-by-element with values they already
    /// produced.
    pub fn set(&mut self, index: usize, value: Value) -> Result<(), String> {
        if index < self.elements.len() {
            self.elements[index] = value;
            Ok(())
        } else {
            Err(format!(
                "Array index {} out of bounds (length: {})",
                index,
                self.elements.len()
            ))
        }
    }

    /// Push element to end of array, returns new length (unchecked element
    /// type). Prefer [`Array::checked_push`] where a resolved constraint must
    /// be honored.
    pub fn push(&mut self, value: Value) -> usize {
        self.elements.push(value);
        self.elements.len()
    }

    /// Whether `value` satisfies this array's resolved element constraint.
    ///
    /// Dynamic arrays (`element_type == None`) and the boxed/opaque descriptors
    /// (`AnyValue`, `Ref`, and all complex descriptors) accept any value. Only
    /// the concrete, unambiguously checkable primitive descriptors are enforced
    /// here. Reads are never affected — this governs stores only. A null value
    /// is always accepted so that newly allocated, null-filled slots and
    /// explicit nulls never spuriously fail a typed store.
    #[inline]
    pub fn element_matches(&self, value: Value) -> bool {
        use crate::compiler::bytecode::RuntimeTypeDescriptor as D;
        let Some(descriptor) = self.element_type.as_ref() else {
            return true;
        };
        if value.is_null() {
            return true;
        }
        match descriptor {
            D::I32 => value.is_i32(),
            D::F64 => value.is_f64() || value.is_i32(),
            D::Bool => value.is_bool(),
            // A `String` slot must hold an actual `RayaString`. Verify the GC
            // header's Rust `TypeId` so an `Object`/`Array` pointer (which is
            // also `is_ptr()`) is rejected rather than silently accepted.
            D::String => value_is_raya_string(value),
            D::Null => value.is_null(),
            // Boxed, opaque, unit, reference, and every complex descriptor are
            // treated as accept-all: soundly enforcing them would require the
            // module-local nominal/layout/shape context that is not available
            // here, and a partial check risks the exact class of unsound
            // mistreatment (Void/AnyValue/Ref, and false rejections of valid
            // complex values) that a prior attempt introduced. This is a
            // deliberately conservative policy: it never wrongly rejects, and
            // no JIT-generated code relies on complex-element enforcement
            // because array capability opcodes remain rejected.
            D::Void
            | D::AnyValue
            | D::Ref
            | D::Object { .. }
            | D::Array { .. }
            | D::Tuple { .. }
            | D::Function { .. }
            | D::Task { .. } => true,
        }
    }

    /// Store `value` at `index`, honoring the resolved element constraint.
    ///
    /// This is the centralized checked mutation path for indexed stores. Bounds
    /// are enforced, and — when a resolved constraint is present — the value is
    /// validated via [`Array::element_matches`]. Reads never route through here,
    /// so null-filled slots are never type-rejected.
    pub fn checked_set(&mut self, index: usize, value: Value) -> Result<(), ArrayStoreError> {
        if index >= self.elements.len() {
            return Err(ArrayStoreError::OutOfBounds {
                index,
                len: self.elements.len(),
            });
        }
        if !self.element_matches(value) {
            return Err(ArrayStoreError::ElementType {
                expected: self.element_type.clone(),
            });
        }
        self.elements[index] = value;
        Ok(())
    }

    /// Append `value`, honoring the resolved element constraint. Returns the new
    /// length. Centralized checked mutation path for pushes.
    pub fn checked_push(&mut self, value: Value) -> Result<usize, ArrayStoreError> {
        if !self.element_matches(value) {
            return Err(ArrayStoreError::ElementType {
                expected: self.element_type.clone(),
            });
        }
        self.elements.push(value);
        Ok(self.elements.len())
    }

    /// Prepend `value`, honoring the resolved element constraint. Returns the
    /// new length. Centralized checked mutation path for `unshift`.
    pub fn checked_unshift(&mut self, value: Value) -> Result<usize, ArrayStoreError> {
        if !self.element_matches(value) {
            return Err(ArrayStoreError::ElementType {
                expected: self.element_type.clone(),
            });
        }
        self.elements.insert(0, value);
        Ok(self.elements.len())
    }

    /// Fill `[start, end)` with `value`, honoring the resolved element
    /// constraint. The range is clamped to the current length. Rejects the
    /// whole operation before mutating if `value` violates the constraint, so a
    /// partially-filled array is never observable on the error path.
    pub fn checked_fill(
        &mut self,
        value: Value,
        start: usize,
        end: usize,
    ) -> Result<(), ArrayStoreError> {
        if !self.element_matches(value) {
            return Err(ArrayStoreError::ElementType {
                expected: self.element_type.clone(),
            });
        }
        let end = end.min(self.elements.len());
        for slot in self.elements.iter_mut().take(end).skip(start) {
            *slot = value;
        }
        Ok(())
    }

    /// Replace the element range `[start, end)` with `items`, honoring the
    /// resolved element constraint for each inserted item. Validates every item
    /// before mutating; returns the removed elements on success.
    pub fn checked_splice(
        &mut self,
        start: usize,
        end: usize,
        items: Vec<Value>,
    ) -> Result<Vec<Value>, ArrayStoreError> {
        for item in &items {
            if !self.element_matches(*item) {
                return Err(ArrayStoreError::ElementType {
                    expected: self.element_type.clone(),
                });
            }
        }
        let len = self.elements.len();
        let start = start.min(len);
        let end = end.clamp(start, len);
        let removed: Vec<Value> = self.elements[start..end].to_vec();
        self.elements.splice(start..end, items);
        Ok(removed)
    }

    /// Pop element from end of array
    pub fn pop(&mut self) -> Option<Value> {
        self.elements.pop()
    }

    /// Shift element from beginning of array
    pub fn shift(&mut self) -> Option<Value> {
        if self.elements.is_empty() {
            None
        } else {
            Some(self.elements.remove(0))
        }
    }

    /// Unshift element to beginning of array, returns new length
    pub fn unshift(&mut self, value: Value) -> usize {
        self.elements.insert(0, value);
        self.elements.len()
    }

    /// Find index of value, returns -1 if not found
    /// For string elements, uses value equality instead of pointer equality
    pub fn index_of(&self, value: Value) -> i32 {
        for (i, elem) in self.elements.iter().enumerate() {
            // Check equality - for strings, compare string data instead of pointers
            let equal = if *elem == value {
                true
            } else if elem.is_ptr() && value.is_ptr() {
                // Both are pointers - check if they're strings and compare data
                let elem_str = unsafe { elem.as_ptr::<RayaString>() };
                let val_str = unsafe { value.as_ptr::<RayaString>() };
                if let (Some(e_ptr), Some(v_ptr)) = (elem_str, val_str) {
                    let e = unsafe { &*e_ptr.as_ptr() };
                    let v = unsafe { &*v_ptr.as_ptr() };
                    e.data == v.data
                } else {
                    false
                }
            } else {
                false
            };

            if equal {
                return i as i32;
            }
        }
        -1
    }

    /// Check if array contains value
    pub fn includes(&self, value: Value) -> bool {
        self.index_of(value) >= 0
    }
}

/// Closure object (heap-allocated)
///
/// A closure captures the function ID and any captured variables from
/// the enclosing scope. When the closure is called, the captured values
/// are available to the function body.
#[derive(Debug, Clone)]
pub struct Closure {
    /// Function ID (index into module's function table)
    pub func_id: usize,
    /// Captured variable values
    pub captures: Vec<Value>,
    /// Optional explicit module binding for cross-module calls.
    pub module: Option<Arc<crate::compiler::Module>>,
}

impl Closure {
    /// Create a new closure with captured variables
    pub fn new(func_id: usize, captures: Vec<Value>) -> Self {
        Self {
            func_id,
            captures,
            module: None,
        }
    }

    /// Create a closure bound to a specific module.
    pub fn with_module(
        func_id: usize,
        captures: Vec<Value>,
        module: Arc<crate::compiler::Module>,
    ) -> Self {
        Self {
            func_id,
            captures,
            module: Some(module),
        }
    }

    /// Get the function ID
    pub fn func_id(&self) -> usize {
        self.func_id
    }

    /// Get the optional module binding.
    pub fn module(&self) -> Option<Arc<crate::compiler::Module>> {
        self.module.clone()
    }

    /// Get a captured variable by index
    pub fn get_captured(&self, index: usize) -> Option<Value> {
        self.captures.get(index).copied()
    }

    /// Set a captured variable by index
    pub fn set_captured(&mut self, index: usize, value: Value) -> Result<(), String> {
        if index < self.captures.len() {
            self.captures[index] = value;
            Ok(())
        } else {
            Err(format!(
                "Captured variable index {} out of bounds (closure has {} captures)",
                index,
                self.captures.len()
            ))
        }
    }

    /// Get number of captured variables
    pub fn capture_count(&self) -> usize {
        self.captures.len()
    }
}

/// A method bound to its receiver object (heap-allocated).
///
/// Created when accessing `obj.method` as a value (not a call).
/// When called, the receiver is automatically passed as `this` (locals[0]).
#[derive(Debug, Clone)]
pub struct BoundMethod {
    /// The receiver object (becomes `this`)
    pub receiver: Value,
    /// Function ID of the method (resolved from vtable at bind time)
    pub func_id: usize,
    /// Optional explicit module binding for cross-module method dispatch.
    pub module: Option<Arc<crate::compiler::Module>>,
}

/// A native method bound to its receiver object.
///
/// Used when dynamic member lookup resolves to a VM native method instead of
/// a bytecode function-backed class/vtable method.
#[derive(Debug, Clone)]
pub struct BoundNativeMethod {
    /// The receiver object (becomes arg0 to native dispatch).
    pub receiver: Value,
    /// VM native method ID.
    pub native_id: u16,
}

/// RefCell - A heap-allocated mutable cell for capture-by-reference semantics
///
/// When a variable is captured by a closure AND modified (either in the closure
/// or in the outer scope), both need to share the same storage. RefCell provides
/// this shared mutable storage - both the outer scope and closure hold a pointer
/// to the same RefCell, and all reads/writes go through it.
#[derive(Debug, Clone)]
pub struct RefCell {
    /// The contained value
    pub value: Value,
}

impl RefCell {
    /// Create a new RefCell with an initial value
    pub fn new(value: Value) -> Self {
        Self { value }
    }

    /// Get the current value
    pub fn get(&self) -> Value {
        self.value
    }

    /// Set a new value
    pub fn set(&mut self, value: Value) {
        self.value = value;
    }
}

/// String object (heap-allocated) with cached metadata for fast comparison
///
/// The hash is computed lazily on first comparison and cached for O(1)
/// subsequent access. This enables the multi-level SEQ optimization:
/// 1. Pointer equality (O(1))
/// 2. Length check (O(1))
/// 3. Hash check (O(1) after first computation)
/// 4. Character comparison (O(n)) - only if all else fails
pub struct RayaString {
    /// UTF-8 string data
    pub data: String,
    /// Cached hash plus one (0 = uncached).
    hash_plus_one: AtomicU64,
}

impl std::fmt::Debug for RayaString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RayaString")
            .field("data", &self.data)
            .field(
                "hash",
                &match self.hash_plus_one.load(Ordering::Relaxed) {
                    0 => None,
                    value => Some(value - 1),
                },
            )
            .finish()
    }
}

impl Clone for RayaString {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            hash_plus_one: AtomicU64::new(self.hash_plus_one.load(Ordering::Relaxed)),
        }
    }
}

impl RayaString {
    /// Create a new string
    pub fn new(data: String) -> Self {
        Self {
            data,
            hash_plus_one: AtomicU64::new(0),
        }
    }

    /// Get string length (in bytes)
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if string is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get or compute hash (O(n) first time, O(1) subsequent)
    pub fn hash(&self) -> u64 {
        let cached = self.hash_plus_one.load(Ordering::Acquire);
        if cached != 0 {
            return cached - 1;
        }
        let h = self.compute_hash();
        self.hash_plus_one.compare_exchange(
            0,
            h.wrapping_add(1),
            Ordering::AcqRel,
            Ordering::Acquire,
        ).ok();
        h
    }

    /// Compute hash using FxHasher for speed
    fn compute_hash(&self) -> u64 {
        let mut hasher = rustc_hash::FxHasher::default();
        self.data.hash(&mut hasher);
        hasher.finish()
    }

    /// Concatenate two strings
    pub fn concat(&self, other: &RayaString) -> RayaString {
        RayaString::new(format!("{}{}", self.data, other.data))
    }
}

// ============================================================================
// Builtin collection types
// ============================================================================

/// Wrapper type for Value that implements Hash and Eq for use in HashMap/HashSet
///
/// For pointer values (like strings), this compares by content rather than pointer address.
/// For primitive values, it uses raw bit comparison.
#[derive(Clone, Copy, Debug)]
pub struct HashableValue(pub Value);

impl HashableValue {
    /// Try to get the string content if this value is a RayaString pointer
    fn try_as_string(&self) -> Option<&str> {
        if self.0.is_ptr() {
            let raw_ptr = unsafe { self.0.as_ptr::<u8>() }?;
            let header = unsafe { &*crate::vm::gc::header_ptr_from_value_ptr(raw_ptr.as_ptr()) };
            if header.type_id() == TypeId::of::<RayaString>() {
                let ptr = unsafe { self.0.as_ptr::<RayaString>() }?;
                let raya_str = unsafe { &*ptr.as_ptr() };
                return Some(raya_str.data.as_str());
            }
        }
        None
    }

    /// Canonical numeric key for SameValueZero-like numeric matching.
    ///
    /// - `int` and `number` compare by numeric value
    /// - `-0` and `+0` are treated as equal
    /// - all `NaN` payloads normalize to one canonical key
    fn numeric_key_bits(&self) -> Option<u64> {
        let value = if let Some(i) = self.0.as_i32() {
            i as f64
        } else if let Some(n) = self.0.as_f64() {
            n
        } else {
            return None;
        };

        if value == 0.0 {
            return Some(0.0f64.to_bits());
        }
        if value.is_nan() {
            return Some(f64::NAN.to_bits());
        }
        Some(value.to_bits())
    }
}

impl Hash for HashableValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // For string pointers, hash the string content
        if let Some(s) = self.try_as_string() {
            // Hash a discriminator first to distinguish strings from raw values
            1u8.hash(state);
            s.hash(state);
        } else if let Some(bits) = self.numeric_key_bits() {
            // Numeric values use canonicalized SameValueZero-like hashing.
            2u8.hash(state);
            bits.hash(state);
        } else {
            // For primitive values (numbers, booleans, null), use raw bits
            0u8.hash(state);
            self.0.raw().hash(state);
        }
    }
}

impl PartialEq for HashableValue {
    fn eq(&self, other: &Self) -> bool {
        // First try to compare as strings (by content)
        match (self.try_as_string(), other.try_as_string()) {
            (Some(s1), Some(s2)) => s1 == s2,
            (None, None) => match (self.numeric_key_bits(), other.numeric_key_bits()) {
                (Some(a), Some(b)) => a == b,
                _ => self.0 == other.0,
            },
            _ => false,                        // One is string, one is not
        }
    }
}

impl Eq for HashableValue {}

/// Map builtin - generic key-value store
/// Native IDs: 0x0800-0x080A
#[derive(Debug, Clone)]
pub struct MapObject {
    /// Internal HashMap storage
    pub inner: HashMap<HashableValue, Value>,
}

impl MapObject {
    /// Create a new empty map
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    /// Get the number of entries
    pub fn size(&self) -> usize {
        self.inner.len()
    }

    /// Get a value by key
    pub fn get(&self, key: Value) -> Option<Value> {
        self.inner.get(&HashableValue(key)).copied()
    }

    /// Set a key-value pair
    pub fn set(&mut self, key: Value, value: Value) {
        self.inner.insert(HashableValue(key), value);
    }

    /// Check if key exists
    pub fn has(&self, key: Value) -> bool {
        self.inner.contains_key(&HashableValue(key))
    }

    /// Delete a key, returns true if key existed
    pub fn delete(&mut self, key: Value) -> bool {
        self.inner.remove(&HashableValue(key)).is_some()
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Get all keys as a vector
    pub fn keys(&self) -> Vec<Value> {
        self.inner.keys().map(|k| k.0).collect()
    }

    /// Get all values as a vector
    pub fn values(&self) -> Vec<Value> {
        self.inner.values().copied().collect()
    }

    /// Get all entries as key-value pairs
    pub fn entries(&self) -> Vec<(Value, Value)> {
        self.inner.iter().map(|(k, v)| (k.0, *v)).collect()
    }
}

impl Default for MapObject {
    fn default() -> Self {
        Self::new()
    }
}

/// Set builtin - collection of unique values
/// Native IDs: 0x0900-0x090A
#[derive(Debug, Clone)]
pub struct SetObject {
    /// Internal HashSet storage
    pub inner: HashSet<HashableValue>,
}

impl SetObject {
    /// Create a new empty set
    pub fn new() -> Self {
        Self {
            inner: HashSet::new(),
        }
    }

    /// Get the number of elements
    pub fn size(&self) -> usize {
        self.inner.len()
    }

    /// Add a value to the set
    pub fn add(&mut self, value: Value) {
        self.inner.insert(HashableValue(value));
    }

    /// Check if value exists
    pub fn has(&self, value: Value) -> bool {
        self.inner.contains(&HashableValue(value))
    }

    /// Delete a value, returns true if value existed
    pub fn delete(&mut self, value: Value) -> bool {
        self.inner.remove(&HashableValue(value))
    }

    /// Clear all elements
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Get all values as a vector
    pub fn values(&self) -> Vec<Value> {
        self.inner.iter().map(|v| v.0).collect()
    }
}

impl Default for SetObject {
    fn default() -> Self {
        Self::new()
    }
}

/// Buffer builtin - raw binary data buffer
/// Native IDs: 0x0700-0x070B
#[derive(Debug, Clone)]
pub struct Buffer {
    /// Raw byte data
    pub data: Vec<u8>,
}

impl Buffer {
    /// Create a new buffer of given size (zero-filled)
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0; size],
        }
    }

    /// Get buffer length in bytes
    pub fn length(&self) -> usize {
        self.data.len()
    }

    /// Get byte at index
    pub fn get_byte(&self, index: usize) -> Option<u8> {
        self.data.get(index).copied()
    }

    /// Set byte at index
    pub fn set_byte(&mut self, index: usize, value: u8) -> Result<(), String> {
        if index < self.data.len() {
            self.data[index] = value;
            Ok(())
        } else {
            Err(format!(
                "Buffer index {} out of bounds (length: {})",
                index,
                self.data.len()
            ))
        }
    }

    /// Get 32-bit signed integer at index (little-endian)
    pub fn get_int32(&self, index: usize) -> Option<i32> {
        if index + 4 <= self.data.len() {
            let bytes = [
                self.data[index],
                self.data[index + 1],
                self.data[index + 2],
                self.data[index + 3],
            ];
            Some(i32::from_le_bytes(bytes))
        } else {
            None
        }
    }

    /// Set 32-bit signed integer at index (little-endian)
    pub fn set_int32(&mut self, index: usize, value: i32) -> Result<(), String> {
        if index + 4 <= self.data.len() {
            let bytes = value.to_le_bytes();
            self.data[index..index + 4].copy_from_slice(&bytes);
            Ok(())
        } else {
            Err(format!(
                "Buffer index {} out of bounds for int32 (length: {})",
                index,
                self.data.len()
            ))
        }
    }

    /// Get 64-bit float at index (little-endian)
    pub fn get_float64(&self, index: usize) -> Option<f64> {
        if index + 8 <= self.data.len() {
            let bytes = [
                self.data[index],
                self.data[index + 1],
                self.data[index + 2],
                self.data[index + 3],
                self.data[index + 4],
                self.data[index + 5],
                self.data[index + 6],
                self.data[index + 7],
            ];
            Some(f64::from_le_bytes(bytes))
        } else {
            None
        }
    }

    /// Set 64-bit float at index (little-endian)
    pub fn set_float64(&mut self, index: usize, value: f64) -> Result<(), String> {
        if index + 8 <= self.data.len() {
            let bytes = value.to_le_bytes();
            self.data[index..index + 8].copy_from_slice(&bytes);
            Ok(())
        } else {
            Err(format!(
                "Buffer index {} out of bounds for float64 (length: {})",
                index,
                self.data.len()
            ))
        }
    }

    /// Create a slice of this buffer (returns new buffer)
    pub fn slice(&self, start: usize, end: usize) -> Buffer {
        let end = end.min(self.data.len());
        let start = start.min(end);
        Buffer {
            data: self.data[start..end].to_vec(),
        }
    }
}

/// RegExp builtin - regular expression pattern
/// Native IDs: 0x0A01-0x0A06
#[derive(Debug, Clone)]
pub struct RegExpObject {
    /// The pattern string
    pub pattern: String,
    /// Regex flags (g=global, i=ignoreCase, m=multiline)
    pub flags: String,
    /// Compiled regex (using Rust regex crate)
    pub compiled: regex::Regex,
}

impl RegExpObject {
    /// Create a new RegExp from pattern and flags
    pub fn new(pattern: &str, flags: &str) -> Result<Self, String> {
        // Build regex pattern with flags
        let mut regex_pattern = String::new();

        // Handle flags
        let case_insensitive = flags.contains('i');
        let multiline = flags.contains('m');

        if case_insensitive || multiline {
            regex_pattern.push_str("(?");
            if case_insensitive {
                regex_pattern.push('i');
            }
            if multiline {
                regex_pattern.push('m');
            }
            regex_pattern.push(')');
        }

        regex_pattern.push_str(pattern);

        let compiled = regex::Regex::new(&regex_pattern)
            .map_err(|e| format!("Invalid regular expression: {}", e))?;

        Ok(Self {
            pattern: pattern.to_string(),
            flags: flags.to_string(),
            compiled,
        })
    }

    /// Get the source pattern
    pub fn source(&self) -> &str {
        &self.pattern
    }

    /// Get the flags string
    pub fn flags(&self) -> &str {
        &self.flags
    }

    /// Check if global flag is set
    pub fn global(&self) -> bool {
        self.flags.contains('g')
    }

    /// Check if case-insensitive flag is set
    pub fn ignore_case(&self) -> bool {
        self.flags.contains('i')
    }

    /// Check if multiline flag is set
    pub fn multiline(&self) -> bool {
        self.flags.contains('m')
    }

    /// Test if pattern matches string
    pub fn test(&self, text: &str) -> bool {
        self.compiled.is_match(text)
    }

    /// Execute pattern on string, return first match
    /// Returns (matched_text, index, groups) or None
    pub fn exec(&self, text: &str) -> Option<(String, usize, Vec<String>)> {
        self.compiled.captures(text).map(|caps| {
            let full_match = caps.get(0).unwrap();
            let matched_text = full_match.as_str().to_string();
            let index = full_match.start();

            // Collect captured groups (skip group 0 which is the full match)
            let groups: Vec<String> = caps
                .iter()
                .skip(1)
                .map(|m| m.map(|m| m.as_str().to_string()).unwrap_or_default())
                .collect();

            (matched_text, index, groups)
        })
    }

    /// Execute pattern on string, return all matches
    pub fn exec_all(&self, text: &str) -> Vec<(String, usize, Vec<String>)> {
        self.compiled
            .captures_iter(text)
            .map(|caps| {
                let full_match = caps.get(0).unwrap();
                let matched_text = full_match.as_str().to_string();
                let index = full_match.start();

                let groups: Vec<String> = caps
                    .iter()
                    .skip(1)
                    .map(|m| m.map(|m| m.as_str().to_string()).unwrap_or_default())
                    .collect();

                (matched_text, index, groups)
            })
            .collect()
    }

    /// Replace first match (or all if global)
    pub fn replace(&self, text: &str, replacement: &str) -> String {
        if self.global() {
            self.compiled.replace_all(text, replacement).to_string()
        } else {
            self.compiled.replace(text, replacement).to_string()
        }
    }

    /// Split string by pattern
    pub fn split(&self, text: &str, limit: Option<usize>) -> Vec<String> {
        match limit {
            Some(n) => self
                .compiled
                .splitn(text, n)
                .map(|s| s.to_string())
                .collect(),
            None => self.compiled.split(text).map(|s| s.to_string()).collect(),
        }
    }
}

/// Date builtin - date and time handling
/// Native IDs: 0x0B00-0x0B23
#[derive(Debug, Clone, Copy)]
pub struct DateObject {
    /// Timestamp in milliseconds since Unix epoch
    pub timestamp_ms: i64,
}

impl DateObject {
    /// Create a new date with current time
    pub fn now() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        Self { timestamp_ms }
    }

    /// Create a date from timestamp (milliseconds since epoch)
    pub fn from_timestamp(timestamp_ms: i64) -> Self {
        Self { timestamp_ms }
    }

    /// Get timestamp in milliseconds
    pub fn get_time(&self) -> i64 {
        self.timestamp_ms
    }

    // ---- Civil date helpers (Howard Hinnant's algorithms) ----

    /// Convert days since Unix epoch to (year, month[1-12], day[1-31])
    fn civil_from_days(days: i64) -> (i32, i32, i32) {
        let z = days + 719468;
        let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
        let doe = (z - era * 146097) as u32;
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let y = yoe as i64 + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let y = if m <= 2 { y + 1 } else { y };
        (y as i32, m as i32, d as i32)
    }

    /// Convert (year, month[1-12], day[1-31]) to days since Unix epoch
    fn days_from_civil(y: i32, m: i32, d: i32) -> i64 {
        let y = if m <= 2 { y as i64 - 1 } else { y as i64 };
        let era = (if y >= 0 { y } else { y - 399 }) / 400;
        let yoe = (y - era * 400) as u32;
        let m = m as u32;
        let d = d as u32;
        let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146097 + doe as i64 - 719468
    }

    /// Decompose timestamp into (year, month[0-11], day[1-31], hour, min, sec, ms)
    fn decompose(&self) -> (i32, i32, i32, i32, i32, i32, i32) {
        let total_ms = self.timestamp_ms;
        let ms = ((total_ms % 1000 + 1000) % 1000) as i32;
        let total_secs = if total_ms >= 0 {
            total_ms / 1000
        } else {
            (total_ms - 999) / 1000
        };
        let day_secs = ((total_secs % 86400) + 86400) % 86400;
        let hour = (day_secs / 3600) as i32;
        let min = ((day_secs % 3600) / 60) as i32;
        let sec = (day_secs % 60) as i32;
        let days = (total_secs - day_secs) / 86400;
        let (y, m, d) = Self::civil_from_days(days);
        (y, m - 1, d, hour, min, sec, ms)
    }

    /// Recompose from (year, month[0-11], day[1-31], hour, min, sec, ms) to timestamp_ms
    fn recompose(y: i32, m: i32, d: i32, h: i32, min: i32, sec: i32, ms: i32) -> i64 {
        let days = Self::days_from_civil(y, m + 1, d);
        days * 86_400_000
            + h as i64 * 3_600_000
            + min as i64 * 60_000
            + sec as i64 * 1000
            + ms as i64
    }

    // ---- Getters ----

    /// Get year (4-digit)
    pub fn get_full_year(&self) -> i32 {
        self.decompose().0
    }

    /// Get month (0-11)
    pub fn get_month(&self) -> i32 {
        self.decompose().1
    }

    /// Get day of month (1-31)
    pub fn get_date(&self) -> i32 {
        self.decompose().2
    }

    /// Get day of week (0-6, 0=Sunday)
    pub fn get_day(&self) -> i32 {
        let total_secs = if self.timestamp_ms >= 0 {
            self.timestamp_ms / 1000
        } else {
            (self.timestamp_ms - 999) / 1000
        };
        let day_secs = ((total_secs % 86400) + 86400) % 86400;
        let days = (total_secs - day_secs) / 86400;
        (((days + 4) % 7 + 7) % 7) as i32
    }

    /// Get hours (0-23)
    pub fn get_hours(&self) -> i32 {
        self.decompose().3
    }

    /// Get minutes (0-59)
    pub fn get_minutes(&self) -> i32 {
        self.decompose().4
    }

    /// Get seconds (0-59)
    pub fn get_seconds(&self) -> i32 {
        self.decompose().5
    }

    /// Get milliseconds (0-999)
    pub fn get_milliseconds(&self) -> i32 {
        self.decompose().6
    }

    // ---- Setters (return new timestamp) ----

    pub fn set_full_year(&self, year: i32) -> i64 {
        let (_, m, d, h, min, sec, ms) = self.decompose();
        Self::recompose(year, m, d, h, min, sec, ms)
    }

    pub fn set_month(&self, month: i32) -> i64 {
        let (y, _, d, h, min, sec, ms) = self.decompose();
        Self::recompose(y, month, d, h, min, sec, ms)
    }

    pub fn set_date(&self, day: i32) -> i64 {
        let (y, m, _, h, min, sec, ms) = self.decompose();
        Self::recompose(y, m, day, h, min, sec, ms)
    }

    pub fn set_hours(&self, hours: i32) -> i64 {
        let (y, m, d, _, min, sec, ms) = self.decompose();
        Self::recompose(y, m, d, hours, min, sec, ms)
    }

    pub fn set_minutes(&self, minutes: i32) -> i64 {
        let (y, m, d, h, _, sec, ms) = self.decompose();
        Self::recompose(y, m, d, h, minutes, sec, ms)
    }

    pub fn set_seconds(&self, seconds: i32) -> i64 {
        let (y, m, d, h, min, _, ms) = self.decompose();
        Self::recompose(y, m, d, h, min, seconds, ms)
    }

    pub fn set_milliseconds(&self, millis: i32) -> i64 {
        let (y, m, d, h, min, sec, _) = self.decompose();
        Self::recompose(y, m, d, h, min, sec, millis)
    }

    // ---- Formatting ----

    fn day_name(dow: i32) -> &'static str {
        match dow {
            0 => "Sun",
            1 => "Mon",
            2 => "Tue",
            3 => "Wed",
            4 => "Thu",
            5 => "Fri",
            6 => "Sat",
            _ => "???",
        }
    }

    fn month_name(m: i32) -> &'static str {
        match m {
            0 => "Jan",
            1 => "Feb",
            2 => "Mar",
            3 => "Apr",
            4 => "May",
            5 => "Jun",
            6 => "Jul",
            7 => "Aug",
            8 => "Sep",
            9 => "Oct",
            10 => "Nov",
            11 => "Dec",
            _ => "???",
        }
    }

    /// Human-readable string: "Mon Jan 15 2024 10:30:00"
    pub fn to_string_repr(&self) -> String {
        let (y, m, d, h, min, sec, _) = self.decompose();
        let dow = self.get_day();
        format!(
            "{} {} {:02} {:04} {:02}:{:02}:{:02}",
            Self::day_name(dow),
            Self::month_name(m),
            d,
            y,
            h,
            min,
            sec
        )
    }

    /// ISO 8601: "2024-01-15T10:30:00.000Z"
    pub fn to_iso_string(&self) -> String {
        let (y, m, d, h, min, sec, ms) = self.decompose();
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
            y,
            m + 1,
            d,
            h,
            min,
            sec,
            ms
        )
    }

    /// Date portion: "Mon Jan 15 2024"
    pub fn to_date_string(&self) -> String {
        let (y, m, d, _, _, _, _) = self.decompose();
        let dow = self.get_day();
        format!(
            "{} {} {:02} {:04}",
            Self::day_name(dow),
            Self::month_name(m),
            d,
            y
        )
    }

    /// Time portion: "10:30:00"
    pub fn to_time_string(&self) -> String {
        let (_, _, _, h, min, sec, _) = self.decompose();
        format!("{:02}:{:02}:{:02}", h, min, sec)
    }

    // ---- Parsing ----

    /// Parse ISO 8601 date string to timestamp ms
    pub fn parse(s: &str) -> Option<i64> {
        let s = s.trim().trim_end_matches('Z');
        let (date_part, time_part) = if let Some(idx) = s.find('T') {
            (&s[..idx], Some(&s[idx + 1..]))
        } else {
            (s, None)
        };

        let date_parts: Vec<&str> = date_part.split('-').collect();
        if date_parts.len() != 3 {
            return None;
        }
        let y: i32 = date_parts[0].parse().ok()?;
        let m: i32 = date_parts[1].parse().ok()?;
        let d: i32 = date_parts[2].parse().ok()?;
        if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
            return None;
        }

        let (h, min, sec, ms) = if let Some(tp) = time_part {
            let (time_str, ms) = if let Some(dot_idx) = tp.find('.') {
                let ms: i32 = tp[dot_idx + 1..].parse().ok()?;
                (&tp[..dot_idx], ms)
            } else {
                (tp, 0)
            };
            let time_parts: Vec<&str> = time_str.split(':').collect();
            let h: i32 = time_parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
            let min: i32 = time_parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            let sec: i32 = time_parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
            (h, min, sec, ms)
        } else {
            (0, 0, 0, 0)
        };

        Some(Self::recompose(y, m - 1, d, h, min, sec, ms))
    }
}

/// Channel builtin - inter-task communication primitive
/// Native IDs: 0x0400-0x0408
///
/// Simple bounded queue. All waiter tracking and waking is managed by the
/// reactor's channel_waiters + pair matching. The interpreter uses try_send/
/// try_receive and suspends on failure; the reactor retries and matches
/// sender-receiver pairs (critical for unbuffered channels).
pub struct ChannelObject {
    /// Internal state protected by a mutex
    inner: parking_lot::Mutex<ChannelInner>,
}

/// Internal channel state
struct ChannelInner {
    /// Buffer capacity (0 = unbuffered)
    capacity: usize,
    /// Message queue
    queue: VecDeque<Value>,
    /// Whether channel is closed
    closed: bool,
}

impl std::fmt::Debug for ChannelObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.lock();
        f.debug_struct("ChannelObject")
            .field("capacity", &inner.capacity)
            .field("length", &inner.queue.len())
            .field("closed", &inner.closed)
            .finish()
    }
}

impl ChannelObject {
    /// Create a new channel with given buffer capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: parking_lot::Mutex::new(ChannelInner {
                capacity,
                queue: VecDeque::with_capacity(capacity),
                closed: false,
            }),
        }
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.inner.lock().capacity
    }

    /// Get number of items in queue
    pub fn length(&self) -> usize {
        self.inner.lock().queue.len()
    }

    /// Check if channel is closed
    pub fn is_closed(&self) -> bool {
        self.inner.lock().closed
    }

    /// Snapshot the buffered queue without mutating the live channel.
    pub fn queued_values(&self) -> Vec<Value> {
        self.inner.lock().queue.iter().copied().collect()
    }

    /// Close the channel. Reactor handles waking any waiting tasks.
    pub fn close(&self) {
        self.inner.lock().closed = true;
    }

    /// Try to send a value (non-blocking)
    /// Returns true if sent, false if full or closed
    pub fn try_send(&self, value: Value) -> bool {
        let mut inner = self.inner.lock();
        if inner.closed {
            return false;
        }
        if inner.queue.len() < inner.capacity {
            inner.queue.push_back(value);
            true
        } else {
            false
        }
    }

    /// Try to receive a value (non-blocking)
    /// Returns Some(value) if available, None if empty
    pub fn try_receive(&self) -> Option<Value> {
        self.inner.lock().queue.pop_front()
    }
}

// ============================================================================
// Proxy Objects (Phase 9 Reflect API)
// ============================================================================

/// Proxy object for intercepting property access and method calls
///
/// A Proxy wraps a target object and delegates operations through
/// trap handlers. When a property is accessed or a method is called
/// on the proxy, the corresponding trap handler is invoked if present.
///
/// Traps:
/// - `get(target, property)` - intercept property read
/// - `set(target, property, value)` - intercept property write
/// - `has(target, property)` - intercept property existence check
/// - `invoke(target, method, args)` - intercept method call
#[derive(Debug, Clone)]
pub struct Proxy {
    /// Unique proxy ID for identity checking
    pub proxy_id: u64,
    /// The underlying target object (as a Value pointing to Object)
    pub target: Value,
    /// The handler object containing trap functions (as a Value pointing to Object)
    /// Handler fields by name:
    /// - "get": (target, property) -> value
    /// - "set": (target, property, value) -> boolean
    /// - "has": (target, property) -> boolean
    /// - "invoke": (target, method, args) -> value
    pub handler: Value,
}

impl Proxy {
    /// Create a new proxy wrapping the target with the given handler
    pub fn new(target: Value, handler: Value) -> Self {
        Self {
            proxy_id: generate_object_id(),
            target,
            handler,
        }
    }

    /// Get the target object
    pub fn get_target(&self) -> Value {
        self.target
    }

    /// Get the handler object
    pub fn get_handler(&self) -> Value {
        self.handler
    }
}

impl Clone for ChannelObject {
    fn clone(&self) -> Self {
        let inner = self.inner.lock();
        Self {
            inner: parking_lot::Mutex::new(ChannelInner {
                capacity: inner.capacity,
                queue: inner.queue.clone(),
                closed: inner.closed,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::bytecode::RuntimeTypeDescriptor as D;

    #[test]
    fn typed_checked_set_and_push_enforce_primitive_constraint() {
        // I32-typed array: i32 stores succeed, bool/f64 stores are ElementType errors.
        let mut arr = Array::with_element_type(0, Some(D::I32), 2);
        assert!(arr.checked_set(0, Value::i32(7)).is_ok());
        assert!(matches!(
            arr.checked_set(1, Value::bool(true)),
            Err(ArrayStoreError::ElementType { .. })
        ));
        assert!(matches!(
            arr.checked_push(Value::f64(1.5)),
            Err(ArrayStoreError::ElementType { .. })
        ));
        assert!(arr.checked_push(Value::i32(9)).is_ok());
        // null is always accepted, even into a typed slot.
        assert!(arr.checked_set(0, Value::null()).is_ok());
    }

    #[test]
    fn typed_checked_set_out_of_bounds_is_distinct_from_type_error() {
        let mut arr = Array::with_element_type(0, Some(D::I32), 1);
        assert!(matches!(
            arr.checked_set(5, Value::i32(1)),
            Err(ArrayStoreError::OutOfBounds { .. })
        ));
    }

    #[test]
    fn dynamic_array_accepts_any_value() {
        // element_type == None => accept-all.
        let mut arr = Array::new(0, 1);
        assert!(arr.checked_set(0, Value::i32(1)).is_ok());
        assert!(arr.checked_push(Value::bool(true)).is_ok());
        assert!(arr.checked_push(Value::f64(2.0)).is_ok());
    }

    #[test]
    fn typed_unshift_and_fill_enforce_constraint() {
        let mut arr = Array::with_element_type(0, Some(D::Bool), 3);
        assert!(arr.checked_unshift(Value::bool(false)).is_ok());
        assert!(matches!(
            arr.checked_unshift(Value::i32(1)),
            Err(ArrayStoreError::ElementType { .. })
        ));
        assert!(arr.checked_fill(Value::bool(true), 0, 3).is_ok());
        assert!(matches!(
            arr.checked_fill(Value::i32(0), 0, 3),
            Err(ArrayStoreError::ElementType { .. })
        ));
    }

    #[test]
    fn typed_splice_validates_items_before_mutating() {
        let mut arr = Array::with_element_type(0, Some(D::I32), 3);
        arr.checked_set(0, Value::i32(1)).unwrap();
        arr.checked_set(1, Value::i32(2)).unwrap();
        arr.checked_set(2, Value::i32(3)).unwrap();
        // A bad item is rejected and the array is left unchanged.
        assert!(matches!(
            arr.checked_splice(1, 2, vec![Value::bool(true)]),
            Err(ArrayStoreError::ElementType { .. })
        ));
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.get(1), Some(Value::i32(2)));
        // A valid splice removes the range [1,3) (two elements) and inserts
        // one item: [1,2,3] -> [1,20], removed = [2,3].
        let removed = arr.checked_splice(1, 3, vec![Value::i32(20)]).unwrap();
        assert_eq!(removed, vec![Value::i32(2), Value::i32(3)]);
        assert_eq!(arr.len(), 2);
        assert_eq!(arr.get(0), Some(Value::i32(1)));
        assert_eq!(arr.get(1), Some(Value::i32(20)));
    }

    #[test]
    fn f64_constraint_admits_integral_i32_bridge() {
        // F64 slot admits i32 (the interpreter's integral bridge) and f64.
        let mut arr = Array::with_element_type(0, Some(D::F64), 2);
        assert!(arr.checked_set(0, Value::f64(1.5)).is_ok());
        assert!(arr.checked_set(1, Value::i32(3)).is_ok());
        assert!(matches!(
            arr.checked_push(Value::bool(true)),
            Err(ArrayStoreError::ElementType { .. })
        ));
    }

    #[test]
    fn complex_descriptor_constraints_accept_all_by_policy() {
        // Object/Array/Tuple/Function/Task/AnyValue/Ref are accept-all: they are
        // not soundly checkable without module-local context, so they must not
        // reject valid values. Verify a non-null primitive is accepted.
        for descriptor in [D::AnyValue, D::Ref, D::Void] {
            let label = format!("{descriptor:?}");
            let mut arr = Array::with_element_type(0, Some(descriptor), 1);
            assert!(
                arr.checked_set(0, Value::i32(1)).is_ok(),
                "descriptor {label} should accept any value"
            );
        }
    }

    #[test]
    fn test_object_creation() {
        let obj = Object::new_synthetic_structural(3);
        assert_eq!(obj.field_count(), 3);
        assert_eq!(obj.nominal_type_id_usize(), None);
        assert_ne!(obj.layout_id(), 0);
    }

    #[test]
    fn test_object_field_access() {
        let mut obj = Object::new_synthetic_structural(2);
        let value = Value::i32(42);

        obj.set_field(0, value).unwrap();
        assert_eq!(obj.get_field(0).unwrap(), value);

        obj.set_field(1, Value::bool(true)).unwrap();
        assert_eq!(obj.get_field(1).unwrap(), Value::bool(true));
    }

    #[test]
    fn test_object_field_bounds() {
        let mut obj = Object::new_synthetic_structural(2);
        assert!(obj.set_field(2, Value::null()).is_err());
        assert_eq!(obj.get_field(10), None);
    }

    #[test]
    fn test_class_creation() {
        let class = Class::new(0, "Point".to_string(), 2);
        assert_eq!(class.id, 0);
        assert_eq!(class.name, "Point");
        assert_eq!(class.field_count, 2);
        assert_eq!(class.parent_id, None);
    }

    #[test]
    fn test_class_with_parent() {
        let class = Class::with_parent(1, "ColoredPoint".to_string(), 3, 0);
        assert_eq!(class.parent_id, Some(0));
        assert_eq!(class.field_count, 3);
    }

    #[test]
    fn test_structural_layout_ids_use_tagged_domain() {
        let layout_id = layout_id_from_ordered_names(&["a".to_string(), "b".to_string()]);
        assert_ne!(layout_id, 0);
        assert_eq!(
            layout_id & STRUCTURAL_LAYOUT_ID_TAG,
            STRUCTURAL_LAYOUT_ID_TAG
        );
    }

    #[test]
    fn test_vtable() {
        let mut vtable = VTable::new();
        vtable.add_method(10); // function ID 10
        vtable.add_method(20); // function ID 20

        assert_eq!(vtable.method_count(), 2);
        assert_eq!(vtable.get_method(0), Some(10));
        assert_eq!(vtable.get_method(1), Some(20));
        assert_eq!(vtable.get_method(2), None);
    }

    #[test]
    fn test_vtable_override() {
        let mut vtable = VTable::new();
        vtable.add_method(10);
        vtable.add_method(20);

        vtable.override_method(0, 30).unwrap();
        assert_eq!(vtable.get_method(0), Some(30));
    }

    #[test]
    fn test_array_creation() {
        let arr = Array::new(0, 5);
        assert_eq!(arr.len(), 5);
        assert_eq!(arr.type_id, 0);
    }

    #[test]
    fn test_array_access() {
        let mut arr = Array::new(0, 3);

        arr.set(0, Value::i32(10)).unwrap();
        arr.set(1, Value::i32(20)).unwrap();
        arr.set(2, Value::i32(30)).unwrap();

        assert_eq!(arr.get(0), Some(Value::i32(10)));
        assert_eq!(arr.get(1), Some(Value::i32(20)));
        assert_eq!(arr.get(2), Some(Value::i32(30)));
    }

    #[test]
    fn test_array_bounds() {
        let mut arr = Array::new(0, 2);
        assert!(arr.set(2, Value::null()).is_err());
        assert_eq!(arr.get(5), None);
    }

    #[test]
    fn test_string_creation() {
        let s = RayaString::new("hello".to_string());
        assert_eq!(s.len(), 5);
        assert_eq!(s.data, "hello");
    }

    #[test]
    fn test_string_concat() {
        let s1 = RayaString::new("hello".to_string());
        let s2 = RayaString::new(" world".to_string());
        let s3 = s1.concat(&s2);

        assert_eq!(s3.data, "hello world");
    }
}
