//! Runtime type descriptors (plan task R2)
//!
//! Stable, serializable type descriptions that survive into bytecode and are
//! the vocabulary for typed function signatures (B1), the typed verifier
//! (B5), and JIT signature guards (J1).
//!
//! These are deliberately coarser than checker `Type`s: they describe what
//! the runtime must know to execute exactly, per
//! docs/architecture/adr-typed-runtime-contract.md. Structural information
//! that only affects compile-time checking (unions, interfaces) is erased or
//! boxed here; enrichment (nominal ids, layout ids) lands with B2 when
//! codegen wires class information through.

use super::encoder::{BytecodeReader, BytecodeWriter, DecodeError};
use crate::parser::types::context::TypeContext;
use crate::parser::types::ty::{PrimitiveType, Type, TypeId};
use std::fmt;

/// Index into a module's interned `runtime_types` table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuntimeTypeId(pub u32);

/// Index into a module's interned `function_signatures` table.
/// `UNTYPED_SIGNATURE_ID` marks functions compiled without signature data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FunctionSignatureId(pub u32);

/// Reserved signature id meaning "no signature recorded".
pub const UNTYPED_SIGNATURE_ID: u32 = 0;
/// ABI version of the function calling convention recorded at compile time.
pub const CURRENT_ABI_VERSION: u16 = 1;

/// Well-known primitive descriptor ids. These double as compact encodings:
/// descriptors below `FIRST_COMPLEX` encode as a single tag byte.
pub mod prim {
    use super::RuntimeTypeId;
    pub const I32: RuntimeTypeId = RuntimeTypeId(0);
    pub const F64: RuntimeTypeId = RuntimeTypeId(1);
    pub const BOOL: RuntimeTypeId = RuntimeTypeId(2);
    pub const STRING: RuntimeTypeId = RuntimeTypeId(3);
    pub const NULL: RuntimeTypeId = RuntimeTypeId(4);
    pub const VOID: RuntimeTypeId = RuntimeTypeId(5);
    pub const ANY_VALUE: RuntimeTypeId = RuntimeTypeId(6);
    pub const REF: RuntimeTypeId = RuntimeTypeId(7);
}

/// First descriptor kind that requires a payload when encoded.
const FIRST_COMPLEX_TAG: u8 = 8;

/// A runtime-observable type description.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RuntimeTypeDescriptor {
    /// Signed 32-bit integer (ADR D1).
    I32,
    /// IEEE-754 binary64 (ADR D1).
    F64,
    Bool,
    String,
    Null,
    /// Return-only unit type.
    Void,
    /// Boxed dynamic value. Strict mode restricts where this can appear.
    AnyValue,
    /// Managed GC reference whose layout is not (yet) described.
    Ref,
    /// Object with known nominal class and/or structural layout.
    Object {
        nominal_id: Option<u32>,
        layout_id: u32,
    },
    Array {
        element: Box<RuntimeTypeDescriptor>,
    },
    Tuple {
        elements: Vec<RuntimeTypeDescriptor>,
    },
    Function {
        signature: FunctionSignatureId,
    },
    Task {
        result: Box<RuntimeTypeDescriptor>,
    },
}

/// Base id for descriptors stored in a module's `runtime_types` table:
/// ids below this are implicit primitives.
pub const COMPLEX_BASE: u32 = 8;

impl RuntimeTypeDescriptor {
    /// Id if this is an implicitly-encoded primitive; complex descriptors
    /// must go through a module's interned table.
    pub fn primitive_id(&self) -> Option<RuntimeTypeId> {
        match self {
            RuntimeTypeDescriptor::I32 => Some(prim::I32),
            RuntimeTypeDescriptor::F64 => Some(prim::F64),
            RuntimeTypeDescriptor::Bool => Some(prim::BOOL),
            RuntimeTypeDescriptor::String => Some(prim::STRING),
            RuntimeTypeDescriptor::Null => Some(prim::NULL),
            RuntimeTypeDescriptor::Void => Some(prim::VOID),
            RuntimeTypeDescriptor::AnyValue => Some(prim::ANY_VALUE),
            RuntimeTypeDescriptor::Ref => Some(prim::REF),
            _ => None,
        }
    }

    /// Panics for complex descriptors; see [`primitive_id`].
    pub fn to_id(&self) -> RuntimeTypeId {
        match self {
            RuntimeTypeDescriptor::I32 => prim::I32,
            RuntimeTypeDescriptor::F64 => prim::F64,
            RuntimeTypeDescriptor::Bool => prim::BOOL,
            RuntimeTypeDescriptor::String => prim::STRING,
            RuntimeTypeDescriptor::Null => prim::NULL,
            RuntimeTypeDescriptor::Void => prim::VOID,
            RuntimeTypeDescriptor::AnyValue => prim::ANY_VALUE,
            RuntimeTypeDescriptor::Ref => prim::REF,
            _ => unimplemented!("complex descriptors require module interning (B2)"),
        }
    }

    pub fn encode(&self, writer: &mut BytecodeWriter) {
        match self {
            RuntimeTypeDescriptor::I32 => writer.emit_u8(0),
            RuntimeTypeDescriptor::F64 => writer.emit_u8(1),
            RuntimeTypeDescriptor::Bool => writer.emit_u8(2),
            RuntimeTypeDescriptor::String => writer.emit_u8(3),
            RuntimeTypeDescriptor::Null => writer.emit_u8(4),
            RuntimeTypeDescriptor::Void => writer.emit_u8(5),
            RuntimeTypeDescriptor::AnyValue => writer.emit_u8(6),
            RuntimeTypeDescriptor::Ref => writer.emit_u8(7),
            RuntimeTypeDescriptor::Object {
                nominal_id,
                layout_id,
            } => {
                writer.emit_u8(FIRST_COMPLEX_TAG); // 8
                writer.emit_u8(nominal_id.is_some() as u8);
                if let Some(id) = nominal_id {
                    writer.emit_u32(*id);
                }
                writer.emit_u32(*layout_id);
            }
            RuntimeTypeDescriptor::Array { element } => {
                writer.emit_u8(FIRST_COMPLEX_TAG + 1); // 9
                element.encode(writer);
            }
            RuntimeTypeDescriptor::Tuple { elements } => {
                writer.emit_u8(FIRST_COMPLEX_TAG + 2); // 10
                writer.emit_u32(elements.len() as u32);
                for el in elements {
                    el.encode(writer);
                }
            }
            RuntimeTypeDescriptor::Function { signature } => {
                writer.emit_u8(FIRST_COMPLEX_TAG + 3); // 11
                writer.emit_u32(signature.0);
            }
            RuntimeTypeDescriptor::Task { result } => {
                writer.emit_u8(FIRST_COMPLEX_TAG + 4); // 12
                result.encode(writer);
            }
        }
    }

    pub fn decode(reader: &mut BytecodeReader<'_>) -> Result<Self, DecodeError> {
        let tag = reader.read_u8()?;
        Ok(match tag {
            0 => RuntimeTypeDescriptor::I32,
            1 => RuntimeTypeDescriptor::F64,
            2 => RuntimeTypeDescriptor::Bool,
            3 => RuntimeTypeDescriptor::String,
            4 => RuntimeTypeDescriptor::Null,
            5 => RuntimeTypeDescriptor::Void,
            6 => RuntimeTypeDescriptor::AnyValue,
            7 => RuntimeTypeDescriptor::Ref,
            8 => {
                let has_nominal = reader.read_u8()? != 0;
                let nominal_id = if has_nominal {
                    Some(reader.read_u32()?)
                } else {
                    None
                };
                let layout_id = reader.read_u32()?;
                RuntimeTypeDescriptor::Object {
                    nominal_id,
                    layout_id,
                }
            }
            9 => RuntimeTypeDescriptor::Array {
                element: Box::new(RuntimeTypeDescriptor::decode(reader)?),
            },
            10 => {
                let count = reader.read_u32()? as usize;
                let mut elements = Vec::with_capacity(count.min(1024));
                for _ in 0..count {
                    elements.push(RuntimeTypeDescriptor::decode(reader)?);
                }
                RuntimeTypeDescriptor::Tuple { elements }
            }
            11 => RuntimeTypeDescriptor::Function {
                signature: FunctionSignatureId(reader.read_u32()?),
            },
            12 => RuntimeTypeDescriptor::Task {
                result: Box::new(RuntimeTypeDescriptor::decode(reader)?),
            },
            other => {
                return Err(DecodeError::Corrupted(format!(
                    "unknown runtime type tag {other}"
                )))
            }
        })
    }
}

/// Why a checker type cannot become a runtime descriptor yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedTypeError {
    pub type_name: &'static str,
    pub reason: &'static str,
}

impl fmt::Display for UnsupportedTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "type '{}' has no runtime descriptor: {}",
            self.type_name, self.reason
        )
    }
}
impl std::error::Error for UnsupportedTypeError {}

/// Convert a checker type into its runtime descriptor.
///
/// Conservative by design (ADR R1/D10): anything the runtime cannot yet
/// describe precisely becomes an explicit error or the boxed `Ref`/`AnyValue`
/// forms — never a silently-wrong precise claim. Enrichment of `Object`
/// descriptors with real nominal/layout ids happens in B2 when codegen gains
/// access to class registries.
pub fn runtime_type_of(
    type_ctx: &TypeContext,
    ty: TypeId,
) -> Result<RuntimeTypeDescriptor, UnsupportedTypeError> {
    const MAX_DEPTH: usize = 32;
    convert(type_ctx, ty, MAX_DEPTH, 0)
}

fn depth_err(name: &'static str) -> UnsupportedTypeError {
    UnsupportedTypeError {
        type_name: name,
        reason: "nesting too deep",
    }
}

fn err(name: &'static str, reason: &'static str) -> UnsupportedTypeError {
    UnsupportedTypeError { type_name: name, reason }
}

fn convert(
    ctx: &TypeContext,
    ty: TypeId,
    max_depth: usize,
    depth: usize,
) -> Result<RuntimeTypeDescriptor, UnsupportedTypeError> {
    if depth > max_depth {
        return Err(depth_err("recursive"));
    }
        let resolved = match ctx.get(ty) {
            Some(t) => t.clone(),
            None => return Err(err("unknown", "type id not present in context")),
        };
        Ok(match resolved {
            Type::Primitive(p) => match p {
                PrimitiveType::Int => RuntimeTypeDescriptor::I32,
                PrimitiveType::Number => RuntimeTypeDescriptor::F64,
                PrimitiveType::Boolean => RuntimeTypeDescriptor::Bool,
                PrimitiveType::String => RuntimeTypeDescriptor::String,
                PrimitiveType::Null => RuntimeTypeDescriptor::Null,
                PrimitiveType::Void => RuntimeTypeDescriptor::Void,
            },
            Type::StringLiteral(_) => RuntimeTypeDescriptor::String,
            Type::NumberLiteral(_) => RuntimeTypeDescriptor::F64,
            Type::BooleanLiteral(_) => RuntimeTypeDescriptor::Bool,
            Type::Array(at) => RuntimeTypeDescriptor::Array {
                element: Box::new(convert(ctx, at.element, max_depth, depth + 1)?),
            },
            Type::Tuple(tt) => {
                let mut elements = Vec::with_capacity(tt.elements.len());
                for el in &tt.elements {
                    elements.push(convert(ctx, *el, max_depth, depth + 1)?);
                }
                RuntimeTypeDescriptor::Tuple { elements }
            }
            Type::Task(task) => RuntimeTypeDescriptor::Task {
                result: Box::new(convert(ctx, task.result, max_depth, depth + 1)?),
            },
            // Managed references without precise layout claims (v1). See
            // module docs: enriched Object descriptors arrive with B2.
            Type::Class(_)
            | Type::Map(_)
            | Type::Set(_)
            | Type::Channel(_)
            | Type::Mutex
            | Type::Buffer
            | Type::Date
            | Type::RegExp
            | Type::Json
            | Type::Object(_)
            | Type::Interface(_) => RuntimeTypeDescriptor::Ref,
            // Dynamic forms. Strict mode rejects `any` upstream; these remain
            // for Js/NodeCompat signatures.
            Type::Any | Type::Unknown | Type::JSObject => RuntimeTypeDescriptor::AnyValue,
            // Named references that resolve to concrete types recurse; those
            // that do not are treated as opaque references.
            Type::Reference(r) => {
                let name = r.name.as_str();
                match ctx.resolve_named_type(name) {
                    Ok(concrete) if concrete != ty => convert(ctx, concrete, max_depth, depth + 1)?,
                    _ => RuntimeTypeDescriptor::Ref,
                }
            }
            Type::Union(u) => {
                // All members must agree after conversion; otherwise reject.
                let mut members = u.members.iter().map(|m| convert(ctx, *m, max_depth, depth + 1));
                let first = match members.next() {
                    Some(Ok(m)) => m,
                    Some(Err(e)) => return Err(e),
                    None => return Err(err("union", "empty union")),
                };
                for rest in members {
                    if rest? != first {
                        return Err(err("union", "members map to different runtime types"));
                    }
                }
                first
            }
            Type::Function(_) => {
                // Signature interning lands with B3; until then function-typed
                // values are boxed references.
                RuntimeTypeDescriptor::Ref
            }
            Type::TypeVar(_) => return Err(err("TypeVar", "generic substitution incomplete")),
            Type::Generic(_) => return Err(err("Generic", "generic substitution incomplete")),
            Type::Keyof(_) => return Err(err("keyof", "no runtime representation")),
            Type::IndexedAccess(_) => return Err(err("indexed access", "no runtime representation")),
            Type::Never => return Err(err("never", "uninhabited type in value position")),
        })
    }

/// Convert like [`runtime_type_of`], but map unsupported types to the boxed
/// dynamic form instead of failing. Used where the compiler cannot yet prove
/// strict representability (Js/NodeCompat mode); the typed verifier (B5) is
/// the enforcement point for strict contexts.
pub fn runtime_type_of_lenient(type_ctx: &TypeContext, ty: TypeId) -> RuntimeTypeDescriptor {
    runtime_type_of(type_ctx, ty).unwrap_or(RuntimeTypeDescriptor::AnyValue)
}

/// Module-level interners for descriptors and signatures (task B2).
///
/// Primitive descriptors use their implicit ids and are never stored;
/// complex descriptors get `COMPLEX_BASE + index`; signatures get
/// `index + 1` so that id 0 stays reserved for UNTYPED_SIGNATURE_ID.
#[derive(Debug, Default)]
pub struct TypeTables {
    runtime_types: Vec<RuntimeTypeDescriptor>,
    runtime_index: rustc_hash::FxHashMap<RuntimeTypeDescriptor, u32>,
    function_signatures: Vec<FunctionSignature>,
    signature_index: rustc_hash::FxHashMap<FunctionSignature, u32>,
}

impl TypeTables {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern_descriptor(&mut self, d: RuntimeTypeDescriptor) -> u32 {
        if let Some(id) = d.primitive_id() {
            return id.0;
        }
        if let Some(&id) = self.runtime_index.get(&d) {
            return id;
        }
        let id = COMPLEX_BASE + self.runtime_types.len() as u32;
        self.runtime_types.push(d.clone());
        self.runtime_index.insert(d, id);
        id
    }

    pub fn intern_signature(&mut self, sig: FunctionSignature) -> u32 {
        if let Some(&id) = self.signature_index.get(&sig) {
            return id;
        }
        let id = self.function_signatures.len() as u32 + 1; // 0 reserved
        self.function_signatures.push(sig.clone());
        self.signature_index.insert(sig, id);
        id
    }

    /// Move the accumulated tables into a module.
    pub fn install_into(&mut self, module: &mut crate::compiler::bytecode::Module) {
        module.runtime_types = std::mem::take(&mut self.runtime_types);
        module.function_signatures = std::mem::take(&mut self.function_signatures);
    }
}

/// Canonical function signature recorded in bytecode (task R3).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionSignature {
    pub params: Vec<RuntimeTypeDescriptor>,
    pub return_type: RuntimeTypeDescriptor,
    pub rest_element: Option<RuntimeTypeDescriptor>,
    pub flags: FunctionFlags,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FunctionFlags(u16);

impl FunctionFlags {
    pub const IS_ASYNC: FunctionFlags = FunctionFlags(1 << 0);
    pub const HAS_REST: FunctionFlags = FunctionFlags(1 << 1);

    pub fn is_async(&self) -> bool {
        self.0 & Self::IS_ASYNC.0 != 0
    }
    pub fn has_rest(&self) -> bool {
        self.0 & Self::HAS_REST.0 != 0
    }
    pub fn bits(self) -> u16 {
        self.0
    }
    pub fn from_bits(bits: u16) -> Self {
        FunctionFlags(bits)
    }
}

impl std::ops::BitOr for FunctionFlags {
    type Output = FunctionFlags;
    fn bitor(self, rhs: FunctionFlags) -> FunctionFlags {
        FunctionFlags(self.0 | rhs.0)
    }
}

impl FunctionSignature {
    pub fn encode(&self, writer: &mut BytecodeWriter) {
        writer.emit_u16(self.flags.bits());
        writer.emit_u32(self.params.len() as u32);
        for p in &self.params {
            p.encode(writer);
        }
        self.return_type.encode(writer);
        writer.emit_u8(self.rest_element.is_some() as u8);
        if let Some(rest) = &self.rest_element {
            rest.encode(writer);
        }
    }

    pub fn decode(reader: &mut BytecodeReader<'_>) -> Result<Self, DecodeError> {
        let flags = FunctionFlags::from_bits(reader.read_u16()?);
        let param_count = reader.read_u32()? as usize;
        let mut params = Vec::with_capacity(param_count.min(4096));
        for _ in 0..param_count {
            params.push(RuntimeTypeDescriptor::decode(reader)?);
        }
        let return_type = RuntimeTypeDescriptor::decode(reader)?;
        let rest_element = if reader.read_u8()? != 0 {
            Some(RuntimeTypeDescriptor::decode(reader)?)
        } else {
            None
        };
        Ok(Self {
            params,
            return_type,
            rest_element,
            flags,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::types::context::TypeContext;

    #[test]
    fn primitives_round_trip() {
        let cases = [
            RuntimeTypeDescriptor::I32,
            RuntimeTypeDescriptor::F64,
            RuntimeTypeDescriptor::Bool,
            RuntimeTypeDescriptor::String,
            RuntimeTypeDescriptor::Null,
            RuntimeTypeDescriptor::Void,
            RuntimeTypeDescriptor::AnyValue,
            RuntimeTypeDescriptor::Ref,
        ];
        for d in cases {
            let mut w = BytecodeWriter::new();
            d.encode(&mut w);
            let bytes = w.into_bytes();
            let mut r = BytecodeReader::new(&bytes);
            assert_eq!(RuntimeTypeDescriptor::decode(&mut r).unwrap(), d);
            assert_eq!(bytes.len(), 1, "primitives encode as one byte");
        }
    }

    #[test]
    fn nested_descriptors_round_trip() {
        let d = RuntimeTypeDescriptor::Task {
            result: Box::new(RuntimeTypeDescriptor::Array {
                element: Box::new(RuntimeTypeDescriptor::Tuple {
                    elements: vec![
                        RuntimeTypeDescriptor::I32,
                        RuntimeTypeDescriptor::Object {
                            nominal_id: Some(7),
                            layout_id: 42,
                        },
                    ],
                }),
            }),
        };
        let mut w = BytecodeWriter::new();
        d.encode(&mut w);
        let bytes = w.into_bytes();
        let mut r = BytecodeReader::new(&bytes);
        assert_eq!(RuntimeTypeDescriptor::decode(&mut r).unwrap(), d);
    }

    #[test]
    fn signature_round_trip() {
        let sig = FunctionSignature {
            params: vec![RuntimeTypeDescriptor::I32, RuntimeTypeDescriptor::F64],
            return_type: RuntimeTypeDescriptor::Bool,
            rest_element: Some(RuntimeTypeDescriptor::String),
            flags: FunctionFlags::IS_ASYNC | FunctionFlags::HAS_REST,
        };
        let mut w = BytecodeWriter::new();
        sig.encode(&mut w);
        let bytes = w.into_bytes();
        let mut r = BytecodeReader::new(&bytes);
        assert_eq!(FunctionSignature::decode(&mut r).unwrap(), sig);
    }

    #[test]
    fn checker_types_convert_per_contract() {
        let mut ctx = TypeContext::new();
        let int_ty = ctx.int_type();
        let num_ty = ctx.number_type();
        let str_ty = ctx.string_type();

        assert_eq!(runtime_type_of(&ctx, int_ty).unwrap(), RuntimeTypeDescriptor::I32);
        assert_eq!(runtime_type_of(&ctx, num_ty).unwrap(), RuntimeTypeDescriptor::F64);
        assert_eq!(runtime_type_of(&ctx, str_ty).unwrap(), RuntimeTypeDescriptor::String);

        // int[] -> Array<I32>
        let arr = ctx.intern(Type::Array(crate::parser::types::ty::ArrayType {
            element: int_ty,
        }));
        assert_eq!(
            runtime_type_of(&ctx, arr).unwrap(),
            RuntimeTypeDescriptor::Array {
                element: Box::new(RuntimeTypeDescriptor::I32)
            }
        );

        // Task<string> -> Task<String>
        let task = ctx.intern(Type::Task(crate::parser::types::ty::TaskType {
            result: str_ty,
        }));
        assert_eq!(
            runtime_type_of(&ctx, task).unwrap(),
            RuntimeTypeDescriptor::Task {
                result: Box::new(RuntimeTypeDescriptor::String)
            }
        );

        // heterogeneous union is rejected, not silently widened
        let union = ctx.intern(Type::Union(crate::parser::types::ty::UnionType {
            members: vec![int_ty, str_ty],
            discriminant: None,
            is_bare: false,
            internal_union: None,
        }));
        assert!(runtime_type_of(&ctx, union).is_err());
    }
}
