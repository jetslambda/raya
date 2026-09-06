//! Code Generation from IR to Bytecode
//!
//! This module transforms the optimized IR into bytecode for the Raya VM.
//!
//! # Pipeline
//!
//! ```text
//! IR Module → IrCodeGenerator → Bytecode Module
//! ```
//!
//! # Phases
//!
//! 1. **Basic Emission**: Constants, locals, binary/unary ops
//! 2. **Control Flow**: Branches, loops, switches
//! 3. **Classes/Objects**: Field access, method calls, constructors
//! 4. **Closures**: Captured variables
//! 5. **Optimizations**: String comparison optimization

mod context;
#[allow(dead_code)]
mod control;
pub mod emit;

pub use context::IrCodeGenerator;

use crate::compiler::bytecode::Module;
use crate::compiler::error::CompileResult;
use crate::compiler::ir::IrModule;

/// Generate bytecode from an IR module
///
/// Reflection metadata (class/field/method names) is always included
/// to support runtime introspection via the Reflect API.
///
/// When `emit_sourcemap` is true, the generated module includes debug info
/// with bytecode offset → source location mappings.
pub fn generate(ir_module: &IrModule, emit_sourcemap: bool) -> CompileResult<Module> {
    generate_with_types(ir_module, emit_sourcemap, None)
}

/// Generate bytecode with typed signature derivation from checker types.
///
/// When `type_ctx` is provided, every function records a canonical signature
/// and per-slot runtime descriptors into the module's interned tables (B2).
pub fn generate_with_types(
    ir_module: &IrModule,
    emit_sourcemap: bool,
    type_ctx: Option<&crate::parser::types::context::TypeContext>,
) -> CompileResult<Module> {
    let mut generator = IrCodeGenerator::new(&ir_module.name);
    generator.set_emit_sourcemap(emit_sourcemap);
    if let Some(type_ctx) = type_ctx {
        generator.set_type_ctx(type_ctx);
    }
    generator.generate(ir_module)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ir::block::Terminator;
    use crate::compiler::ir::instr::{BinaryOp, IrInstr};
    use crate::compiler::ir::value::{IrConstant, IrValue, Register, RegisterId};
    use crate::compiler::ir::{BasicBlock, BasicBlockId, IrFunction, IrModule};
    use crate::parser::TypeId;

    fn make_reg(id: u32, ty: u32) -> Register {
        Register::new(RegisterId::new(id), TypeId::new(ty))
    }

    #[test]
    fn test_generate_empty_module() {
        let mut module = IrModule::new("test");

        // Add a simple main function that returns null
        let mut main = IrFunction::new("main", vec![], TypeId::new(0));
        let mut entry = BasicBlock::new(BasicBlockId(0));
        entry.set_terminator(Terminator::Return(None));
        main.add_block(entry);
        module.add_function(main);

        let result = generate(&module, false);
        assert!(result.is_ok());

        let bytecode = result.unwrap();
        assert_eq!(bytecode.functions.len(), 1);
        assert_eq!(bytecode.functions[0].name, "main");
    }

    #[test]
    fn b2_typed_signatures_persist_into_bytecode() {
        use crate::compiler::bytecode::types::{
            runtime_type_of_lenient, RuntimeTypeDescriptor, CURRENT_ABI_VERSION,
            UNTYPED_SIGNATURE_ID,
        };
        use crate::parser::types::context::TypeContext;

        let mut type_ctx = TypeContext::new();
        let int_ty = type_ctx.int_type();

        // add(a: int, b: int): int { return a + b }  (IR level)
        let mut module = IrModule::new("typed_math");
        let params = vec![make_reg(0, int_ty.as_u32()), make_reg(1, int_ty.as_u32())];
        let mut func = IrFunction::new("add", params.clone(), int_ty);
        let mut entry = BasicBlock::new(BasicBlockId(0));
        entry.add_instr(IrInstr::Assign {
            dest: make_reg(2, int_ty.as_u32()),
            value: IrValue::Constant(IrConstant::I32(7)),
        });
        entry.set_terminator(Terminator::Return(Some(make_reg(2, int_ty.as_u32()))));
        func.add_block(entry);
        module.add_function(func);

        let bytecode = generate_with_types(&module, false, Some(&type_ctx)).unwrap();

        let f = &bytecode.functions[0];
        assert_ne!(f.signature_id, UNTYPED_SIGNATURE_ID, "signature recorded");
        assert_eq!(f.abi_version, CURRENT_ABI_VERSION);

        let sig = &bytecode.function_signatures[f.signature_id as usize - 1];
        let i32_desc = runtime_type_of_lenient(&type_ctx, int_ty);
        assert_eq!(i32_desc, RuntimeTypeDescriptor::I32);
        assert_eq!(sig.params.len(), 2);
        assert_eq!(sig.params[0], RuntimeTypeDescriptor::I32);
        assert_eq!(sig.params[1], RuntimeTypeDescriptor::I32);
        assert_eq!(sig.return_type, RuntimeTypeDescriptor::I32);

        // every local slot (params + temp) is typed i32
        assert!(f.local_types.iter().all(|&id| id == 0)); // I32 implicit id
    }

    #[test]
    fn b2_without_type_ctx_stays_untyped() {
        let mut module = IrModule::new("untyped");
        let mut func = IrFunction::new("main", vec![], TypeId::new(0));
        let mut entry = BasicBlock::new(BasicBlockId(0));
        entry.set_terminator(Terminator::Return(None));
        func.add_block(entry);
        module.add_function(func);

        let bytecode = generate(&module, false).unwrap();
        assert_eq!(
            bytecode.functions[0].signature_id,
            crate::compiler::bytecode::types::UNTYPED_SIGNATURE_ID
        );
        assert!(bytecode.function_signatures.is_empty());
    }

    #[test]
    fn test_generate_return_constant() {
        let mut module = IrModule::new("test");

        let mut func = IrFunction::new("answer", vec![], TypeId::new(1));
        let mut entry = BasicBlock::new(BasicBlockId(0));

        // r0 = 42
        let r0 = make_reg(0, 1);
        entry.add_instr(IrInstr::Assign {
            dest: r0.clone(),
            value: IrValue::Constant(IrConstant::I32(42)),
        });
        entry.set_terminator(Terminator::Return(Some(r0)));
        func.add_block(entry);
        module.add_function(func);

        let result = generate(&module, false);
        assert!(result.is_ok());

        let bytecode = result.unwrap();
        assert_eq!(bytecode.functions.len(), 1);
        // Code should contain: CONST_I32 42, RETURN
        assert!(!bytecode.functions[0].code.is_empty());
    }

    #[test]
    fn test_generate_binary_add() {
        let mut module = IrModule::new("test");

        let mut func = IrFunction::new("add", vec![], TypeId::new(1));
        let mut entry = BasicBlock::new(BasicBlockId(0));

        // r0 = 10
        let r0 = make_reg(0, 1);
        entry.add_instr(IrInstr::Assign {
            dest: r0.clone(),
            value: IrValue::Constant(IrConstant::I32(10)),
        });

        // r1 = 20
        let r1 = make_reg(1, 1);
        entry.add_instr(IrInstr::Assign {
            dest: r1.clone(),
            value: IrValue::Constant(IrConstant::I32(20)),
        });

        // r2 = r0 + r1
        let r2 = make_reg(2, 1);
        entry.add_instr(IrInstr::BinaryOp {
            dest: r2.clone(),
            op: BinaryOp::Add,
            left: r0,
            right: r1,
        });

        entry.set_terminator(Terminator::Return(Some(r2)));
        func.add_block(entry);
        module.add_function(func);

        let result = generate(&module, false);
        assert!(result.is_ok());
    }
}
