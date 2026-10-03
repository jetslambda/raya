//! JIT capability table (plan task S0)
//!
//! Single source of truth for what the JIT may compile.
//!
//! Three questions are answered here, and nowhere else:
//!
//! 1. [`jit_support`] — how would this opcode be compiled?
//! 2. [`opcode_supported_for_jit`] — may heuristic candidate selection pick a
//!    function containing this opcode?
//! 3. [`produces_incorrect_native_results`] — does the current native lowering
//!    of this opcode produce *wrong results*? These are hard-rejected by the
//!    lifter itself, regardless of how compilation was requested (prewarm,
//!    jit_hints, AOT adapter).
//!
//! Invariants enforced by tests:
//!
//! - Every classification is explicit. There is no default "supported".
//! - An opcode never appears in two classifications.
//! - CI fails when a new opcode is added without a deliberate decision
//!   (the wildcard arm maps to [`JitSupport::Rejected`], and the known-wrong
//!   list must be updated explicitly).

use crate::compiler::bytecode::Opcode;

/// How the JIT handles an opcode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JitSupport {
    /// Lowered directly to native instructions with exact semantics.
    NativeExact,
    /// Lowered through a runtime helper whose semantics are verified against
    /// the interpreter.
    HelperExact,
    /// Lowered as an exit to the interpreter at this point. Exact, but not
    /// selected for compilation on its own merits.
    InterpreterBoundary,
    /// Not compilable yet. Functions containing this opcode are excluded from
    /// candidate selection.
    Rejected,
}

/// Classification for every opcode. Each opcode appears in exactly one arm;
/// unknown opcodes fall through to [`JitSupport::Rejected`].
pub fn jit_support(opcode: Opcode) -> JitSupport {
    match opcode {
        // ===== Stack manipulation =====
        Opcode::Nop
        | Opcode::Pop
        | Opcode::Dup
        | Opcode::Swap => JitSupport::NativeExact,

        // ===== Constants =====
        Opcode::ConstNull
        | Opcode::ConstTrue
        | Opcode::ConstFalse
        | Opcode::ConstI32
        | Opcode::ConstF64 => JitSupport::NativeExact,
        Opcode::ConstStr | Opcode::LoadConst => JitSupport::HelperExact,

        // ===== Locals =====
        Opcode::LoadLocal
        | Opcode::StoreLocal
        | Opcode::LoadLocal0
        | Opcode::LoadLocal1
        | Opcode::StoreLocal0
        | Opcode::StoreLocal1 => JitSupport::NativeExact,

        // ===== Integer arithmetic =====
        // Note: IPow is deliberately absent. Its previous lowering multiplied
        // the operands (S1); it is rejected until an exact implementation exists.
        Opcode::Iadd
        | Opcode::Isub
        | Opcode::Imul
        | Opcode::Ineg
        | Opcode::Ishl
        | Opcode::Ishr
        | Opcode::Iushr
        | Opcode::Iand
        | Opcode::Ior
        | Opcode::Ixor
        | Opcode::Inot => JitSupport::NativeExact,

        // ===== Float arithmetic =====
        // Note: FPow/FMod deliberately absent (S1).
        Opcode::Fadd
        | Opcode::Fsub
        | Opcode::Fmul
        | Opcode::Fdiv
        | Opcode::Fneg => JitSupport::NativeExact,

        // ===== Integer/float comparison =====
        Opcode::Ieq
        | Opcode::Ine
        | Opcode::Ilt
        | Opcode::Ile
        | Opcode::Igt
        | Opcode::Ige
        | Opcode::Feq
        | Opcode::Fne
        | Opcode::Flt
        | Opcode::Fle
        | Opcode::Fgt
        | Opcode::Fge => JitSupport::NativeExact,

        // ===== Logical =====
        Opcode::Not | Opcode::And | Opcode::Or => JitSupport::NativeExact,

        // ===== Control flow =====
        Opcode::Jmp
        | Opcode::JmpIfTrue
        | Opcode::JmpIfFalse
        | Opcode::JmpIfNull
        | Opcode::JmpIfNotNull
        | Opcode::Return
        | Opcode::ReturnVoid => JitSupport::NativeExact,

        // ===== Strings and generic equality =====
        Opcode::Slen
        | Opcode::ToString
        | Opcode::Sconcat
        | Opcode::Seq
        | Opcode::Sne
        | Opcode::Slt
        | Opcode::Sle
        | Opcode::Sgt
        | Opcode::Sge
        | Opcode::Eq
        | Opcode::Ne
        | Opcode::StrictEq
        | Opcode::StrictNe => JitSupport::HelperExact,

        // ===== Objects (helper-backed) =====
        //
        // Only opcodes whose interpreter handler is a pure leaf read of the
        // object belong here. `NewType` through `ImplementsShape` qualify: they
        // allocate, test nominality, or resolve a shape, and their helpers do the
        // same thing.
        //
        // The field-access opcodes deliberately do NOT. Their interpreter
        // handlers do things no leaf helper can do:
        //
        //  * `Object.defineProperty` installs a `get`/`set` descriptor as
        //    `__node_compat_descriptor` metadata, and the field handlers consult
        //    it. `LoadFieldExact`/`LoadFieldShape` *invoke the getter as a
        //    callable frame* (`callable_frame_for_value`), and `StoreFieldShape`
        //    invokes a setter plus the writability checks. A helper that only
        //    does `object.get_field(slot)` returns the raw field where the
        //    interpreter returns the accessor's value.
        //  * every field handler calls `unwrap_proxy_target` first, and can
        //    produce a `BoundMethod` for a method slot.
        //
        // The helpers do neither, so these opcodes are demoted to `Rejected`
        // until the helpers guard on both conditions and fall back. They were
        // `HelperExact` and wired into the Cranelift lowering, which made this a
        // live miscompilation rather than a latent one. See
        // /workspace/specs/2026-10-03-raya-d4-fixed-layout-objects.md, Gap 4
        // and Gap 6.
        Opcode::NewType
        | Opcode::IsNominal
        | Opcode::CastNominal
        | Opcode::CastShape
        | Opcode::ImplementsShape => JitSupport::HelperExact,
        Opcode::StoreFieldExact => JitSupport::InterpreterBoundary,
        Opcode::LoadFieldExact
        | Opcode::OptionalFieldExact
        | Opcode::LoadFieldShape
        | Opcode::OptionalFieldShape
        | Opcode::StoreFieldShape => JitSupport::Rejected,

        // ===== Calls (helper-backed) =====
        Opcode::Call
        | Opcode::CallMethodExact
        | Opcode::OptionalCallMethodExact
        | Opcode::CallMethodShape
        | Opcode::OptionalCallMethodShape
        | Opcode::ConstructType
        | Opcode::CallConstructor
        | Opcode::CallSuper
        | Opcode::CallStatic
        | Opcode::NativeCall
        | Opcode::ModuleNativeCall => JitSupport::HelperExact,

        // Everything else — arrays, closures, globals/statics, exceptions,
        // concurrency, dynamic keyed access, rest args, remaining casts,
        // remaining string ops — is not compilable yet.
        _ => JitSupport::Rejected,
    }
}

/// May candidate selection choose a function containing this opcode?
pub fn opcode_supported_for_jit(opcode: Opcode) -> bool {
    matches!(
        jit_support(opcode),
        JitSupport::NativeExact | JitSupport::HelperExact
    )
}

/// Opcodes whose *current* native lowering produces incorrect results.
///
/// These are hard-rejected inside the lifter itself (not just during
/// candidate selection) because executing them natively silently yields
/// wrong values, corrupts JIT stack state, or panics where the interpreter
/// would raise a catchable error:
///
/// - `IPow`/`FPow`: lowered as multiplication (S1)
/// - `FMod`: returned the left operand unchanged (S1)
/// - `BindMethod`: emitted nothing, corrupting the lifted stack model
/// - `GetArgCount`/`LoadArgLocal`: no-op / constant zero (S2)
/// - `Try`/`Rethrow`/`Throw`: placeholder handler installation; throw and
///   deopt helpers panic instead of propagating
///   integer division/remainder currently lack catchable zero-divisor paths
pub fn produces_incorrect_native_results(opcode: Opcode) -> bool {
    matches!(
        opcode,
        Opcode::Ipow
            | Opcode::Idiv
            | Opcode::Imod
            | Opcode::Fpow
            | Opcode::Fmod
            | Opcode::BindMethod
            | Opcode::GetArgCount
            | Opcode::LoadArgLocal
            | Opcode::Try
            | Opcode::Rethrow
            | Opcode::Throw
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn division_and_remainder_are_rejected_until_error_paths_are_exact() {
        assert_eq!(jit_support(Opcode::Idiv), JitSupport::Rejected);
        assert_eq!(jit_support(Opcode::Imod), JitSupport::Rejected);
        assert!(produces_incorrect_native_results(Opcode::Idiv));
        assert!(produces_incorrect_native_results(Opcode::Imod));
    }

    #[test]
    fn arithmetic_core_is_native_exact() {
        for op in [
            Opcode::Iadd,
            Opcode::Isub,
            Opcode::Imul,
            Opcode::Ineg,
            Opcode::Fadd,
            Opcode::Fsub,
            Opcode::Fmul,
            Opcode::Fdiv,
            Opcode::Fneg,
        ] {
            assert_eq!(jit_support(op), JitSupport::NativeExact, "{op:?}");
        }
    }

    #[test]
    fn known_wrong_lowerings_are_rejected() {
        for op in [
            Opcode::Ipow,
            Opcode::Fpow,
            Opcode::Fmod,
            Opcode::BindMethod,
            Opcode::GetArgCount,
            Opcode::LoadArgLocal,
            Opcode::Try,
            Opcode::Rethrow,
            Opcode::Throw,
        ] {
            assert!(
                produces_incorrect_native_results(op),
                "{op:?} must stay on the known-wrong list"
            );
            assert!(
                !opcode_supported_for_jit(op),
                "{op:?} must not be selectable"
            );
        }
    }

    #[test]
    fn accessor_and_proxy_field_opcodes_are_rejected_until_exact() {
        // Fail-closed posture (D4.3). These five were `HelperExact` and wired
        // into the Cranelift lowering, but their interpreter handlers unwrap a
        // proxy receiver and consult `__node_compat_descriptor` accessors,
        // invoking a user getter or setter as a frame. The field helpers do
        // neither, so promoting them produced a live divergence between the two
        // engines. Each stays `Rejected` until the helper can detect both
        // conditions and return the fallback sentinel so the interpreter handles
        // the frame.
        for op in [
            Opcode::LoadFieldExact,
            Opcode::OptionalFieldExact,
            Opcode::LoadFieldShape,
            Opcode::OptionalFieldShape,
            Opcode::StoreFieldShape,
        ] {
            assert_eq!(jit_support(op), JitSupport::Rejected, "{op:?}");
            assert!(!opcode_supported_for_jit(op), "{op:?} must not be selectable");
        }
    }

    #[test]
    fn nominal_and_shape_predicates_stay_promoted() {
        // The complement of the test above: these handlers and helpers agree, so
        // the demotion must not silently swallow the whole object family.
        for op in [
            Opcode::NewType,
            Opcode::IsNominal,
            Opcode::CastNominal,
            Opcode::CastShape,
            Opcode::ImplementsShape,
        ] {
            assert_eq!(jit_support(op), JitSupport::HelperExact, "{op:?}");
            assert!(opcode_supported_for_jit(op), "{op:?} should stay selectable");
        }
    }

    #[test]
    fn array_opcodes_are_rejected_until_exact() {
        // Fail-closed posture (D4.2): no array opcode is JIT-selectable until
        // its native/helper path is proven exact and covered by JIT-active
        // differential tests. InitArray in particular stays rejected.
        for op in [
            Opcode::NewArray,
            Opcode::LoadElem,
            Opcode::StoreElem,
            Opcode::ArrayLen,
            Opcode::ArrayPush,
            Opcode::ArrayPop,
            Opcode::ArrayLiteral,
            Opcode::InitArray,
        ] {
            assert_eq!(jit_support(op), JitSupport::Rejected, "{op:?}");
            assert!(!opcode_supported_for_jit(op), "{op:?} must not be selectable");
        }
    }

    #[test]
    fn control_flow_and_locals_are_selectable() {
        for op in [
            Opcode::Jmp,
            Opcode::JmpIfTrue,
            Opcode::JmpIfFalse,
            Opcode::Return,
            Opcode::ReturnVoid,
            Opcode::LoadLocal,
            Opcode::StoreLocal,
        ] {
            assert!(opcode_supported_for_jit(op), "{op:?}");
        }
    }

    #[test]
    fn helper_backed_calls_are_selectable() {
        assert!(opcode_supported_for_jit(Opcode::Call));
        assert!(opcode_supported_for_jit(Opcode::NativeCall));
        assert!(opcode_supported_for_jit(Opcode::ModuleNativeCall));
    }

    #[test]
    fn concurrency_is_never_selectable() {
        for op in [
            Opcode::Spawn,
            Opcode::SpawnClosure,
            Opcode::Await,
            Opcode::Sleep,
            Opcode::MutexLock,
            Opcode::MutexUnlock,
            Opcode::NewChannel,
        ] {
            assert!(!opcode_supported_for_jit(op), "{op:?}");
        }
    }

    #[test]
    fn store_field_exact_is_boundary_not_selectable() {
        assert_eq!(jit_support(Opcode::StoreFieldExact), JitSupport::InterpreterBoundary);
        assert!(!opcode_supported_for_jit(Opcode::StoreFieldExact));
    }
}
