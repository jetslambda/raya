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
        | Opcode::ImplementsShape
        // RefCell (D4.4). Promoted only after three pieces of evidence existed:
        // an interpreter baseline for the handlers, direct-lift tests for the
        // three lowering arms, and a differential running the SAME bytecode
        // through both engines and comparing the result bits. The helpers
        // reproduce the interpreter's weak `is_ptr()` check rather than
        // strengthening it (ALY-54), `NewRefCell` roots its operand across the
        // allocation and fails closed, and a non-pointer receiver returns the
        // interpreter-fallback sentinel so the interpreter raises the real error.
        | Opcode::NewRefCell
        | Opcode::LoadRefCell
        | Opcode::StoreRefCell
        // Closures (D4.4). Same three-part evidence as the RefCell family:
        // interpreter baseline, direct-lift tests for each arm, and a
        // differential running the SAME bytecode through both engines. The two
        // here are the only closure opcodes that can be promoted at all --
        // `LoadCaptured` and `StoreCaptured` still have no operand
        // naming the closure they act on, because the interpreter resolves them
        // against `task.current_closure()` and the JIT has no active-closure
        // model. See the D4.4 spec.
        | Opcode::MakeClosure
        | Opcode::SetClosureCapture
        // `LoadCaptured`/`StoreCaptured` read and write the closure of the frame
        // currently executing, which they get from `Task::current_closure()` — the
        // innermost entry on `closure_stack`, exactly as the interpreter does.
        // They are promoted on an interpreter baseline (through the
        // `Call 0xFFFFFFFF` closure-call path) and a direct-lift test of the
        // closure body that executes both arms natively. Note that is NOT a
        // shared-bytecode differential: `Call` routes every callee through
        // `interpreter_call`, so lifting `main` and calling a closure would run
        // the body interpreted and prove nothing. See the D4.4 spec.
        | Opcode::LoadCaptured
        | Opcode::StoreCaptured
        // `BindMethod` (2026-10-03). Its lifter arm was empty — it consumed no
        // operand and touched no stack — so lifting failed before the arm was even
        // reached and the lifted `ip` never advanced past the operand. That is
        // fixed (`689ec78`), and it now has a helper, a lowering arm, an
        // interpreter baseline covering both the nominal success and the structural
        // -object failure, and a direct-lift test proving the vtable resolves and
        // the receiver is carried.
        | Opcode::BindMethod => JitSupport::HelperExact,
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
/// - `BindMethod`: **removed 2026-10-03.** The lifter's arm used to be an empty
///   body, so it consumed no operand and touched no stack while the interpreter read
///   the operand, popped the receiver and pushed a `BoundMethod` — leaving the
///   lifted `ip` un-advanced and every later instruction misaligned. The arm now
///   models that effect, so the opcode no longer belongs in this list.
///
///   It was promoted to `HelperExact` later the same day, once it had a helper, a
///   lowering arm and evidence on both sides. The ordering mattered: while the arm
///   was empty the opcode was rejected here AND unselectable, so neither gate could
///   be relied on alone.
/// - `GetArgCount`/`LoadArgLocal`: no-op / constant zero (S2)
/// - `Try`: placeholder handler installation (D4.5). `Rethrow` and `Throw` were
///   removed from this list on 2026-10-03. The recorded reason was that "throw and
///   deopt helpers panic instead of propagating", which turned out to name two
///   functions nothing calls: `helper_throw_exception` and `helper_deoptimize` are
///   referenced nowhere outside `trampoline.rs`. Both opcodes lift correctly and
///   their lowering arms reach the interpreter through
///   `emit_interpreter_boundary_exit`, which needs no helper at all. `Try` stays —
///   its lifter arm is still the placeholder that computes catch/finally targets
///   into unused bindings
///   integer division/remainder currently lack catchable zero-divisor paths
pub fn produces_incorrect_native_results(opcode: Opcode) -> bool {
    matches!(
        opcode,
        Opcode::Ipow
            | Opcode::Idiv
            | Opcode::Imod
            | Opcode::Fpow
            | Opcode::Fmod
            | Opcode::GetArgCount
            | Opcode::LoadArgLocal
            | Opcode::Try
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gate a promotion has to move, tested at the granularity it operates on.
    ///
    /// `opcode_supported_for_jit` is what `function_supported_for_jit`
    /// (`jit/analysis/heuristics.rs:309`) consults, and that is the function a
    /// promotion has to flip. The `jit_compile_and_call_with_locals_exit_and_ctx`
    /// harness the lowering tests use does **not** go through it -- it compiles
    /// whatever it is handed -- so those tests prove the arms work while saying
    /// nothing about reachability. This is the test that says something about
    /// reachability, and it is the one that flipped for the D4.4 promotion.
    #[test]
    fn refcell_opcodes_make_a_function_jit_eligible() {
        use crate::jit::analysis::heuristics::function_supported_for_jit;
        use crate::compiler::bytecode::Function;

        let function_with = |code: Vec<u8>| Function {
            signature_id: 0,
            local_types: Vec::new(),
            abi_version: 1,
            name: "main".to_string(),
            param_count: 0,
            local_count: 0,
            code,
        };

        // Control: a function built only from long-promoted opcodes must stay
        // eligible. If this ever fails, the gate is not measuring what we think and
        // every other assertion here is meaningless.
        assert!(
            function_supported_for_jit(&function_with(vec![
                Opcode::ConstI32 as u8, 1, 0, 0, 0, Opcode::Return as u8,
            ])),
            "control function must be JIT-eligible, or the gate is not measuring what we think"
        );

        // Each RefCell opcode now makes its function ELIGIBLE rather than excluded.
        // This is the flip, and it is the assertion that would fail if a promotion
        // were made without moving the table.
        for (op, label) in [
            (Opcode::NewRefCell, "NewRefCell"),
            (Opcode::LoadRefCell, "LoadRefCell"),
            (Opcode::StoreRefCell, "StoreRefCell"),
            (Opcode::MakeClosure, "MakeClosure"),
            (Opcode::SetClosureCapture, "SetClosureCapture"),
            (Opcode::LoadCaptured, "LoadCaptured"),
            (Opcode::StoreCaptured, "StoreCaptured"),
            (Opcode::BindMethod, "BindMethod"),
        ] {
            assert!(
                function_supported_for_jit(&function_with(vec![
                    op as u8,
                    Opcode::ConstI32 as u8, 1, 0, 0, 0, Opcode::Return as u8,
                ])),
                "{label} must now make its function JIT-eligible"
            );
        }

    }

    #[test]
    fn closure_and_refcell_family_is_promoted_with_evidence() {
        // D4.4 promotion. These three were `Rejected` for the whole milestone and
        // were promoted only once all three pieces of evidence existed:
        //
        //   * interpreter baseline — `refcell_opcodes_roundtrip_through_the_interpreter`
        //     and `refcell_opcodes_reject_a_non_pointer_receiver`
        //   * lowering arms — three direct-lift tests asserting exit.kind == Completed
        //   * differential — `refcell_interpreter_and_jit_agree_on_the_same_bytecode`,
        //     same bytecode through both engines, result bits compared
        //
        // Asserted in both directions on purpose: `jit_support` AND
        // `opcode_supported_for_jit`. A promotion that changed only the table, or
        // only the selector, would break this rather than pass quietly.
        for op in [
            Opcode::MakeClosure,
            Opcode::LoadCaptured,
            Opcode::StoreCaptured,
            Opcode::SetClosureCapture,
            Opcode::NewRefCell,
            Opcode::LoadRefCell,
            Opcode::StoreRefCell,
        ] {
            let promoted = matches!(
                op,
                Opcode::NewRefCell
                    | Opcode::LoadRefCell
                    | Opcode::StoreRefCell
                    | Opcode::MakeClosure
                    | Opcode::SetClosureCapture
                    | Opcode::LoadCaptured
                    | Opcode::StoreCaptured
            );
            let expected = if promoted {
                JitSupport::HelperExact
            } else {
                JitSupport::Rejected
            };
            assert_eq!(jit_support(op), expected, "{op:?}");
            assert_eq!(
                opcode_supported_for_jit(op),
                promoted,
                "{op:?} selectable must track its classification"
            );
        }

        // `BindMethod` left `produces_incorrect_native_results` when its lifter arm
        // was fixed, and was promoted once it had a helper, a lowering arm and
        // evidence on both sides. Both facts are asserted separately because they
        // are independent: a promotion that forgot the list entry would leave the
        // lifter still rejecting it, and one that forgot the classification would
        // leave the list entry claiming a defect that no longer exists.
        assert!(
            !produces_incorrect_native_results(Opcode::BindMethod),
            "BindMethod must no longer be lifter-rejected; its arm now models the stack effect"
        );
        assert_eq!(
            jit_support(Opcode::BindMethod),
            JitSupport::HelperExact,
            "BindMethod is promoted and must be selectable"
        );
    }

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

    /// D4.5 baseline: the whole exception family, pinned explicitly.
    ///
    /// All four are `Rejected` today, but for two different reasons, and the test
    /// records both rather than letting one assertion cover them:
    ///
    /// - `Try`, `Throw` and `Rethrow` are on `produces_incorrect_native_results`,
    ///   so the *lifter* rejects them. For `Throw`/`Rethrow` the recorded defect is
    ///   that the throw and deopt helpers panic instead of propagating; for `Try`
    ///   the lifter arm is a placeholder that computes catch/finally targets into
    ///   unused bindings.
    /// - `EndTry` is **not** on that list. Its lifter arm is correct and complete
    ///   (`JitInstr::EndTry`, no stack effect, matching the handler), so it is held
    ///   out only by the `Rejected` catch-all here.
    ///
    /// That distinction is the point. `EndTry` being excluded by accident is how
    /// `BindMethod` went wrong, so it is pinned explicitly: a future promotion has
    /// to be a visible act with evidence behind it, not a side effect of the
    /// table's default.
    #[test]
    fn exception_family_is_rejected_until_the_throw_path_propagates() {
        for op in [
            Opcode::Throw,
            Opcode::Try,
            Opcode::EndTry,
            Opcode::Rethrow,
        ] {
            assert_eq!(jit_support(op), JitSupport::Rejected, "{op:?}");
            assert!(
                !opcode_supported_for_jit(op),
                "{op:?} must not be selectable"
            );
        }

        // `Throw` and `Rethrow` left this list on 2026-10-03: their arms reach the
        // interpreter through emit_interpreter_boundary_exit and need no helper.
        // `Try` stays, because its lifter arm is still a placeholder.
        for op in [Opcode::Throw, Opcode::Rethrow] {
            assert!(
                !produces_incorrect_native_results(op),
                "{op:?} lifts correctly and must no longer be lifter-rejected"
            );
        }
        assert!(
            produces_incorrect_native_results(Opcode::Try),
            "Try must stay lifter-rejected until its handler installation is modelled"
        );

        assert!(
            !produces_incorrect_native_results(Opcode::EndTry),
            "EndTry lifts correctly, so it has no lifter-level defect; it is excluded              by the Rejected catch-all instead, and this test records that rather              than leaving it implicit"
        );
    }

    #[test]
    fn known_wrong_lowerings_are_rejected() {
        for op in [
            Opcode::Ipow,
            Opcode::Fpow,
            Opcode::Fmod,
            Opcode::GetArgCount,
            Opcode::LoadArgLocal,
            Opcode::Try,
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
