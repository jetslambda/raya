use crate::compiler::Opcode;
use crate::vm::interpreter::core::value_to_f64;
use crate::vm::interpreter::execution::OpcodeResult;
use crate::vm::interpreter::Interpreter;
use crate::vm::stack::Stack;
use crate::vm::value::Value;
use crate::vm::VmError;

/// Bridge a value to i32 for integer opcodes (task I1).
///
/// Native i32 payloads pass through. Integral f64 values bridge exactly —
/// this keeps programs compiled against earlier checker inference gaps
/// working (loop accumulators observed as f64 at runtime). Anything else
/// (fractional floats, strings, objects) is `None`: callers must raise a
/// type error instead of the historical silent zero.
fn bridge_i32(v: &Value) -> Option<i32> {
    v.as_i32().or_else(|| {
        let f = v.as_f64()?;
        if f.is_finite() && f.fract() == 0.0 && f >= i32::MIN as f64 && f <= i32::MAX as f64 {
            Some(f as i32)
        } else {
            None
        }
    })
}

fn expect_i32(v: &Value, opcode: &str) -> Result<i32, VmError> {
    bridge_i32(v).ok_or_else(|| {
        VmError::RuntimeError(format!(
            "type error: {} requires int operands, got {}",
            opcode,
            v.type_name()
        ))
    })
}

/// Extract an i32 operand or bail out of the opcode handler with a type error.
macro_rules! i32_of {
    ($val:expr, $op:literal) => {
        match expect_i32($val, $op) {
            Ok(x) => x,
            Err(e) => return OpcodeResult::Error(e),
        }
    };
}

impl<'a> Interpreter<'a> {
    pub(in crate::vm::interpreter) fn exec_arithmetic_ops(
        &mut self,
        stack: &mut Stack,
        opcode: Opcode,
    ) -> OpcodeResult {
        match opcode {
            // =========================================================
            // Integer Arithmetic
            // =========================================================
            Opcode::Iadd => {
                let b_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a = i32_of!(&a_val, "Iadd");
                let b = i32_of!(&b_val, "Iadd");
                if let Err(e) = stack.push(Value::i32(a.wrapping_add(b))) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::Isub => {
                let b_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a = i32_of!(&a_val, "Isub");
                let b = i32_of!(&b_val, "Isub");
                if let Err(e) = stack.push(Value::i32(a.wrapping_sub(b))) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::Imul => {
                let b_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                // Try i32 first, fall back to f64→i32 conversion for values that
                // are f64 at runtime due to type inference gaps (e.g., loop accumulators).
                let a = i32_of!(&a_val, "Imul");
                let b = i32_of!(&b_val, "Imul");
                if let Err(e) = stack.push(Value::i32(a.wrapping_mul(b))) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::Idiv => {
                let b_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a = i32_of!(&a_val, "Idiv");
                let b = i32_of!(&b_val, "Idiv");
                if b == 0 {
                    return OpcodeResult::Error(VmError::RuntimeError(
                        "division by zero".to_string(),
                    ));
                }
                if let Err(e) = stack.push(Value::i32(a.wrapping_div(b))) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::Imod => {
                let b_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a = i32_of!(&a_val, "Imod");
                let b = i32_of!(&b_val, "Imod");
                if b == 0 {
                    return OpcodeResult::Error(VmError::RuntimeError(
                        "division by zero".to_string(),
                    ));
                }
                if let Err(e) = stack.push(Value::i32(a.wrapping_rem(b))) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::Ineg => {
                let a_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a = i32_of!(&a_val, "Ineg");
                if let Err(e) = stack.push(Value::i32(a.wrapping_neg())) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::Ipow => {
                let b_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a = i32_of!(&a_val, "Ipow");
                let b = i32_of!(&b_val, "Ipow");
                let result = if b < 0 { 0 } else { a.wrapping_pow(b as u32) };
                if let Err(e) = stack.push(Value::i32(result)) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            // =========================================================
            // Integer Bitwise
            // =========================================================
            Opcode::Ishl => {
                let b = match stack.pop() {
                    Ok(v) => i32_of!(&v, "Ishl"),
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a = match stack.pop() {
                    Ok(v) => i32_of!(&v, "Ishl"),
                    Err(e) => return OpcodeResult::Error(e),
                };
                if let Err(e) = stack.push(Value::i32(a << (b & 31))) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::Ishr => {
                let b = match stack.pop() {
                    Ok(v) => i32_of!(&v, "Ishr"),
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a = match stack.pop() {
                    Ok(v) => i32_of!(&v, "Ishr"),
                    Err(e) => return OpcodeResult::Error(e),
                };
                if let Err(e) = stack.push(Value::i32(a >> (b & 31))) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::Iushr => {
                let b = match stack.pop() {
                    Ok(v) => i32_of!(&v, "Iushr"),
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a = match stack.pop() {
                    Ok(v) => i32_of!(&v, "Iushr"),
                    Err(e) => return OpcodeResult::Error(e),
                };
                if let Err(e) = stack.push(Value::i32(((a as u32) >> (b & 31)) as i32)) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::Iand => {
                let b = match stack.pop() {
                    Ok(v) => i32_of!(&v, "Iand"),
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a = match stack.pop() {
                    Ok(v) => i32_of!(&v, "Iand"),
                    Err(e) => return OpcodeResult::Error(e),
                };
                if let Err(e) = stack.push(Value::i32(a & b)) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::Ior => {
                let b = match stack.pop() {
                    Ok(v) => i32_of!(&v, "Ior"),
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a = match stack.pop() {
                    Ok(v) => i32_of!(&v, "Ior"),
                    Err(e) => return OpcodeResult::Error(e),
                };
                if let Err(e) = stack.push(Value::i32(a | b)) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::Ixor => {
                let b = match stack.pop() {
                    Ok(v) => i32_of!(&v, "Ixor"),
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a = match stack.pop() {
                    Ok(v) => i32_of!(&v, "Ixor"),
                    Err(e) => return OpcodeResult::Error(e),
                };
                if let Err(e) = stack.push(Value::i32(a ^ b)) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::Inot => {
                let a = match stack.pop() {
                    Ok(v) => i32_of!(&v, "Inot"),
                    Err(e) => return OpcodeResult::Error(e),
                };
                if let Err(e) = stack.push(Value::i32(!a)) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            // =========================================================
            // Float Arithmetic
            // =========================================================
            Opcode::Fadd => {
                let b = match stack.pop().and_then(value_to_f64) {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a = match stack.pop().and_then(value_to_f64) {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                if let Err(e) = stack.push(Value::f64(a + b)) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::Fsub => {
                let b = match stack.pop().and_then(value_to_f64) {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a = match stack.pop().and_then(value_to_f64) {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                if let Err(e) = stack.push(Value::f64(a - b)) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::Fmul => {
                let b = match stack.pop().and_then(value_to_f64) {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a = match stack.pop().and_then(value_to_f64) {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                if let Err(e) = stack.push(Value::f64(a * b)) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::Fdiv => {
                let b_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                if b_val.as_i32().is_some() && a_val.as_i32().is_some() && b_val.as_i32() == Some(0) {
                    return OpcodeResult::Error(VmError::RuntimeError(
                        "division by zero".to_string(),
                    ));
                }
                let b = match value_to_f64(b_val) {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a = match value_to_f64(a_val) {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                if let Err(e) = stack.push(Value::f64(a / b)) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::Fneg => {
                let a = match stack.pop().and_then(value_to_f64) {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                if let Err(e) = stack.push(Value::f64(-a)) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::Fpow => {
                let b = match stack.pop().and_then(value_to_f64) {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a = match stack.pop().and_then(value_to_f64) {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                if let Err(e) = stack.push(Value::f64(a.powf(b))) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            Opcode::Fmod => {
                let b_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a_val = match stack.pop() {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                if b_val.as_i32().is_some() && a_val.as_i32().is_some() && b_val.as_i32() == Some(0) {
                    return OpcodeResult::Error(VmError::RuntimeError(
                        "division by zero".to_string(),
                    ));
                }
                let b = match value_to_f64(b_val) {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                let a = match value_to_f64(a_val) {
                    Ok(v) => v,
                    Err(e) => return OpcodeResult::Error(e),
                };
                if let Err(e) = stack.push(Value::f64(a % b)) {
                    return OpcodeResult::Error(e);
                }
                OpcodeResult::Continue
            }

            _ => unreachable!("Not an arithmetic opcode: {:?}", opcode),
        }
    }
}

#[cfg(test)]
mod i3_semantics {
    //! Table-driven arithmetic semantics (task I3). These cases are the
    //! normative reference for JIT differential tests (plan J5).

    use crate::compiler::bytecode::{Function, Module, Opcode};
    use crate::vm::value::Value;
    use crate::vm::Vm;
    use crate::vm::VmError;

    fn run(code: Vec<u8>) -> Result<Value, VmError> {
        run_with_constants(code, Vec::new())
    }

    fn run_with_constants(code: Vec<u8>, strings: Vec<&str>) -> Result<Value, VmError> {
        let mut module = Module::new("i3".to_string());
        for s in strings {
            module.constants.add_string(s.to_string());
        }
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
    fn const_f64(v: f64) -> Vec<u8> {
        let mut c = vec![Opcode::ConstF64 as u8];
        c.extend_from_slice(&v.to_le_bytes());
        c
    }
    fn binop(a: Vec<u8>, b: Vec<u8>, op: Opcode) -> Vec<u8> {
        let mut c = a;
        c.extend(b);
        c.push(op as u8);
        c.push(Opcode::Return as u8);
        c
    }

    #[test]
    fn wrapping_overflow() {
        // i32::MAX + 1 wraps to i32::MIN
        let code = binop(const_i32(i32::MAX), const_i32(1), Opcode::Iadd);
        assert_eq!(run(code).unwrap(), Value::i32(i32::MIN));
    }

    #[test]
    fn int_min_div_negative_one_wraps() {
        let code = binop(const_i32(i32::MIN), const_i32(-1), Opcode::Idiv);
        assert_eq!(run(code).unwrap(), Value::i32(i32::MIN));
    }

    #[test]
    fn int_min_rem_negative_one_is_zero() {
        let code = binop(const_i32(i32::MIN), const_i32(-1), Opcode::Imod);
        assert_eq!(run(code).unwrap(), Value::i32(0));
    }

    #[test]
    fn division_by_zero_raises() {
        for op in [Opcode::Idiv, Opcode::Imod] {
            let code = binop(const_i32(5), const_i32(0), op);
            let err = run(code).unwrap_err();
            assert!(err.to_string().contains("division by zero"), "{err}");
        }
    }

    #[test]
    fn shift_counts_mask_to_five_bits() {
        // 1 << 33 == 1 << 1 == 2
        let shl = binop(const_i32(1), const_i32(33), Opcode::Ishl);
        assert_eq!(run(shl).unwrap(), Value::i32(2));
        // -1 >> 33 == -1 >> 1 == -1 (arithmetic)
        let shr = binop(const_i32(-1), const_i32(33), Opcode::Ishr);
        assert_eq!(run(shr).unwrap(), Value::i32(-1));
        // logical: u32(-1) >> 1
        let ushr = binop(const_i32(-1), const_i32(1), Opcode::Iushr);
        assert_eq!(run(ushr).unwrap(), Value::i32(((-1i32) as u32 >> 1) as i32));
    }

    #[test]
    fn integer_power_table() {
        let cases = [
            (2, 10, 1024),
            (-2, 3, -8),
            (0, 0, 1), // x^0 == 1 by convention
            (5, 0, 1),
            (2, -1, 0), // negative exponent truncates to 0 (pinned; revisit under ADR non-decisions)
        ];
        for (a, b, expected) in cases {
            let code = binop(const_i32(a), const_i32(b), Opcode::Ipow);
            assert_eq!(run(code).unwrap(), Value::i32(expected), "{a}^{b}");
        }
    }

    #[test]
    fn string_operand_raises_instead_of_zero() {
        let mut code = vec![Opcode::ConstStr as u8, 0, 0, 0, 0];
        code.extend(const_i32(2));
        code.push(Opcode::Iadd as u8);
        code.push(Opcode::Return as u8);
        let err = run_with_constants(code, vec!["hello"]).unwrap_err();
        assert!(err.to_string().contains("type error"), "{err}");
    }

    #[test]
    fn fractional_float_operand_raises() {
        let code = binop(const_f64(2.5), const_i32(2), Opcode::Iadd);
        let err = run(code).unwrap_err();
        assert!(err.to_string().contains("type error"), "{err}");
    }

    #[test]
    fn integral_float_bridges_exactly() {
        // Legacy inference-gap bridge: 2.0 + 2 == 4 (exact integral f64).
        let code = binop(const_f64(2.0), const_i32(2), Opcode::Iadd);
        assert_eq!(run(code).unwrap(), Value::i32(4));
    }

    #[test]
    fn float_nan_and_sign_zero_preserved() {
        // NaN propagates through Fadd
        let nan_code = binop(const_f64(f64::NAN), const_f64(1.0), Opcode::Fadd);
        assert!(run(nan_code).unwrap().as_f64().unwrap().is_nan());

        // -1.0 * 0.0 == -0.0
        let neg_zero = binop(const_f64(-1.0), const_f64(0.0), Opcode::Fmul);
        let v = run(neg_zero).unwrap().as_f64().unwrap();
        assert_eq!(v, 0.0);
        assert!(v.is_sign_negative());
    }
}
