//! CFG-based bytecode verification (plan tasks B4/B5)
//!
//! Replaces the linear stack-depth walk with control-flow-graph traversal:
//!
//! - B4: basic blocks from jump targets and terminators, worklist analysis,
//!   per-block entry state, join consistency.
//! - B5: the abstract state carries typed operand slots and local
//!   initialization bits. Transfer rules come from the runtime descriptor
//!   contract (ADR R1). Functions without recorded signatures verify under
//!   the legacy `Any` discipline: depths and initialization are still
//!   checked, types are not.

use super::module::{Function, Module};
use super::opcode::Opcode;
use super::types::{FunctionSignature, RuntimeTypeDescriptor, COMPLEX_BASE, UNTYPED_SIGNATURE_ID};
use super::verify::VerifyError;
use super::verify::{parse_instructions, Instruction};
use std::collections::HashMap;

/// Maximum operand-stack depth (matches the legacy linear verifier).
const MAX_STACK_DEPTH: usize = 1024;

/// Abstract value type on the operand stack (B5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbsType {
    /// Unknown/boxed — compatible with everything at joins.
    Any,
    /// A concrete runtime descriptor id (0-7 implicit primitives; larger ids
    /// index the module's `runtime_types` table).
    Known(u32),
}

impl AbsType {
    fn merge(a: AbsType, b: AbsType) -> AbsType {
        if a == b { a } else { AbsType::Any } // widen at joins
    }
}

/// Abstract execution state at a program point (B5).
#[derive(Debug, Clone, PartialEq)]
pub struct AbstractState {
    /// Operand stack, bottom is index 0.
    pub stack: Vec<AbsType>,
    /// Local initialization bits.
    pub locals_init: Vec<bool>,
}

impl AbstractState {
    fn new(function: &Function) -> Self {
        let mut locals_init = vec![false; function.local_count];
        // Parameters arrive initialized by the calling convention.
        for slot in locals_init.iter_mut().take(function.param_count) {
            *slot = true;
        }
        AbstractState { stack: Vec::new(), locals_init }
    }

    fn push(&mut self, ty: AbsType, offset: usize) -> Result<(), VerifyError> {
        if self.stack.len() >= MAX_STACK_DEPTH {
            return Err(VerifyError::StackOverflow(offset, self.stack.len() as i32));
        }
        self.stack.push(ty);
        Ok(())
    }

    fn pop(&mut self, offset: usize) -> Result<AbsType, VerifyError> {
        self.stack.pop().ok_or(VerifyError::StackUnderflow(offset))
    }

    fn pop_expect(&mut self, expected: AbsType, offset: usize) -> Result<AbsType, VerifyError> {
        let got = self.pop(offset)?;
        match (expected, got) {
            (AbsType::Any, _) | (_, AbsType::Any) => Ok(got),
            (AbsType::Known(x), AbsType::Known(y)) if x == y => Ok(got),
            _ => Err(VerifyError::TypeMismatch {
                offset,
                expected: format!("{expected:?}"),
                actual: format!("{got:?}"),
            }),
        }
    }
}

/// A basic block: contiguous decoded instructions [start, end).
#[derive(Debug, Clone)]
pub struct CfgBlock {
    pub start: usize,
    pub end: usize,
    /// Successor instruction indices.
    pub successors: Vec<usize>,
}

fn instr_size(instr: &Instruction) -> usize {
    1 + instr.operands.len()
}

fn jump_rel_i32(instr: &Instruction) -> Result<i32, VerifyError> {
    if instr.operands.len() < 4 {
        return Err(VerifyError::DecodeError(format!(
            "jump with truncated operand at offset {}",
            instr.offset
        )));
    }
    Ok(i32::from_le_bytes([
        instr.operands[0],
        instr.operands[1],
        instr.operands[2],
        instr.operands[3],
    ]))
}

/// Absolute byte-offset successors for a control-transfer instruction.
///
/// Jump targets are encoded relative to the IP after reading the i32
/// operand (opcode byte + 4-byte immediate), matching the legacy
/// collector in verify.rs.
fn successor_offsets(instr: &Instruction) -> Result<Vec<usize>, VerifyError> {
    let absolute =
        |rel: i32| -> usize { (instr.offset as i64 + 5 + rel as i64) as usize };

    Ok(match instr.opcode {
        Opcode::Jmp => vec![absolute(jump_rel_i32(instr)?)],
        Opcode::JmpIfTrue | Opcode::JmpIfFalse | Opcode::JmpIfNull | Opcode::JmpIfNotNull => {
            vec![absolute(jump_rel_i32(instr)?), instr.offset + instr_size(instr)]
        }
        Opcode::Return | Opcode::ReturnVoid | Opcode::Throw | Opcode::Rethrow | Opcode::Trap => {
            vec![]
        }
        _ => vec![instr.offset + instr_size(instr)],
    })
}

/// Build basic blocks over decoded instructions.
///
/// Leaders: instruction 0, every jump target, every instruction following a
/// terminator. Returns blocks in address order plus an offset→index map.
pub fn build_cfg(
    instrs: &[Instruction],
) -> Result<(Vec<CfgBlock>, HashMap<usize, usize>), VerifyError> {
    let mut offset_to_index: HashMap<usize, usize> = HashMap::new();
    for (idx, instr) in instrs.iter().enumerate() {
        offset_to_index.insert(instr.offset, idx);
    }

    // A function whose final instruction is not a terminator falls off the
    // end; report that directly rather than as a bad fallthrough target.
    if let Some(last) = instrs.last() {
        if !last.opcode.is_terminator() {
            return Err(VerifyError::FallOffEnd(last.offset));
        }
    }

    // Collect leader indices.
    let mut leaders: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    if !instrs.is_empty() {
        leaders.insert(0);
    }
    for (idx, instr) in instrs.iter().enumerate() {
        if instr.opcode.is_terminator() {
            if idx + 1 < instrs.len() {
                leaders.insert(idx + 1);
            }
            for succ_off in successor_offsets(instr)? {
                match offset_to_index.get(&succ_off) {
                    Some(&i) => leaders.insert(i),
                    None => {
                        return Err(VerifyError::InvalidJumpTarget {
                            target: succ_off,
                            offset: instr.offset,
                        })
                    }
                };
            }
        }
    }

    // Carve blocks at leader boundaries and wire successors.
    let leader_list: Vec<usize> = leaders.into_iter().collect();
    let block_of_instr: HashMap<usize, usize> = leader_list
        .iter()
        .enumerate()
        .map(|(pos, &start)| (start, pos))
        .collect();

    let mut blocks = Vec::with_capacity(leader_list.len());
    for (i, &start) in leader_list.iter().enumerate() {
        let end = leader_list.get(i + 1).copied().unwrap_or(instrs.len());
        let last = &instrs[end - 1];
        let successors = successor_offsets(last)?
            .into_iter()
            .map(|off| {
                let instr_idx = offset_to_index.get(&off).copied().ok_or(
                    VerifyError::InvalidJumpTarget { target: off, offset: last.offset },
                )?;
                block_of_instr.get(&instr_idx).copied().ok_or_else(|| {
                    VerifyError::ModuleValidation(format!(
                        "jump target {} is not a block leader",
                        off
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        blocks.push(CfgBlock { start, end, successors });
    }

    Ok((blocks, offset_to_index))
}

/// CFG-based verification of one function (B4 depth + B5 types).
pub fn verify_function_cfg(function: &Function, module: &Module) -> Result<(), VerifyError> {
    if function.code.is_empty() {
        return Ok(());
    }

    let instrs = parse_instructions(&function.code)?;
    if instrs.is_empty() {
        return Ok(());
    }

    let (blocks, _offset_map) = build_cfg(&instrs)?;

    // Typed mode requires recorded signature data (B1/B2). Untyped functions
    // verify under the legacy discipline: depths + init only.
    let signature: Option<&FunctionSignature> = if function.signature_id != UNTYPED_SIGNATURE_ID {
        Some(
            module
                .function_signatures
                .get(function.signature_id as usize - 1)
                .ok_or_else(|| {
                    VerifyError::ModuleValidation(format!(
                        "function '{}' references unknown signature {}",
                        function.name, function.signature_id
                    ))
                })?,
        )
    } else {
        None
    };

    let mut states: Vec<Option<AbstractState>> = vec![None; blocks.len()];
    states[0] = Some(AbstractState::new(function));
    let mut worklist: Vec<usize> = vec![0];

    while let Some(bi) = worklist.pop() {
        let mut cur = states[bi].clone().expect("queued block has state");
        let block = &blocks[bi];

        for idx in block.start..block.end {
            transfer(&instrs[idx], &mut cur, signature, function, module)?;
        }

        for &succ_pos in &block.successors {
            let succ_offset = instrs[blocks[succ_pos].start].offset;
            match &mut states[succ_pos] {
                None => {
                    states[succ_pos] = Some(cur.clone());
                    worklist.push(succ_pos);
                }
                Some(existing) => {
                    let merged = merge_states(existing, &cur, succ_offset)?;
                    if merged != *existing {
                        *existing = merged;
                        worklist.push(succ_pos);
                    }
                }
            }
        }
    }

    Ok(())
}

fn merge_states(
    existing: &AbstractState,
    incoming: &AbstractState,
    offset: usize,
) -> Result<AbstractState, VerifyError> {
    if existing.stack.len() != incoming.stack.len() {
        return Err(VerifyError::StackDepthMismatch {
            offset,
            expected: existing.stack.len(),
            actual: incoming.stack.len(),
        });
    }
    let stack = existing
        .stack
        .iter()
        .zip(incoming.stack.iter())
        .map(|(a, b)| AbsType::merge(*a, *b))
        .collect();
    let locals_init = existing
        .locals_init
        .iter()
        .zip(incoming.locals_init.iter())
        .map(|(a, b)| a | b)
        .collect();
    Ok(AbstractState { stack, locals_init })
}

// Descriptor ids for the implicit primitives.
const I32: u32 = 0;
const F64: u32 = 1;
const BOOL: u32 = 2;
const STRING: u32 = 3;
const NULL: u32 = 4;

/// Transfer rule for one instruction (B5).
fn transfer(
    instr: &Instruction,
    s: &mut AbstractState,
    signature: Option<&FunctionSignature>,
    function: &Function,
    module: &Module,
) -> Result<(), VerifyError> {
    use AbsType::Known;
    let off = instr.offset;

    match instr.opcode {
        // ===== Constants =====
        Opcode::ConstNull => s.push(Known(NULL), off)?,
        Opcode::ConstTrue | Opcode::ConstFalse => s.push(Known(BOOL), off)?,
        Opcode::ConstI32 => s.push(Known(I32), off)?,
        Opcode::ConstF64 => s.push(Known(F64), off)?,
        Opcode::ConstStr | Opcode::LoadConst => s.push(Known(STRING), off)?,

        // ===== Locals =====
        Opcode::LoadLocal | Opcode::LoadLocal0 | Opcode::LoadLocal1 => {
            let slot = local_slot(&instr.operands);
            require_local(slot, function.local_count, off)?;
            if !s.locals_init.get(slot as usize).copied().unwrap_or(false) {
                return Err(VerifyError::UninitializedLocal { offset: off, index: slot as usize });
            }
            s.push(local_type(slot as usize, function, module), off)?;
        }
        Opcode::StoreLocal | Opcode::StoreLocal0 | Opcode::StoreLocal1 => {
            let slot = local_slot(&instr.operands);
            require_local(slot, function.local_count, off)?;
            let value = s.pop(off)?;
            let declared = local_type(slot as usize, function, module);
            check_assignable(value, declared, off)?;
            if slot as usize >= s.locals_init.len() {
                s.locals_init.resize(slot as usize + 1, false);
            }
            s.locals_init[slot as usize] = true;
        }

        // ===== Integer arithmetic =====
        Opcode::Iadd | Opcode::Isub | Opcode::Imul | Opcode::Idiv | Opcode::Imod => {
            s.pop_expect(Known(I32), off)?;
            s.pop_expect(Known(I32), off)?;
            s.push(Known(I32), off)?;
        }
        Opcode::Ishl | Opcode::Ishr | Opcode::Iushr | Opcode::Iand | Opcode::Ior | Opcode::Ixor => {
            s.pop_expect(Known(I32), off)?;
            s.pop_expect(Known(I32), off)?;
            s.push(Known(I32), off)?;
        }
        Opcode::Ineg | Opcode::Inot => {
            s.pop_expect(Known(I32), off)?;
            s.push(Known(I32), off)?;
        }

        // ===== Float arithmetic =====
        Opcode::Fadd | Opcode::Fsub | Opcode::Fmul | Opcode::Fdiv => {
            s.pop_expect(Known(F64), off)?;
            s.pop_expect(Known(F64), off)?;
            s.push(Known(F64), off)?;
        }
        Opcode::Fneg => {
            s.pop_expect(Known(F64), off)?;
            s.push(Known(F64), off)?;
        }

        // ===== Comparisons produce bool =====
        Opcode::Ieq | Opcode::Ine | Opcode::Ilt | Opcode::Ile | Opcode::Igt | Opcode::Ige => {
            s.pop_expect(Known(I32), off)?;
            s.pop_expect(Known(I32), off)?;
            s.push(Known(BOOL), off)?;
        }
        Opcode::Feq | Opcode::Fne | Opcode::Flt | Opcode::Fle | Opcode::Fgt | Opcode::Fge => {
            s.pop_expect(Known(F64), off)?;
            s.pop_expect(Known(F64), off)?;
            s.push(Known(BOOL), off)?;
        }

        // ===== Logical =====
        Opcode::Not => {
            s.pop_expect(Known(BOOL), off)?;
            s.push(Known(BOOL), off)?;
        }
        Opcode::And | Opcode::Or => {
            s.pop_expect(Known(BOOL), off)?;
            s.pop_expect(Known(BOOL), off)?;
            s.push(Known(BOOL), off)?;
        }

        // ===== Stack manipulation =====
        Opcode::Pop => {
            s.pop(off)?;
        }
        Opcode::Dup => {
            let v = s.pop(off)?;
            s.push(v, off)?;
            s.push(v, off)?;
        }
        Opcode::Swap => {
            let a = s.pop(off)?;
            let b = s.pop(off)?;
            s.push(a, off)?;
            s.push(b, off)?;
        }
        Opcode::Nop => {}

        // ===== Control flow =====
        Opcode::Jmp => {}
        Opcode::JmpIfTrue | Opcode::JmpIfFalse => {
            s.pop_expect(Known(BOOL), off)?;
        }
        Opcode::JmpIfNull | Opcode::JmpIfNotNull => {
            s.pop_expect(Known(NULL), off)?;
        }
        Opcode::Return => {
            let got = s.pop(off)?;
            if let Some(sig) = signature {
                check_assignable(got, abs_of_descriptor(&sig.return_type, module), off)?;
            }
        }
        Opcode::ReturnVoid => {
            if let Some(sig) = signature {
                if sig.return_type != RuntimeTypeDescriptor::Void {
                    return Err(VerifyError::TypeMismatch {
                        offset: off,
                        expected: "void".into(),
                        actual: "value".into(),
                    });
                }
            }
        }

        // ===== Typed calls: push the callee's recorded return type =====
        Opcode::Call => {
            if instr.operands.len() >= 4 {
                let func_index = u32::from_le_bytes([
                    instr.operands[0],
                    instr.operands[1],
                    instr.operands[2],
                    instr.operands[3],
                ]);
                if let Some(callee) = module.functions.get(func_index as usize) {
                    if callee.signature_id != UNTYPED_SIGNATURE_ID {
                        if let Some(sig) =
                            module.function_signatures.get(callee.signature_id as usize - 1)
                        {
                            if sig.return_type != RuntimeTypeDescriptor::Void {
                                s.push(abs_of_descriptor(&sig.return_type, module), off)?;
                            }
                            return Ok(());
                        }
                    }
                }
            }
            legacy_transfer(instr, s, off)?;
        }

        // ===== Everything else: dynamic discipline =====
        _ => legacy_transfer(instr, s, off)?,
    }
    Ok(())
}

fn legacy_transfer(
    instr: &Instruction,
    s: &mut AbstractState,
    off: usize,
) -> Result<(), VerifyError> {
    let (pops, pushes) = super::verify::stack_effect(instr.opcode);
    if (s.stack.len() as i32) < pops {
        return Err(VerifyError::StackUnderflow(off));
    }
    for _ in 0..pops {
        s.pop(off)?;
    }
    for _ in 0..pushes {
        s.push(AbsType::Any, off)?;
    }
    Ok(())
}

fn local_slot(operands: &[u8]) -> u16 {
    if operands.len() >= 2 {
        u16::from_le_bytes([operands[0], operands[1]])
    } else {
        0
    }
}

fn require_local(slot: u16, max: usize, off: usize) -> Result<(), VerifyError> {
    if slot as usize >= max {
        return Err(VerifyError::InvalidLocalRef {
            index: slot as usize,
            max,
            offset: off,
        });
    }
    Ok(())
}

/// Declared type of a local slot, or Any when untyped.
fn local_type(slot: usize, function: &Function, module: &Module) -> AbsType {
    match function.local_types.get(slot) {
        Some(&id) => abs_of_id(id, module),
        None => AbsType::Any,
    }
}

/// Resolve a descriptor id to an abstract type; unknown complex ids degrade
/// to Any rather than lying.
fn abs_of_id(id: u32, module: &Module) -> AbsType {
    if id < COMPLEX_BASE {
        AbsType::Known(id)
    } else {
        match module.runtime_types.get((id - COMPLEX_BASE) as usize) {
            Some(_) => AbsType::Known(id),
            None => AbsType::Any,
        }
    }
}

fn abs_of_descriptor(d: &RuntimeTypeDescriptor, module: &Module) -> AbsType {
    // `AnyValue` is the fallback for a type the compiler could not resolve — it
    // means "no type information", not "some concrete type". It has a primitive
    // id (prim::ANY_VALUE = 6), so the `primitive_id()` arm below turned it into
    // `AbsType::Known(6)`, and `check_assignable` then demanded an EXACT match
    // against it. That made every unresolved type fail against everything:
    //
    //   function f(): int { return 7; } return f();
    //   -> TypeMismatch { expected: Known(6), actual: Known(0) }
    //
    // Mapping it to `AbsType::Any` restores its meaning: `check_assignable`
    // already treats `Any` as compatible in both directions.
    if matches!(d, RuntimeTypeDescriptor::AnyValue) {
        return AbsType::Any;
    }
    match d.primitive_id() {
        Some(id) => AbsType::Known(id.0),
        None => match module.runtime_types.iter().position(|t| t == d) {
            Some(pos) => AbsType::Known(COMPLEX_BASE + pos as u32),
            None => AbsType::Any,
        },
    }
}

fn check_assignable(value: AbsType, declared: AbsType, offset: usize) -> Result<(), VerifyError> {
    match (value, declared) {
        (AbsType::Any, _) | (_, AbsType::Any) => Ok(()),
        (AbsType::Known(v), AbsType::Known(d)) if v == d => Ok(()),
        _ => Err(VerifyError::TypeMismatch {
            offset,
            expected: format!("{declared:?}"),
            actual: format!("{value:?}"),
        }),
    }
}
