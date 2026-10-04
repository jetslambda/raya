//! Pins every well-known primitive `TypeId` to the descriptor the runtime actually
//! resolves for it. **This test fails today, and that is the point** — it is the
//! executable form of the bug behind ALY-84's eight failing `test_runner` tests.
//!
//! `resolve_element_descriptor` decodes a sub-`COMPLEX_BASE` index through a
//! hardcoded table (`0 => I32, 1 => F64, 2 => Bool, 3 => String, ...`). Those
//! indices are the checker's *interned* `TypeId`s, assigned as
//! `TypeId(self.types.len())` in seeding order — so the table is asserting a mapping
//! that was never true. Measured on a fresh `TypeContext`:
//!
//! ```text
//! number=0  int=16  string=1  boolean=2  void=4  never=5
//! ```
//!
//! So `0` is Number (f64), not I32; `1` is String, not F64; `4` is Void, not Null.
//! `never` is 5, which falls through to `_ => None` — unconstrained, which is the
//! intended behaviour and is why `never[]` was never the culprit.
//!
//! The user-visible consequence is that `let a: string[] = []; a.push("x")` is
//! rejected with *"array element type is f64"*.

use raya_engine::compiler::bytecode::types::RuntimeTypeDescriptor;
use raya_engine::parser::types::context::TypeContext;

fn descriptor_for(type_ctx: &mut TypeContext, ty: raya_engine::parser::types::ty::TypeId) -> String {
    // Mirrors the runtime's decode: sub-COMPLEX_BASE indices go through the fixed
    // table. Reproduced here so this test fails without needing a Module.
    let idx = ty.as_u32();
    let atom = if idx < raya_engine::compiler::bytecode::types::COMPLEX_BASE {
        match idx {
            0 => Some(RuntimeTypeDescriptor::I32),
            1 => Some(RuntimeTypeDescriptor::F64),
            2 => Some(RuntimeTypeDescriptor::Bool),
            3 => Some(RuntimeTypeDescriptor::String),
            4 => Some(RuntimeTypeDescriptor::Null),
            5 => Some(RuntimeTypeDescriptor::Void),
            6 => None,
            7 => Some(RuntimeTypeDescriptor::Ref),
            _ => None,
        }
    } else {
        None
    };
    match atom {
        Some(d) => d.atom().to_string(),
        None => "<unconstrained>".to_string(),
    }
}

#[test]
fn each_primitive_type_id_resolves_to_its_own_descriptor() {
    let mut ctx = TypeContext::new();
    let number = ctx.number_type();
    let string = ctx.string_type();
    let boolean = ctx.boolean_type();
    let void = ctx.void_type();
    let cases: Vec<(&str, String, &str)> = vec![
        ("number", descriptor_for(&mut ctx, number), "f64"),
        ("string", descriptor_for(&mut ctx, string), "string"),
        ("boolean", descriptor_for(&mut ctx, boolean), "bool"),
        ("void", descriptor_for(&mut ctx, void), "void"),
    ];
    let mut mismatches = Vec::new();
    for (name, got, want) in cases {
        println!("{name}: got {got}, want {want}");
        if got != want {
            mismatches.push(format!("{name}: got {got}, want {want}"));
        }
    }
    assert!(
        mismatches.is_empty(),
        "primitive TypeIds do not resolve to their own descriptors: {}",
        mismatches.join("; ")
    );
}
