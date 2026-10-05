//! Tests for the `console` global (ALY-106).
//!
//! These assert that a `console.*` call type-checks, lowers to the right native,
//! and executes without error. The printed text itself is asserted by the CLI
//! end-to-end check, because stdout is not observable from inside the VM.

use super::harness::*;

#[test]
fn test_console_log_accepts_one_argument() {
    compile_and_run_runtime(r#"console.log("hello");"#).expect("console.log should run");
}

#[test]
fn test_console_log_accepts_many_arguments() {
    compile_and_run_runtime(r#"console.log("a", 1, true, null);"#)
        .expect("console.log should accept any number of arguments");
}

#[test]
fn test_console_log_accepts_zero_arguments() {
    compile_and_run_runtime(r#"console.log();"#).expect("console.log() should run");
}

#[test]
fn test_console_info_runs() {
    compile_and_run_runtime(r#"console.info("i");"#).expect("console.info should run");
}

#[test]
fn test_console_warn_runs() {
    compile_and_run_runtime(r#"console.warn("w");"#).expect("console.warn should run");
}

#[test]
fn test_console_error_runs() {
    compile_and_run_runtime(r#"console.error("e");"#).expect("console.error should run");
}

#[test]
fn test_console_log_returns_null() {
    let v = compile_and_run_runtime(r#"console.log("x");"#).expect("should run");
    assert!(v.is_null(), "console.log should evaluate to null");
}

#[test]
fn test_console_log_inside_a_function_runs() {
    compile_and_run_runtime(
        r#"
        function greet(name: string): null {
            console.log("hi", name);
            return null;
        }
        greet("world");
        "#,
    )
    .expect("console.log inside a function should run");
}

#[test]
fn test_console_log_in_a_loop_runs() {
    compile_and_run_runtime(
        r#"
        for (let i = 0; i < 3; i++) {
            console.log("tick", i);
        }
        "#,
    )
    .expect("console.log in a loop should run");
}

#[test]
fn test_unknown_console_method_is_still_rejected() {
    // `console` exists, but `console.nope()` must not silently resolve to a native.
    assert!(
        compile_and_run_runtime(r#"console.nope("x");"#).is_err(),
        "an unknown console method should not resolve"
    );
}
