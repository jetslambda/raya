# ADR 0001: Strict-mode runtime type contract

Status: proposed
Supersedes: nothing
Related plan: plans/raya-typescript-runtime-implementation-plan.md (R1)

## Context

Raya runs TypeScript-like source through a type checker whose modes already
diverge (`TypeMode::Raya` strict, `Ts`, `Js` in `crates/raya-runtime/src/compile.rs`).
Downstream stages disagree about semantics because no written contract binds them:
the verifier checks stack depth but not types, the interpreter silently coerces
bad operands to zero, and the JIT previously invented results for pow/mod.
Typed signatures (B1), a typed verifier (B5), exact interpreter semantics (I1-I3),
and any future JIT expansion all need one authoritative answer to "what does this
type mean at runtime?"

This document is that answer for **strict mode** (`TypeMode::Raya`). NodeCompat/Js
mode deliberately preserves JavaScript coercion semantics and is called out where
it differs. Where a decision conflicts with current behavior, the decision wins
and the behavior is a bug to fix in the referenced task.

## Decisions

### D1. Numeric types

- `int` is a signed 32-bit integer (i32). Arithmetic wraps (two's complement),
  matching `Int32Array`/WASM i32 expectations. `INT_MIN / -1` yields `INT_MIN`.
- `number` is IEEE-754 binary64 (f64) with all JavaScript numeric behaviors:
  NaN payloads collapse, `-0` distinct from `+0`, `1/0 = Infinity`.
- Integer division or remainder by zero raises a catchable runtime error
  (`DivisionByZero`), never a native trap and never a silent zero.
  Float division by zero follows IEEE (`Infinity`/`NaN`).
- Mixing `int` and `number` requires an explicit conversion expression.
  Implicit widening (`int -> number`) is allowed at boundaries marked safe;
  implicit narrowing is an error.
- Overflow/underflow of float-to-int conversions (`f64 -> int`) saturates in
  NodeCompat mode and raises in strict mode.

Consequences: the interpreter's `unwrap_or(0)` coercions become errors (task I1);
JIT lowering must insert the same wrap/div-check semantics (task J3).

### D2. Other primitives

- `boolean` is a distinct bool, never coerced to/from number in strict mode.
- `string` is an immutable managed reference; concatenation allocates.
- `null` is the sole bottom-ish value literal. Strict mode does not have a
  separate `undefined`; existing `undefined` sources normalize to `null`
  at strict boundaries. (Revisit if real-world porting demands both.)

### D3. Objects, classes, interfaces

- Classes are **nominal** at runtime: every instance carries its nominal type id
  (existing `NewType`/`IsNominal`/`CastNominal` machinery). `instanceof`-style
  checks are exact.
- Interfaces are **structural at compile time**. The compiler may attach a
  structural shape id (existing shapes/layouts metadata) so `as Interface`
  can emit a checked cast (`CastShape`) that throws on mismatch in strict mode.
- Object literals conforming to an interface get the interface's shape id when
  statically known; otherwise they are open objects and casts verify member-wise.
- Field order/layout is compiler-chosen; layouts are versioned and JIT code
  depending on them must invalidate on change (task D3 in plan).

### D4. Unions and narrowing

- Unions are erased at runtime. A value of `A | B` is represented as whichever
  member it carries; no tag word is materialized in v1.
- Narrowing happens through ordinary checks (`typeof`, `== null`, nominal tests,
  shape tests). After narrowing, the compiler emits the member's operations.
- Discriminated unions are encouraged but receive no special runtime support yet.

### D5. any, unknown, assertions

- `any` is rejected in strict mode (checker already enforces). In Js/NodeCompat
  mode `any` maps to the boxed dynamic `Value`.
- `unknown` is permitted and represented as boxed `Value`; use requires narrowing.
- `expr as T` compiles to a **checked cast** in strict mode: failure throws
  `TypeCastError` naming target type and source position. In Js mode it is an
  unchecked reinterpretation, preserving TS semantics.
- Non-null assertion `x!` inserts a null check that throws in strict mode.

### D6. Functions and generics

- Every function carries a canonical signature `(params, return, rest?, flags)`
  in bytecode metadata (tasks R3/B1). Signatures are interned and hashed;
  module linking verifies them structurally (task B3).
- Generics are **monomorphized by use** (existing `MonomorphizationMode::
  ConsumerLink`). Runtime reification of generic parameters is out of scope.
- Closures capture by reference cells (existing ref-cell model).

### D7. Arrays, tuples, records

- `T[]` is a typed array carrying its element descriptor; element stores of the
  wrong type raise rather than coerce (strict). `any[]` cannot be expressed in
  strict mode.
- Tuples are fixed-length typed arrays with per-index descriptors.
- `Record<string, T>` lowers to a string-keyed map, not an object layout, so
  dynamic keys stay out of the fixed-layout fast path (aligns with RT2002).

### D8. Tasks and async

- `Task<T>` is a typed eventual value; `await` is the only suspension point.
- Task result types are preserved end-to-end; the known bug where non-number
  task results lose their type identity is tracked as task I4 and blocks no
  other work here.
- JIT compilation of suspendable functions is out of scope until deopt
  infrastructure exists (plan Phase 7); they run interpreted.

### D9. Exceptions

- `try/throw/rethrow` execute correctly in the interpreter today and stay
  interpreter-only under the JIT until handler tables + unwinding land
  (consistent with the Phase 0 capability gate).

### D10. Boundaries between strict and dynamic worlds

- External data (JSON, network, FFI) enters strict scope only through values
  typed `Json`/`unknown` followed by validation or checked casts. There is no
  implicit bridge.
- NodeCompat modules may pass boxed `Value`s among themselves; crossing into a
  strict function invokes signature-checked entry (guards + deopt-or-error,
  tasks J1/D2).

## Non-decisions (explicitly deferred)

- Tagged union representations and pattern-match lowering.
- Value-type structs / non-boxed aggregates.
- Threads/shared memory beyond the existing message-passing tasks.
- A second numeric width (i64/u64) — revisit after real workloads exist.

## Adoption path

Each decision above maps to plan tasks that make the codebase obey it:
I1-I3 (interpreter exactness), B1-B3 (signatures/linking), B4-B6 (verifier),
D3/D4 (layout guards, family expansion), J1-J3 (JIT conformance). A change
that violates this document needs a new ADR, not a local workaround.
