# raya-ts-check

Static readiness checker for ordinary TypeScript projects. It analyzes an
existing project with the TypeScript compiler API and reports what would
break if the code were compiled to the Raya typed runtime — a Rust VM with
C#-style runtime typing that rejects `any`, unsupported syntax, Node-only
APIs, and patterns its JIT cannot specialize.

## Install and run

```sh
cd packages/raya-ts-check
npm install
npm run build          # emits dist/cli.js

npx raya-ts-check --project path/to/project   # when installed or linked
node dist/cli.js --project path/to/project    # direct binary invocation
```

`--project` accepts a directory containing `tsconfig.json` or a path to a
tsconfig file.

## CLI options

| Option       | Values                    | Default  | Description                                        |
| ------------ | ------------------------- | -------- | -------------------------------------------------- |
| `--project`  | tsconfig path or dir      | required | Project to analyze                                 |
| `--format`   | `pretty`, `json`, `sarif` | `pretty` | Output format (`sarif` is SARIF 2.1.0 on stdout)   |
| `--fail-on`  | `info`, `warning`, `error`| `error`  | Severity threshold that turns findings into exit 1 |
| `--emit-plan`| file path                 | off      | Also write a grouped JSON migration plan (see below)|

Exit codes: `0` pass (no finding at/above the threshold), `1` threshold met,
`2` configuration failure (bad option, missing tsconfig, unwritable plan file).

## Conversion gates

Findings roll up into four gates:

- **type safety** — whether every value keeps a precise static type across
  conversion (explicit/implicit `any`, unchecked assertions, unvalidated
  `JSON.parse`, non-null assertions).
- **raya syntax** — whether all syntax is accepted by the Raya parser today
  (decorators, constructor parameter properties, ambient declarations).
- **runtime compatibility** — whether the code depends on APIs or interop the
  Raya runtime does not provide (Node builtins, native addons, CommonJS).
- **jit specialization** — whether hot code can still compile to specialized
  machine code (dynamic keys, prototype mutation, `eval`, Proxy/Reflect,
  ambiguous `number` arithmetic).

## Migration plans

`--emit-plan <file>` writes the findings regrouped for migration work, next
to the normal output; it never changes the exit code. Groups:
`autoFixable` (reserved, currently empty), `manualEdits` (code, file, line,
message, remediation), `dependencyReplacements` (RT3001/RT3003 hits needing a
replacement dependency), `blockedByRaya` (RT3004 plus anything whose message
says it cannot be compiled or is not supported).

## Rules

| Code   | Name                          | Reports                                                              |
| ------ | ----------------------------- | -------------------------------------------------------------------- |
| RT1001 | no-explicit-any               | explicit `any`, which disables type checking and blocks typed compilation |
| RT1002 | no-implicit-any               | parameters or variables whose type is implicitly `any`                |
| RT1003 | no-unsafe-type-assertion      | assertions from `any`/`unknown` to narrower types with no runtime check |
| RT1004 | validate-json-parse           | `JSON.parse` results entering typed positions without runtime validation |
| RT1005 | no-non-null-assertion         | `!` assertions that can hide a null/undefined throwing at runtime     |
| RT2001 | ambiguous-number              | arithmetic on plain `number`, which blocks int/float specialization   |
| RT2002 | dynamic-property-access       | element access with non-literal string keys, preventing fixed layouts |
| RT2003 | prototype-mutation            | `prototype`/`__proto__` mutation invalidating object layout assumptions |
| RT2004 | eval-and-function-constructor | `eval()` / `new Function()`, unverifiable dynamic code                |
| RT2005 | proxy-reflect-usage           | `Proxy` traps and `Reflect.*` calls escaping static property analysis |
| RT3001 | unsupported-node-api          | `node:` imports unimplemented/partial/misleading in Raya, bare builtin specifiers |
| RT3002 | commonjs-boundary             | `require()` calls and `module.exports` assignments, which do not map onto ESM linking |
| RT3003 | native-addon                  | imports of `.node` binaries, which the Raya runtime cannot load       |
| RT3004 | unsupported-typescript-syntax | decorators, parameter properties, ambient `declare global`/`declare module` |

## Example

```console
$ node dist/cli.js --project examples/ts-syntax
raya-ts-check — /workspace/examples/ts-syntax
files analyzed: 1, findings: 3

conversion gates:
  type safety            pass
  raya syntax            FAIL
  runtime compatibility  pass
  jit specialization     likely

src/main.ts:8:1  RT3004 [error] decorators are syntax the Raya parser does not support today
    fix: refactor to wrapper functions or explicit registration instead of decorating classes and members
...
$ echo $?
1
```
