import ts from "typescript";
import type { RuleContext } from "../context.js";
import type { ReadinessRule } from "./rule.js";
import { collectNodes } from "../source-location.js";

/**
 * RT1002: values whose static type is any without an explicit annotation.
 * Covers unannotated parameters on named functions/methods and variables
 * initialized from any-typed expressions. Arrow-function parameters are
 * skipped: their types usually arrive via contextual inference.
 */
export const noImplicitAny: ReadinessRule = {
  code: "RT1002",
  name: "no-implicit-any",
  run(ctx: RuleContext) {
    const findings = [];

    for (const sf of ctx.sourceFiles) {
      // Unannotated parameters on function/method declarations.
      const funcs = collectNodes(
        sf,
        (n) =>
          ts.isFunctionDeclaration(n) ||
          ts.isMethodDeclaration(n),
      );
      for (const fn of funcs as ts.FunctionDeclaration[]) {
        for (const param of fn.parameters) {
          if (param.type !== undefined) continue;
          if (!ts.isIdentifier(param.name)) continue;
          findings.push(
            ctx.makeFinding(sf, param, {
              code: "RT1002",
              severity: "warning",
              stage: "type",
              message: `parameter '${param.name.text}' has no type annotation (implicit any)`,
              remediation: "annotate the parameter type explicitly",
              rule: "no-implicit-any",
            }),
          );
        }
      }

      // Variables inferred as any.
      const vars = collectNodes(sf, (n) => ts.isVariableDeclaration(n));
      for (const decl of vars as ts.VariableDeclaration[]) {
        if (!ts.isIdentifier(decl.name)) continue;
        if (decl.type !== undefined) continue;
        const ty = ctx.checker.getTypeAtLocation(decl.name);
        if ((ty.flags & ts.TypeFlags.Any) === 0) continue;
        findings.push(
          ctx.makeFinding(sf, decl, {
            code: "RT1002",
            severity: "warning",
            stage: "type",
            message: `variable '${decl.name.text}' is implicitly any`,
            remediation: "add an explicit type or initialize from a typed value",
            rule: "no-implicit-any",
          }),
        );
      }
    }

    return findings;
  },
};
