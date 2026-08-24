import ts from "typescript";
import type { RuleContext } from "../context.js";
import type { ReadinessRule } from "./rule.js";
import { collectNodes } from "../source-location.js";

/** RT2004: eval and new Function() are unverifiable dynamic code. */
export const evalAndFunctionConstructor: ReadinessRule = {
  code: "RT2004",
  name: "eval-and-function-constructor",
  run(ctx: RuleContext) {
    const findings = [];
    for (const sf of ctx.sourceFiles) {
      // direct eval() calls
      const calls = collectNodes(
        sf,
        (n) => ts.isCallExpression(n) && ts.isIdentifier(n.expression) && n.expression.text === "eval",
      );
      for (const node of calls) {
        findings.push(
          ctx.makeFinding(sf, node, {
            code: "RT2004",
            severity: "error",
            stage: "runtime",
            message: "eval() executes dynamically generated code and cannot be compiled ahead of time",
            remediation: "replace with explicit parsing/dispatch over known operations",
            rule: "eval-and-function-constructor",
          }),
        );
      }

      // new Function(...)
      const news = collectNodes(
        sf,
        (n) =>
          ts.isNewExpression(n) &&
          ts.isIdentifier(n.expression) &&
          n.expression.text === "Function",
      );
      for (const node of news) {
        findings.push(
          ctx.makeFinding(sf, node, {
            code: "RT2004",
            severity: "error",
            stage: "runtime",
            message: "the Function constructor compiles code from strings at runtime",
            remediation: "replace with explicit parsing/dispatch over known operations",
            rule: "eval-and-function-constructor",
          }),
        );
      }
    }
    return findings;
  },
};
