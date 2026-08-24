import ts from "typescript";
import type { RuleContext } from "../context.js";
import type { ReadinessRule } from "./rule.js";
import { collectNodes } from "../source-location.js";

function isPrototypeAccess(node: ts.Node): boolean {
  if (!ts.isPropertyAccessExpression(node)) return false;
  return node.name.text === "prototype" || node.name.text === "__proto__";
}

/** RT2003: prototype mutation invalidates shape/layout assumptions. */
export const prototypeMutation: ReadinessRule = {
  code: "RT2003",
  name: "prototype-mutation",
  run(ctx: RuleContext) {
    const findings = [];
    for (const sf of ctx.sourceFiles) {
      // X.prototype / __proto__ as an assignment target
      const ASSIGN_OPS = new Set([
        ts.SyntaxKind.EqualsToken,
        ts.SyntaxKind.PlusEqualsToken,
        ts.SyntaxKind.MinusEqualsToken,
      ]);
      const assignments = collectNodes(
        sf,
        (n) => ts.isBinaryExpression(n) && ASSIGN_OPS.has(n.operatorToken.kind),
      ) as ts.BinaryExpression[];
      for (const assign of assignments) {
        const chain = collectNodes(assign.left, isPrototypeAccess);
        if (chain.length === 0) continue;
        findings.push(
          ctx.makeFinding(sf, assign, {
            code: "RT2003",
            severity: "error",
            stage: "jit",
            message: "mutating a prototype invalidates object layout assumptions",
            remediation: "define methods in class bodies; avoid runtime prototype edits",
            rule: "prototype-mutation",
          }),
        );
      }

      // Object.setPrototypeOf(...)
      const setProto = collectNodes(
        sf,
        (n) =>
          ts.isCallExpression(n) &&
          ts.isPropertyAccessExpression(n.expression) &&
          n.expression.expression.getText() === "Object" &&
          n.expression.name.text === "setPrototypeOf",
      );
      for (const node of setProto) {
        findings.push(
          ctx.makeFinding(sf, node, {
            code: "RT2003",
            severity: "error",
            stage: "jit",
            message: "Object.setPrototypeOf changes object identity/layout at runtime",
            remediation: "use composition or class hierarchies instead",
            rule: "prototype-mutation",
          }),
        );
      }
    }
    return findings;
  },
};
