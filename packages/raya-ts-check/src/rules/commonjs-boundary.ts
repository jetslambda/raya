import ts from "typescript";
import type { RuleContext } from "../context.js";
import type { ReadinessRule } from "./rule.js";
import { collectNodes } from "../source-location.js";

/** require("...") with a string literal specifier. */
function requireSpecifier(node: ts.Node): string | undefined {
  if (!ts.isCallExpression(node)) return undefined;
  if (!ts.isIdentifier(node.expression) || node.expression.text !== "require") {
    return undefined;
  }
  const arg = node.arguments[0];
  return arg !== undefined && ts.isStringLiteral(arg) ? arg.text : undefined;
}

/**
 * Assignment target rooted at module.exports or exports.<name>,
 * including nested paths like module.exports.ns.deep.
 */
function isCommonJsExportTarget(left: ts.Expression): boolean {
  if (!ts.isPropertyAccessExpression(left)) return false;
  const base = left.expression;
  if (ts.isIdentifier(base)) {
    return (
      (base.text === "module" && left.name.text === "exports") ||
      base.text === "exports"
    );
  }
  return isCommonJsExportTarget(base);
}

/**
 * RT3002: CommonJS interop is a boundary surface for the Raya loader.
 * require() calls and exports assignments do not map onto ESM linking.
 */
export const commonjsBoundary: ReadinessRule = {
  code: "RT3002",
  name: "commonjs-boundary",
  run(ctx: RuleContext) {
    const findings = [];
    for (const sf of ctx.sourceFiles) {
      const requires = collectNodes(sf, (n) => requireSpecifier(n) !== undefined);
      for (const node of requires as ts.CallExpression[]) {
        findings.push(
          ctx.makeFinding(sf, node, {
            code: "RT3002",
            severity: "warning",
            stage: "compat",
            message: `require('${requireSpecifier(node)}') is a CommonJS boundary surface`,
            remediation: "prefer ESM imports; isolate unavoidable CJS interop behind an explicit wrapper",
            rule: "commonjs-boundary",
          }),
        );
      }

      const assignments = collectNodes(
        sf,
        (n) =>
          ts.isBinaryExpression(n) &&
          n.operatorToken.kind === ts.SyntaxKind.EqualsToken &&
          isCommonJsExportTarget(n.left),
      ) as ts.BinaryExpression[];
      for (const node of assignments) {
        findings.push(
          ctx.makeFinding(sf, node, {
            code: "RT3002",
            severity: "warning",
            stage: "compat",
            message: `assignment to '${node.left.getText()}' is a CommonJS boundary surface`,
            remediation: "prefer ESM imports and named exports instead of mutating module.exports/exports",
            rule: "commonjs-boundary",
          }),
        );
      }
    }
    return findings;
  },
};
