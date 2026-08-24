import ts from "typescript";
import type { RuleContext } from "../context.js";
import type { ReadinessRule } from "./rule.js";
import { collectNodes } from "../source-location.js";

const REFLECT_METHODS = new Set([
  "get",
  "set",
  "has",
  "deleteProperty",
  "defineProperty",
  "getPrototypeOf",
  "setPrototypeOf",
  "ownKeys",
  "apply",
  "construct",
]);

function isProxyNew(node: ts.Node): node is ts.NewExpression {
  return (
    ts.isNewExpression(node) &&
    ts.isIdentifier(node.expression) &&
    node.expression.text === "Proxy"
  );
}

function reflectMethodName(node: ts.Node): string | undefined {
  if (!ts.isCallExpression(node)) return undefined;
  const expr = node.expression;
  if (!ts.isPropertyAccessExpression(expr)) return undefined;
  if (expr.expression.getText() !== "Reflect") return undefined;
  const name = expr.name.text;
  return REFLECT_METHODS.has(name) ? name : undefined;
}

/**
 * RT2005: Proxy traps and Reflect.* calls intercept property operations,
 * so static property analysis and layout specialization see an incomplete
 * picture of what touches each object shape.
 */
export const proxyReflectUsage: ReadinessRule = {
  code: "RT2005",
  name: "proxy-reflect-usage",
  run(ctx: RuleContext) {
    const findings = [];
    for (const sf of ctx.sourceFiles) {
      const proxies = collectNodes(sf, isProxyNew);
      for (const node of proxies) {
        findings.push(
          ctx.makeFinding(sf, node, {
            code: "RT2005",
            severity: "warning",
            stage: "jit",
            message:
              "new Proxy intercepts property access; metaprogramming escapes static property analysis",
            remediation:
              "use direct property access, or replace the trap with an explicit dispatch table over known operations",
            rule: "proxy-reflect-usage",
          }),
        );
      }

      const reflectCalls = collectNodes(sf, (n) => reflectMethodName(n) !== undefined);
      for (const node of reflectCalls as ts.CallExpression[]) {
        findings.push(
          ctx.makeFinding(sf, node, {
            code: "RT2005",
            severity: "warning",
            stage: "jit",
            message: `Reflect.${reflectMethodName(node)}() bypasses direct property semantics; metaprogramming escapes static property analysis`,
            remediation:
              "use direct property access, or replace with an explicit dispatch table over known operations",
            rule: "proxy-reflect-usage",
          }),
        );
      }
    }
    return findings;
  },
};
