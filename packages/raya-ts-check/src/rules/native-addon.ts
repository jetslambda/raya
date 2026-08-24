import ts from "typescript";
import type { RuleContext } from "../context.js";
import type { ReadinessRule } from "./rule.js";
import { collectNodes } from "../source-location.js";

/** RT3003: native addons (.node binaries) cannot load in the Raya runtime. */
export const nativeAddon: ReadinessRule = {
  code: "RT3003",
  name: "native-addon",
  run(ctx: RuleContext) {
    const findings = [];
    for (const sf of ctx.sourceFiles) {
      const imports = collectNodes(
        sf,
        (n) =>
          (ts.isImportDeclaration(n) || ts.isCallExpression(n)) &&
          getSpecifierText(n)?.endsWith(".node") === true,
      );
      for (const node of imports) {
        findings.push(
          ctx.makeFinding(sf, node, {
            code: "RT3003",
            severity: "error",
            stage: "compat",
            message: "native addons (.node) cannot be loaded by the Raya runtime",
            remediation: "find a pure-TypeScript alternative or move the integration behind a service boundary",
            rule: "native-addon",
          }),
        );
      }
    }
    return findings;
  },
};

function getSpecifierText(node: ts.Node): string | undefined {
  if (ts.isImportDeclaration(node)) {
    const mod = node.moduleSpecifier;
    return ts.isStringLiteral(mod) ? mod.text : undefined;
  }
  if (ts.isCallExpression(node)) {
    const expr = node.expression;
    const isRequire =
      ts.isIdentifier(expr) && expr.text === "require";
    const arg = node.arguments[0];
    if (isRequire && arg && ts.isStringLiteral(arg)) return arg.text;
  }
  return undefined;
}
