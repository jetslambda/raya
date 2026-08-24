import ts from "typescript";
import type { RuleContext } from "../context.js";
import type { ReadinessRule } from "./rule.js";
import { collectNodes } from "../source-location.js";

function isAnyOrUnknown(ctx: RuleContext, node: ts.Node): boolean {
  const ty = ctx.checker.getTypeAtLocation(node);
  return (ty.flags & (ts.TypeFlags.Any | ts.TypeFlags.Unknown)) !== 0;
}

/** RT1003: assertions from any/unknown into narrower types are unchecked lies. */
export const noUnsafeTypeAssertion: ReadinessRule = {
  code: "RT1003",
  name: "no-unsafe-type-assertion",
  run(ctx: RuleContext) {
    const findings = [];
    for (const sf of ctx.sourceFiles) {
      const nodes = collectNodes(
        sf,
        (n) =>
          n.kind === ts.SyntaxKind.AsExpression ||
          n.kind === ts.SyntaxKind.TypeAssertionExpression,
      );
      for (const node of nodes) {
        const expr = (node as ts.AsExpression).expression;
        if (!isAnyOrUnknown(ctx, expr)) continue;
        findings.push(
          ctx.makeFinding(sf, node, {
            code: "RT1003",
            severity: "error",
            stage: "runtime",
            message: "type assertion from 'any'/'unknown' performs no runtime check",
            remediation: "validate the value at the boundary (schema/parser) before asserting",
            rule: "no-unsafe-type-assertion",
          }),
        );
      }
    }
    return findings;
  },
};
