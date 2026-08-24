import ts from "typescript";
import type { RuleContext } from "../context.js";
import type { ReadinessRule } from "./rule.js";
import { collectNodes } from "../source-location.js";

function isJsonParseCall(node: ts.Node): node is ts.CallExpression {
  if (!ts.isCallExpression(node)) return false;
  const expr = node.expression;
  return (
    ts.isPropertyAccessExpression(expr) &&
    expr.expression.getText() === "JSON" &&
    expr.name.text === "parse"
  );
}

/**
 * RT1004: JSON.parse results are untyped external data. Flag direct entry
 * into typed positions (assertions or annotated declarations).
 */
export const validateJsonParse: ReadinessRule = {
  code: "RT1004",
  name: "validate-json-parse",
  run(ctx: RuleContext) {
    const findings = [];
    for (const sf of ctx.sourceFiles) {
      for (const node of collectNodes(sf, isJsonParseCall)) {
        const call = node as ts.CallExpression;
        const parent = call.parent;

        if (
          ts.isAsExpression(parent) ||
          ts.isTypeAssertionExpression(parent)
        ) {
          findings.push(
            ctx.makeFinding(sf, call, {
              code: "RT1004",
              severity: "error",
              stage: "runtime",
              message: "JSON.parse result asserted without runtime validation",
              remediation: "pass external data through a validator before casting",
              rule: "validate-json-parse",
            }),
          );
          continue;
        }

        if (ts.isVariableDeclaration(parent) && parent.type !== undefined) {
          findings.push(
            ctx.makeFinding(sf, call, {
              code: "RT1004",
              severity: "error",
              stage: "runtime",
              message: "JSON.parse result assigned to a typed declaration without validation",
              remediation: "validate first; the annotation alone does not check the data",
              rule: "validate-json-parse",
            }),
          );
        }
      }
    }
    return findings;
  },
};
