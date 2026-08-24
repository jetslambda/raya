import ts from "typescript";
import type { RuleContext } from "../context.js";
import type { ReadinessRule } from "./rule.js";
import { collectNodes } from "../source-location.js";

const ARITH_OPS = new Set([
  ts.SyntaxKind.PlusToken,
  ts.SyntaxKind.MinusToken,
  ts.SyntaxKind.AsteriskToken,
  ts.SyntaxKind.SlashToken,
  ts.SyntaxKind.PercentToken,
  ts.SyntaxKind.AsteriskAsteriskToken,
]);

/**
 * RT2001 (informational): arithmetic-heavy functions whose contract uses
 * plain `number` cannot compile to specialized int/float machine code.
 * Raya's strict mode distinguishes int from number.
 */
export const ambiguousNumber: ReadinessRule = {
  code: "RT2001",
  name: "ambiguous-number",
  run(ctx: RuleContext) {
    const findings = [];
    for (const sf of ctx.sourceFiles) {
      const fns = collectNodes(
        sf,
        (n) => ts.isFunctionDeclaration(n) || ts.isMethodDeclaration(n),
      ) as (ts.FunctionDeclaration | ts.MethodDeclaration)[];

      for (const fn of fns) {
        if (!fn.body) continue;

        // public contract mentions plain `number`
        const numberParams = fn.parameters.filter(
          (p) => p.type !== undefined && p.type.getText() === "number",
        );
        const returnsNumber =
          fn.type !== undefined && fn.type.getText() === "number";
        if (numberParams.length === 0 && !returnsNumber) continue;

        // body does arithmetic
        const arith = collectNodes(
          fn.body,
          (n) => ts.isBinaryExpression(n) && ARITH_OPS.has(n.operatorToken.kind),
        );
        if (arith.length < 3) continue; // threshold keeps trivial cases quiet

        findings.push(
          ctx.makeFinding(sf, fn, {
            code: "RT2001",
            severity: "info",
            stage: "jit",
            message: `'${fn.name?.getText() ?? "<anonymous>"}' mixes 'number' with ${arith.length} arithmetic sites`,
            remediation:
              "if values are integral, prefer 'int' so hot loops compile to native integer ops",
            rule: "ambiguous-number",
          }),
        );
      }
    }
    return findings;
  },
};
