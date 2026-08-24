import ts from "typescript";
import type { RuleContext } from "../context.js";
import type { ReadinessRule } from "./rule.js";
import { collectNodes } from "../source-location.js";

/** RT1005: non-null assertions hide possible null/undefined at runtime. */
export const noNonNullAssertion: ReadinessRule = {
  code: "RT1005",
  name: "no-non-null-assertion",
  run(ctx: RuleContext) {
    const findings = [];
    for (const sf of ctx.sourceFiles) {
      for (const node of collectNodes(sf, (n) => n.kind === ts.SyntaxKind.NonNullExpression)) {
        findings.push(
          ctx.makeFinding(sf, node, {
            code: "RT1005",
            severity: "warning",
            stage: "runtime",
            message: "non-null assertion can hide a null/undefined that will throw at runtime",
            remediation: "narrow explicitly ('if (x == null)' or optional chaining) instead of '!'",
            rule: "no-non-null-assertion",
          }),
        );
      }
    }
    return findings;
  },
};
