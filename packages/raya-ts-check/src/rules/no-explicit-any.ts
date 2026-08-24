import ts from "typescript";
import type { RuleContext } from "../context.js";
import type { ReadinessRule } from "./rule.js";
import { collectNodes } from "../source-location.js";

/** RT1001: explicit `any` defeats static typing and blocks typed compilation. */
export const noExplicitAny: ReadinessRule = {
  code: "RT1001",
  name: "no-explicit-any",
  run(ctx: RuleContext) {
    const findings = [];
    for (const sf of ctx.sourceFiles) {
      for (const node of collectNodes(sf, (n) => n.kind === ts.SyntaxKind.AnyKeyword)) {
        findings.push(
          ctx.makeFinding(sf, node, {
            code: "RT1001",
            severity: "warning",
            stage: "type",
            message: "explicit 'any' disables type checking for this value",
            remediation:
              "use 'unknown' with narrowing, or a concrete type; Raya strict mode rejects 'any'",
            rule: "no-explicit-any",
          }),
        );
      }
    }
    return findings;
  },
};
