import ts from "typescript";
import type { RuleContext } from "../context.js";
import type { ReadinessRule } from "./rule.js";
import { collectNodes } from "../source-location.js";

/**
 * RT2002: keyed access with non-literal, non-numeric keys escapes the typed
 * property model. Numeric indexing on arrays stays allowed.
 */
export const dynamicPropertyAccess: ReadinessRule = {
  code: "RT2002",
  name: "dynamic-property-access",
  run(ctx: RuleContext) {
    const findings = [];
    for (const sf of ctx.sourceFiles) {
      const accesses = collectNodes(sf, (n) => ts.isElementAccessExpression(n));
      for (const node of accesses) {
        const elem = node as ts.ElementAccessExpression;
        const arg = elem.argumentExpression;
        if (arg === undefined) continue;

        // literal keys are fine ("a"["b"], arr[0])
        if (ts.isStringLiteral(arg) || ts.isNumericLiteral(arg)) continue;

        // numeric-typed arguments are array indexing — fine
        const argTy = ctx.checker.getTypeAtLocation(arg);
        const isNumeric =
          (argTy.flags &
            (ts.TypeFlags.Number |
              ts.TypeFlags.NumberLike |
              ts.TypeFlags.NumberLiteral)) !== 0;
        if (isNumeric) continue;

        findings.push(
          ctx.makeFinding(sf, elem, {
            code: "RT2002",
            severity: "warning",
            stage: "jit",
            message: "keyed access with a dynamic string key cannot use a fixed object layout",
            remediation:
              "use a Map<string, T> for dynamic keys, or narrow to known property names",
            rule: "dynamic-property-access",
          }),
        );
      }
    }
    return findings;
  },
};
