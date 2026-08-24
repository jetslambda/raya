import ts from "typescript";
import type { RuleContext } from "../context.js";
import type { ReadinessRule } from "./rule.js";
import { collectNodes } from "../source-location.js";

const PARAMETER_PROPERTY_MODIFIERS = new Set([
  ts.SyntaxKind.PublicKeyword,
  ts.SyntaxKind.PrivateKeyword,
  ts.SyntaxKind.ProtectedKeyword,
  ts.SyntaxKind.ReadonlyKeyword,
]);

function isDecorator(node: ts.Node): node is ts.Decorator {
  return node.kind === ts.SyntaxKind.Decorator;
}

function hasDeclareKeyword(node: ts.ModuleDeclaration): boolean {
  return (node.modifiers ?? []).some(
    (m) => m.kind === ts.SyntaxKind.DeclareKeyword,
  );
}

function isAmbientModule(node: ts.Node): node is ts.ModuleDeclaration {
  if (!ts.isModuleDeclaration(node)) return false;
  return (
    (ts.isIdentifier(node.name) && node.name.text === "global") ||
    hasDeclareKeyword(node)
  );
}

/**
 * RT3004: syntax the Raya parser does not support today — decorators,
 * constructor parameter properties, and ambient `declare global` /
 * `declare module` blocks.
 */
export const unsupportedTypescriptSyntax: ReadinessRule = {
  code: "RT3004",
  name: "unsupported-typescript-syntax",
  run(ctx: RuleContext) {
    const findings = [];
    for (const sf of ctx.sourceFiles) {
      // decorators
      for (const node of collectNodes(sf, isDecorator)) {
        findings.push(
          ctx.makeFinding(sf, node, {
            code: "RT3004",
            severity: "error",
            stage: "syntax",
            message: "decorators are syntax the Raya parser does not support today",
            remediation:
              "refactor to wrapper functions or explicit registration instead of decorating classes and members",
            rule: "unsupported-typescript-syntax",
          }),
        );
      }

      // constructor parameter properties
      const ctors = collectNodes(sf, ts.isConstructorDeclaration);
      for (const ctor of ctors as ts.ConstructorDeclaration[]) {
        for (const param of ctor.parameters) {
          const isParamProperty = (param.modifiers ?? []).some((m) =>
            PARAMETER_PROPERTY_MODIFIERS.has(m.kind),
          );
          if (!isParamProperty) continue;
          findings.push(
            ctx.makeFinding(sf, param, {
              code: "RT3004",
              severity: "error",
              stage: "syntax",
              message: "constructor parameter properties are syntax the Raya parser does not support today",
              remediation: "declare the field explicitly and assign it in the constructor body",
              rule: "unsupported-typescript-syntax",
            }),
          );
        }
      }

      // ambient declarations
      for (const node of collectNodes(sf, isAmbientModule)) {
        findings.push(
          ctx.makeFinding(sf, node, {
            code: "RT3004",
            severity: "error",
            stage: "syntax",
            message: "'declare global'/'declare module' ambient declarations are syntax the Raya parser does not support today",
            remediation: "move ambient declarations into a .d.ts excluded from compilation",
            rule: "unsupported-typescript-syntax",
          }),
        );
      }
    }
    return findings;
  },
};
