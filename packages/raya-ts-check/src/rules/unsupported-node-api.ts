import ts from "typescript";
import type { RuleContext } from "../context.js";
import type { ReadinessRule } from "./rule.js";
import { collectNodes } from "../source-location.js";
import { loadNodeManifest } from "../compat/node-manifest.js";

interface ImportHit {
  node: ts.Node;
  specifier: string;
}

function collectImports(sf: ts.SourceFile): ImportHit[] {
  const hits: ImportHit[] = [];
  const visit = (node: ts.Node): void => {
    let specifier: string | undefined;
    if (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) {
      const mod = node.moduleSpecifier;
      if (mod && ts.isStringLiteral(mod)) specifier = mod.text;
    }
    if (specifier !== undefined) hits.push({ node, specifier });
    ts.forEachChild(node, visit);
  };
  ts.forEachChild(sf, visit);
  return hits;
}

const SEVERITY_BY_STATUS = {
  stub: "error",
  misleading: "error",
  partial: "warning",
  supported: undefined,
} as const;

/**
 * RT3001: flags node: imports that will fail, throw, or silently differ
 * from Node semantics under the Raya runtime. Also flags bare specifiers
 * of known builtins ("fs"), which Raya does not resolve.
 */
export const unsupportedNodeApi: ReadinessRule = {
  code: "RT3001",
  name: "unsupported-node-api",
  run(ctx: RuleContext) {
    const findings = [];
    const manifest = loadNodeManifest();

    for (const sf of ctx.sourceFiles) {
      for (const { node, specifier } of collectImports(sf)) {
        // native addons
        if (specifier.endsWith(".node")) continue; // handled by RT3003

        if (specifier.startsWith("node:") || specifier.startsWith("std:")) {
          const name = specifier.slice(specifier.indexOf(":") + 1);
          const entry = manifest.modules[name];

          if (!entry) {
            findings.push(
              ctx.makeFinding(sf, node, {
                code: "RT3001",
                severity: "error",
                stage: "compat",
                message: `'${specifier}' is not implemented by the Raya runtime; compilation will fail`,
                remediation: "replace with an equivalent Raya-supported API or vendor the functionality",
                rule: "unsupported-node-api",
              }),
            );
            continue;
          }

          const severity = SEVERITY_BY_STATUS[entry.status];
          if (severity === undefined) continue;

          findings.push(
            ctx.makeFinding(sf, node, {
              code: "RT3001",
              severity,
              stage: "compat",
              message:
                entry.status === "misleading"
                  ? `'${specifier}' resolves to a different (non-Node) API surface: ${entry.note ?? ""}`
                  : `'${specifier}' is ${entry.status} in Raya${entry.note ? `: ${entry.note}` : ""}`,
              remediation:
                entry.status === "misleading"
                  ? "audit every symbol used from this module against the actual runtime API"
                  : "verify the symbols you use exist and behave identically",
              rule: "unsupported-node-api",
            }),
          );
          continue;
        }

        // bare specifier that names a known builtin -> won't resolve
        if (
          !specifier.startsWith(".") &&
          !specifier.startsWith("/") &&
          !specifier.startsWith("@") &&
          (manifest.modules[specifier] !== undefined ||
            KNOWN_NODE_BUILTINS.has(specifier))
        ) {
          findings.push(
            ctx.makeFinding(sf, node, {
              code: "RT3001",
              severity: "error",
              stage: "compat",
              message: `bare specifier '${specifier}' is not resolved by Raya; use 'node:${specifier}'`,
              remediation: `change the import to 'node:${specifier}'`,
              rule: "unsupported-node-api",
            }),
          );
        }
      }
    }
    return findings;
  },
};

const KNOWN_NODE_BUILTINS = new Set([
  "assert", "async_hooks", "buffer", "child_process", "cluster", "console",
  "constants", "crypto", "dgram", "diagnostics_channel", "dns", "domain",
  "events", "fs", "http", "http2", "https", "inspector", "module", "net",
  "os", "path", "perf_hooks", "process", "punycode", "querystring",
  "readline", "repl", "sqlite", "stream", "string_decoder", "sys", "test",
  "timers", "tls", "trace_events", "tty", "url", "util", "v8", "vm",
  "worker_threads", "zlib",
]);
