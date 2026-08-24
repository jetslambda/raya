import * as fs from "node:fs";
import * as path from "node:path";
import ts from "typescript";
import type { ReadinessFinding, ReadinessReport } from "./types.js";
import { createContext } from "./context.js";
import { rules } from "./rules/registry.js";

/**
 * Analyze a TypeScript project for Raya readiness.
 *
 * C1 scope: load tsconfig via the official Compiler API, build a Program,
 * obtain a TypeChecker, and return an empty-findings report. Rules are
 * added in later tasks; the contract stabilizes here first.
 */
export async function analyzeProject(projectPath: string): Promise<ReadinessReport> {
  const absolute = path.isAbsolute(projectPath)
    ? projectPath
    : path.resolve(process.cwd(), projectPath);

  const { parsed, configPath } = readConfig(absolute);
  const program = ts.createProgram(parsed.fileNames, parsed.options);
  const ctx = createContext(program, path.dirname(configPath));

  // Run every registered rule, then order findings deterministically
  // (file, position, code) so reports are diff-stable across runs.
  const findings: ReadinessFinding[] = [];
  for (const rule of rules) {
    findings.push(...rule.run(ctx));
  }
  findings.sort(
    (a, b) =>
      a.file.localeCompare(b.file) ||
      a.start - b.start ||
      a.code.localeCompare(b.code),
  );

  const filesAnalyzed = ctx.sourceFiles.length;

  return {
    schemaVersion: 1,
    project: absolute,
    filesAnalyzed,
    findings,
    summary: summarize(findings),
  };
}

function readConfig(projectPath: string): { parsed: ts.ParsedCommandLine; configPath: string } {
  const stat = fs.existsSync(projectPath)
    ? fs.statSync(projectPath)
    : undefined;

  let configPath: string;
  if (stat?.isDirectory()) {
    configPath = path.join(projectPath, "tsconfig.json");
  } else if (stat?.isFile()) {
    configPath = projectPath;
  } else {
    throw new Error(`config not found: no tsconfig.json at ${projectPath}`);
  }

  const raw = ts.readConfigFile(configPath, ts.sys.readFile);
  if (raw.error) {
    throw new Error(
      `config error in ${configPath}: ${ts.flattenDiagnosticMessageText(raw.error.messageText, "\n")}`,
    );
  }

  const parsed = ts.parseJsonConfigFileContent(
    raw.config,
    ts.sys,
    path.dirname(configPath),
  );
  if (parsed.errors.length > 0) {
    const first = parsed.errors[0];
    throw new Error(
      `config error in ${configPath}: ${ts.flattenDiagnosticMessageText(first.messageText, "\n")}`,
    );
  }
  return { parsed, configPath };
}

function summarize(findings: ReadinessReport["findings"]): Record<string, number> {
  const summary: Record<string, number> = {};
  for (const finding of findings) {
    summary[finding.severity] = (summary[finding.severity] ?? 0) + 1;
    summary[finding.code] = (summary[finding.code] ?? 0) + 1;
  }
  return summary;
}
