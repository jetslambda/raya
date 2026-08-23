import * as fs from "node:fs";
import * as path from "node:path";
import ts from "typescript";
import type { ReadinessFinding, ReadinessReport } from "./types.js";

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

  const config = readConfig(absolute);
  const program = ts.createProgram(config.fileNames, config.options);
  const _checker = program.getTypeChecker(); // retained: rules consume it from C3 onward

  const findings: ReadinessFinding[] = []; // no rules registered yet (C1)

  const filesAnalyzed = program
    .getSourceFiles()
    .filter((sf) => !sf.isDeclarationFile).length;

  return {
    schemaVersion: 1,
    project: absolute,
    filesAnalyzed,
    findings,
    summary: summarize(findings),
  };
}

function readConfig(projectPath: string): ts.ParsedCommandLine {
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
  return parsed;
}

function summarize(findings: ReadinessReport["findings"]): Record<string, number> {
  const summary: Record<string, number> = {};
  for (const finding of findings) {
    summary[finding.severity] = (summary[finding.severity] ?? 0) + 1;
    summary[finding.code] = (summary[finding.code] ?? 0) + 1;
  }
  return summary;
}
