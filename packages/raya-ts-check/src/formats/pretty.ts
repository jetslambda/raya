import type { ReadinessReport } from "../types.js";
import type { ReportCategories } from "../report.js";

const MARKS: Record<string, string> = {
  pass: "pass",
  warn: "warn",
  fail: "FAIL",
  likely: "likely",
  partial: "partial",
  unlikely: "unlikely",
};

export function renderPretty(report: ReadinessReport, categories: ReportCategories): string {
  const lines: string[] = [];
  lines.push(`raya-ts-check — ${report.project}`);
  lines.push(`files analyzed: ${report.filesAnalyzed}, findings: ${report.findings.length}`);
  lines.push("");
  lines.push("conversion gates:");
  lines.push(`  type safety            ${MARKS[categories.typeSafety]}`);
  lines.push(`  raya syntax            ${MARKS[categories.syntaxCompatibility]}`);
  lines.push(`  runtime compatibility  ${MARKS[categories.runtimeCompatibility]}`);
  lines.push(`  jit specialization     ${MARKS[categories.jitSpecialization]}`);

  if (report.findings.length > 0) {
    lines.push("");
    for (const f of report.findings) {
      lines.push(`${f.file}:${f.line}:${f.column}  ${f.code} [${f.severity}] ${f.message}`);
      if (f.remediation) lines.push(`    fix: ${f.remediation}`);
    }
  }
  return lines.join("\n");
}
