import type { ReadinessFinding, ReadinessReport } from "./types.js";

export type CategoryStatus = "pass" | "warn" | "fail" | "likely" | "partial" | "unlikely";

export interface ReportCategories {
  /** stage=type findings */
  typeSafety: CategoryStatus;
  /** stage=syntax findings (no rules yet; reserved) */
  syntaxCompatibility: CategoryStatus;
  /** stage=runtime + compat findings */
  runtimeCompatibility: CategoryStatus;
  /** stage=jit findings */
  jitSpecialization: CategoryStatus;
}

function worst(findings: readonly ReadinessFinding[]): CategoryStatus {
  if (findings.some((f) => f.severity === "error")) return "fail";
  if (findings.some((f) => f.severity === "warning")) return "warn";
  return "pass";
}

/**
 * Roll findings up into the four conversion-gate categories. A numeric
 * score would hide an error behind averages; statuses do not.
 */
export function categorize(report: ReadinessReport): ReportCategories {
  const byStage = (stage: string) => report.findings.filter((f) => f.stage === stage);

  const jit = byStage("jit");
  let jitStatus: CategoryStatus = "likely";
  if (jit.some((f) => f.severity === "error")) jitStatus = "unlikely";
  else if (jit.length > 0) jitStatus = "partial";

  return {
    typeSafety: worst(byStage("type")),
    syntaxCompatibility: worst(byStage("syntax")),
    runtimeCompatibility: worst([...byStage("runtime"), ...byStage("compat")]),
    jitSpecialization: jitStatus,
  };
}
