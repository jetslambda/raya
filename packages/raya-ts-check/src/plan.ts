import type { ReadinessFinding, ReadinessReport } from "./types.js";

/** A finding grouped into manualEdits: located and remediated by hand. */
export interface PlanManualEdit {
  code: string;
  file: string;
  line: number;
  message: string;
  remediation?: string;
}

/** A finding grouped under a plan section without remediation detail. */
export interface PlanEntry {
  code: string;
  file: string;
  message: string;
}

export interface ConversionPlanGroups {
  /** Diagnostic codes fixable by tooling; reserved and empty for now. */
  autoFixable: string[];
  manualEdits: PlanManualEdit[];
  dependencyReplacements: PlanEntry[];
  blockedByRaya: PlanEntry[];
}

export interface ConversionPlan {
  schemaVersion: 1;
  project: string;
  /** ISO 8601 UTC timestamp of plan generation. */
  generatedAt: string;
  groups: ConversionPlanGroups;
}

const BLOCKED_MESSAGE_PATTERNS = ["cannot be compiled", "not supported"];

function isDependencyReplacement(finding: ReadinessFinding): boolean {
  return (
    finding.code.startsWith("RT3001") || finding.code.startsWith("RT3003")
  );
}

function isBlockedByRaya(finding: ReadinessFinding): boolean {
  return (
    finding.code === "RT3004" ||
    BLOCKED_MESSAGE_PATTERNS.some((p) => finding.message.includes(p))
  );
}

/**
 * Regroup findings into an actionable migration plan. Every finding lands
 * in exactly one group; the dependency check runs before the blocked check,
 * so RT3001/RT3003 stay in dependencyReplacements even when their message
 * mentions a blocker. This is a view over the report: exit codes and
 * scoring are unaffected.
 */
export function buildPlan(
  report: ReadinessReport,
  now: Date = new Date(),
): ConversionPlan {
  const groups: ConversionPlanGroups = {
    autoFixable: [],
    manualEdits: [],
    dependencyReplacements: [],
    blockedByRaya: [],
  };

  for (const finding of report.findings) {
    if (isDependencyReplacement(finding)) {
      const { code, file, message } = finding;
      groups.dependencyReplacements.push({ code, file, message });
    } else if (isBlockedByRaya(finding)) {
      const { code, file, message } = finding;
      groups.blockedByRaya.push({ code, file, message });
    } else {
      const { code, file, line, message, remediation } = finding;
      groups.manualEdits.push(
        remediation === undefined
          ? { code, file, line, message }
          : { code, file, line, message, remediation },
      );
    }
  }

  return {
    schemaVersion: 1,
    project: report.project,
    generatedAt: now.toISOString(),
    groups,
  };
}
