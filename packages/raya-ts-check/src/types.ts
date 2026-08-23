/** Severity of a readiness finding. */
export type ReadinessSeverity = "info" | "warning" | "error";

/** Pipeline stage a finding belongs to. */
export type ReadinessStage = "syntax" | "type" | "runtime" | "jit" | "compat";

/** One actionable report item, located in source. */
export interface ReadinessFinding {
  /** Stable diagnostic code, e.g. RT1001. */
  code: string;
  severity: ReadinessSeverity;
  stage: ReadinessStage;
  /** Project-relative file path. */
  file: string;
  /** Start offset in the file (UTF-16 code units). */
  start: number;
  length: number;
  line: number;
  column: number;
  message: string;
  remediation?: string;
  /** Rule identifier that produced this finding. */
  rule: string;
}

/** Full analysis result for one TypeScript project. */
export interface ReadinessReport {
  schemaVersion: 1;
  project: string;
  filesAnalyzed: number;
  findings: ReadinessFinding[];
  summary: Record<string, number>;
}
