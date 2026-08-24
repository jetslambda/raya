import type { ReadinessFinding, ReadinessReport } from "../types.js";

/** SARIF 2.1.0 result levels; info findings downgrade to note. */
const LEVELS: Record<ReadinessFinding["severity"], "error" | "warning" | "note"> = {
  error: "error",
  warning: "warning",
  info: "note",
};

interface SarifRule {
  id: string;
  name: string;
  shortDescription: { text: string };
}

interface SarifResult {
  ruleId: string;
  level: string;
  message: { text: string };
  locations: Array<{
    physicalLocation: {
      artifactLocation: { uri: string };
      region: { startLine: number; startColumn: number };
    };
  }>;
}

interface SarifLog {
  $schema: string;
  version: "2.1.0";
  runs: Array<{
    tool: {
      driver: {
        name: string;
        informationUri: string;
        rules: SarifRule[];
      };
    };
    results: SarifResult[];
  }>;
}

/** SARIF consumers expect URIs with forward slashes. */
function toUri(file: string): string {
  return file.split("\\").join("/");
}

export function renderSarif(report: ReadinessReport): string {
  const rules = new Map<string, SarifRule>();
  for (const f of report.findings) {
    if (!rules.has(f.code)) {
      rules.set(f.code, {
        id: f.code,
        name: f.rule,
        shortDescription: { text: f.message },
      });
    }
  }

  const results: SarifResult[] = report.findings.map((f) => ({
    ruleId: f.code,
    level: LEVELS[f.severity],
    message: {
      text: f.remediation !== undefined ? `${f.message}\n${f.remediation}` : f.message,
    },
    locations: [
      {
        physicalLocation: {
          artifactLocation: { uri: toUri(f.file) },
          region: { startLine: f.line, startColumn: f.column },
        },
      },
    ],
  }));

  const log: SarifLog = {
    $schema: "https://json.schemastore.org/sarif-2.1.0.json",
    version: "2.1.0",
    runs: [
      {
        tool: {
          driver: {
            name: "raya-ts-check",
            informationUri: "https://github.com/jetslambda/raya",
            rules: [...rules.values()],
          },
        },
        results,
      },
    ],
  };

  return JSON.stringify(log, null, 2);
}
