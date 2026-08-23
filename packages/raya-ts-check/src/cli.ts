#!/usr/bin/env node
import { parseArgs } from "node:util";
import { analyzeProject } from "./index.js";
import type { ReadinessSeverity } from "./types.js";

interface CliOptions {
  project: string;
  format: "pretty" | "json";
  failOn: ReadinessSeverity;
}

const SEVERITY_ORDER: Record<ReadinessSeverity, number> = {
  info: 0,
  warning: 1,
  error: 2,
};

function parseOptions(): CliOptions {
  const { values } = parseArgs({
    options: {
      project: { type: "string" },
      format: { type: "string", default: "pretty" },
      "fail-on": { type: "string", default: "error" },
    },
  });

  if (!values.project) {
    console.error("usage: raya-ts-check --project <tsconfig-or-dir> [--format pretty|json] [--fail-on info|warning|error]");
    process.exit(2);
  }

  const format = values.format as CliOptions["format"];
  if (format !== "pretty" && format !== "json") {
    console.error(`invalid --format: ${values.format}`);
    process.exit(2);
  }

  const failOn = values["fail-on"] as ReadinessSeverity;
  if (!(failOn in SEVERITY_ORDER)) {
    console.error(`invalid --fail-on: ${values["fail-on"]}`);
    process.exit(2);
  }

  return { project: values.project, format, failOn };
}

async function main(): Promise<void> {
  const options = parseOptions();

  let report;
  try {
    report = await analyzeProject(options.project);
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(2);
  }

  if (options.format === "json") {
    console.log(JSON.stringify(report, null, 2));
  } else {
    console.log(`project:         ${report.project}`);
    console.log(`files analyzed:  ${report.filesAnalyzed}`);
    console.log(`findings:        ${report.findings.length}`);
  }

  const threshold = SEVERITY_ORDER[options.failOn];
  const failing = report.findings.some(
    (f) => SEVERITY_ORDER[f.severity] >= threshold,
  );
  process.exit(failing ? 1 : 0);
}

main();
