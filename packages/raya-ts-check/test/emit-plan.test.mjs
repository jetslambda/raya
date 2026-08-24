import { test } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const pkgRoot = path.join(here, "..");
const mixDir = path.join(here, "fixtures", "plan-mix");

function runCli(extraArgs) {
  return spawnSync(
    process.execPath,
    ["dist/cli.js", "--project", mixDir, ...extraArgs],
    { cwd: pkgRoot, encoding: "utf8" },
  );
}

test("--emit-plan writes a parseable v1 plan and leaves normal output intact", () => {
  const outDir = mkdtempSync(path.join(tmpdir(), "raya-plan-"));
  const planFile = path.join(outDir, "plan.json");
  try {
    const res = runCli(["--format", "json", "--emit-plan", planFile]);
    assert.equal(
      res.status,
      1,
      `errors present -> exit 1 unaffected by plan emission, got ${res.status}: ${res.stderr}`,
    );

    // normal output still produced alongside the plan
    const normal = JSON.parse(res.stdout);
    assert.ok(normal.findings.length >= 4, "fixture produced findings");

    assert.equal(existsSync(planFile), true, "plan file written");
    const plan = JSON.parse(readFileSync(planFile, "utf8"));
    assert.equal(plan.schemaVersion, 1);
    assert.ok(plan.project.endsWith("plan-mix"), `project recorded: ${plan.project}`);
    assert.match(plan.generatedAt, /Z$/, "generatedAt is ISO 8601 UTC");
    assert.equal(new Date(plan.generatedAt).toISOString(), plan.generatedAt);
  } finally {
    rmSync(outDir, { recursive: true, force: true });
  }
});

test("plan groups match the code/message mapping on a mixed fixture", async () => {
  const outDir = mkdtempSync(path.join(tmpdir(), "raya-plan-"));
  const planFile = path.join(outDir, "plan.json");
  try {
    const res = runCli(["--format", "json", "--emit-plan", planFile]);
    assert.equal(res.status, 1);
    const plan = JSON.parse(readFileSync(planFile, "utf8"));

    const codes = (entries) => entries.map((e) => e.code).sort();
    // RT3003 (.node import) -> dependencyReplacements
    assert.deepEqual(codes(plan.groups.dependencyReplacements), ["RT3003"]);
    // RT2004 message contains "cannot be compiled" -> blockedByRaya
    assert.deepEqual(codes(plan.groups.blockedByRaya), ["RT2004"]);
    assert.ok(
      plan.groups.blockedByRaya[0].message.includes("cannot be compiled"),
      "blocked entry keeps the matching message",
    );
    // RT1001 + RT1002 (implicit any on the require result) + RT3002 -> manualEdits
    assert.deepEqual(codes(plan.groups.manualEdits), ["RT1001", "RT1002", "RT3002"]);
    for (const edit of plan.groups.manualEdits) {
      assert.ok(Number.isInteger(edit.line) && edit.line >= 1, `line recorded for ${edit.code}`);
      assert.equal(typeof edit.remediation, "string", `remediation carried for ${edit.code}`);
      assert.ok(edit.file.length > 0 && edit.message.length > 0);
    }
    // reserved group stays empty; every finding grouped exactly once
    assert.deepEqual(plan.groups.autoFixable, []);
    const total =
      plan.groups.autoFixable.length +
      plan.groups.manualEdits.length +
      plan.groups.dependencyReplacements.length +
      plan.groups.blockedByRaya.length;
    const report = JSON.parse(res.stdout);
    assert.equal(total, report.findings.length, "no finding dropped or duplicated");
  } finally {
    rmSync(outDir, { recursive: true, force: true });
  }
});

test("buildPlan: message-pattern blockers and dependency precedence", async () => {
  const { buildPlan } = await import("../dist/plan.js");
  const finding = (code, message) => ({
    code, severity: "warning", stage: "runtime",
    file: "src/a.ts", start: 0, length: 1, line: 1, column: 1,
    message, rule: "test",
  });
  const plan = buildPlan({
    schemaVersion: 1,
    project: "/tmp/p",
    filesAnalyzed: 1,
    findings: [
      // not RT3001/RT3003/RT3004 but message matches -> blockedByRaya
      finding("RT2005", "shape 'not supported' by layout analysis"),
      // RT3001 with blocker-ish message still wins the dependency check first
      finding("RT3001", "'fs' is not supported in Raya"),
      finding("RT3003", "native addons cannot be loaded"),
      finding("RT1005", "plain manual edit"),
    ],
    summary: {},
  }, new Date("2026-01-01T00:00:00.000Z"));

  assert.equal(plan.generatedAt, "2026-01-01T00:00:00.000Z");
  assert.deepEqual(plan.groups.blockedByRaya.map((e) => e.code), ["RT2005"]);
  assert.deepEqual(plan.groups.dependencyReplacements.map((e) => e.code).sort(), ["RT3001", "RT3003"]);
  assert.deepEqual(plan.groups.manualEdits.map((e) => e.code), ["RT1005"]);
});
