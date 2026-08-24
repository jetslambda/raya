import { test } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { analyzeProject } from "../dist/index.js";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const fixtureDir = path.join(here, "fixtures", "any-usage");
const emptyDir = path.join(here, "fixtures", "empty-project");

test("rules fire with expected codes and locations", async () => {
  const report = await analyzeProject(fixtureDir);
  const codes = new Set(report.findings.map((f) => f.code));

  for (const expected of ["RT1001", "RT1002", "RT1003", "RT1004", "RT1005"]) {
    assert.ok(codes.has(expected), `missing ${expected}; got ${[...codes].join(",")}`);
  }

  const rt1001 = report.findings.filter((f) => f.code === "RT1001");
  assert.equal(rt1001.length, 1, "exactly one explicit any");
  assert.equal(rt1001[0].line, 6, "explicit any on line 6");

  const rt1005 = report.findings.filter((f) => f.code === "RT1005");
  assert.equal(rt1005.length, 1);
  assert.equal(rt1005[0].line, 15, "non-null assertion on line 15");

  const rt1004 = report.findings.filter((f) => f.code === "RT1004");
  assert.equal(rt1004.length, 1, "only the asserted JSON.parse is flagged");
  assert.equal(rt1004[0].line, 11);

  const rt1003 = report.findings.filter((f) => f.code === "RT1003");
  assert.equal(rt1003.length, 1, "assertion from any is an error");
  assert.equal(rt1003[0].severity, "error");

  const rt1002 = report.findings.filter((f) => f.code === "RT1002");
  assert.ok(rt1002.length >= 3, "two params + one implicit-any variable");

  // every finding carries location + rule attribution
  for (const f of report.findings) {
    assert.ok(f.line >= 1 && f.column >= 1 && f.start >= 0 && f.rule.length > 0);
  }
});

test("findings are deterministically ordered", async () => {
  const a = await analyzeProject(fixtureDir);
  const b = await analyzeProject(fixtureDir);
  assert.equal(JSON.stringify(a), JSON.stringify(b));
});

test("clean project still yields zero findings after rules are active", async () => {
  const report = await analyzeProject(emptyDir);
  assert.deepEqual(report.findings, []);
});

test("cli exits 1 when error-severity findings meet --fail-on error", () => {
  const res = spawnSync(
    process.execPath,
    ["dist/cli.js", "--project", fixtureDir, "--format", "json", "--fail-on", "error"],
    { cwd: path.join(here, ".."), encoding: "utf8" },
  );
  assert.equal(res.status, 1, `expected exit 1, got ${res.status}: ${res.stderr}`);
  const report = JSON.parse(res.stdout);
  assert.equal(report.schemaVersion, 1);
});
