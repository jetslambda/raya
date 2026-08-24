import { test } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { analyzeProject } from "../dist/index.js";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const fixtures = path.join(here, "fixtures");
const metaDir = path.join(fixtures, "metaprogramming");
const cjsDir = path.join(fixtures, "commonjs");
const syntaxDir = path.join(fixtures, "ts-syntax");
const emptyDir = path.join(fixtures, "empty-project");

test("RT2005 flags new Proxy and Reflect.* metaprogramming", async () => {
  const report = await analyzeProject(metaDir);
  const hits = report.findings.filter((f) => f.code === "RT2005");

  assert.equal(hits.length, 4, `expected 4 RT2005 (Proxy + get/set/has), got:\n${hits.map((f) => `${f.line} ${f.message}`).join("\n")}`);
  assert.deepEqual(hits.map((f) => f.line), [3, 5, 6, 7]);

  for (const h of hits) {
    assert.equal(h.stage, "jit");
    assert.equal(h.severity, "warning");
    assert.equal(h.rule, "proxy-reflect-usage");
    assert.ok(h.message.includes("escapes static property analysis"));
  }

  // fixture stays clean otherwise
  assert.equal(report.findings.length, 4);
});

test("RT3002 flags require() and module.exports/exports assignments", async () => {
  const report = await analyzeProject(cjsDir);
  const hits = report.findings.filter((f) => f.code === "RT3002");

  assert.equal(hits.length, 3, `expected 3 RT3002, got:\n${hits.map((f) => `${f.line} ${f.message}`).join("\n")}`);
  assert.deepEqual(hits.map((f) => f.line), [2, 8, 9]);

  for (const h of hits) {
    assert.equal(h.stage, "compat");
    assert.equal(h.severity, "warning");
    assert.equal(h.rule, "commonjs-boundary");
  }

  const requireHit = hits[0];
  assert.ok(requireHit.message.includes("'./legacy-util'"), "require specifier named in message");

  assert.equal(report.findings.length, 3);
});

test("RT3004 flags decorators, parameter properties, and ambient declarations", async () => {
  const report = await analyzeProject(syntaxDir);
  const hits = report.findings.filter((f) => f.code === "RT3004");

  assert.equal(hits.length, 3, `expected 3 RT3004, got:\n${hits.map((f) => `${f.line} ${f.message}`).join("\n")}`);
  assert.deepEqual(hits.map((f) => f.line), [8, 12, 17]);

  for (const h of hits) {
    assert.equal(h.stage, "syntax");
    assert.equal(h.severity, "error");
    assert.equal(h.rule, "unsupported-typescript-syntax");
  }

  // plain class fields (line 10) and non-ambient namespaces stay unflagged:
  // nothing else in the fixture may fire at all.
  assert.equal(report.findings.length, 3);
});

test("new-rule findings are deterministic across two runs", async () => {
  for (const dir of [metaDir, cjsDir, syntaxDir]) {
    const a = await analyzeProject(dir);
    const b = await analyzeProject(dir);
    assert.equal(JSON.stringify(a), JSON.stringify(b), `${dir} differs between runs`);
  }
});

test("cli --format sarif emits SARIF 2.1.0 matching the findings", async () => {
  const res = spawnSync(
    process.execPath,
    ["dist/cli.js", "--project", syntaxDir, "--format", "sarif"],
    { cwd: path.join(here, ".."), encoding: "utf8" },
  );
  assert.equal(res.status, 1, `errors present -> exit 1, got ${res.status}: ${res.stderr}`);

  const log = JSON.parse(res.stdout);
  assert.equal(log.$schema, "https://json.schemastore.org/sarif-2.1.0.json");
  assert.equal(log.version, "2.1.0");

  const run = log.runs[0];
  assert.equal(run.tool.driver.name, "raya-ts-check");
  assert.equal(run.tool.driver.informationUri, "https://github.com/jetslambda/raya");

  const report = await analyzeProject(syntaxDir);
  assert.equal(run.results.length, report.findings.length);

  for (const result of run.results) {
    assert.ok(result.ruleId, "every result has ruleId");
    assert.ok(["error", "warning", "note"].includes(result.level), `valid level: ${result.level}`);
    const loc = result.locations[0].physicalLocation;
    assert.equal(loc.artifactLocation.uri, "src/main.ts");
    assert.ok(loc.region.startLine >= 1);
    assert.ok(loc.region.startColumn >= 1);
    assert.ok(result.message.text.length > 0);
  }

  const ruleIds = run.tool.driver.rules.map((r) => r.id);
  assert.equal(new Set(ruleIds).size, ruleIds.length, "rules deduplicated by code");
  assert.deepEqual(
    [...ruleIds].sort(),
    [...new Set(report.findings.map((f) => f.code))].sort(),
  );
});

test("sarif on a clean project exits 0 with zero results", () => {
  const res = spawnSync(
    process.execPath,
    ["dist/cli.js", "--project", emptyDir, "--format", "sarif"],
    { cwd: path.join(here, ".."), encoding: "utf8" },
  );
  assert.equal(res.status, 0, `clean project -> exit 0, got ${res.status}: ${res.stderr}`);

  const log = JSON.parse(res.stdout);
  assert.equal(log.version, "2.1.0");
  assert.deepEqual(log.runs[0].results, []);
});

test("clean-project fixture still yields zero findings after new rules", async () => {
  const report = await analyzeProject(emptyDir);
  assert.deepEqual(report.findings, []);
});
