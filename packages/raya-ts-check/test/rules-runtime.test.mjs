import { test } from "node:test";
import assert from "node:assert/strict";
import { analyzeProject } from "../dist/index.js";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const emptyDir = path.join(here, "fixtures", "empty-project");
const fixtureDir = path.join(here, "fixtures", "runtime-hazards");

test("runtime hazard rules fire with expected codes and lines", async () => {
  const report = await analyzeProject(fixtureDir);
  const by = (code) => report.findings.filter((f) => f.code === code);

  const proto = by("RT2003");
  assert.equal(proto.length, 1, `expected 1 prototype mutation, got ${proto.length}`);
  assert.equal(proto[0].line, 7);

  const evals = by("RT2004");
  assert.equal(evals.length, 2, "eval + new Function");
  assert.deepEqual(evals.map((f) => f.line).sort(), [11, 16]);

  const dyn = by("RT2002");
  assert.equal(dyn.length, 1);
  assert.equal(dyn[0].line, 21);

  const amb = by("RT2001");
  assert.equal(amb.length, 1, "scale() is arithmetic-heavy with plain number");
  assert.equal(amb[0].severity, "info");
  assert.equal(amb[0].stage, "jit");

  // numeric indexing must NOT be flagged anywhere in this fixture
  for (const f of report.findings) {
    if (f.code === "RT2002") {
      assert.ok(!f.message.includes("i <"), "loop index should not be flagged");
    }
  }
});

test("all findings remain deterministically ordered", async () => {
  const a = await analyzeProject(fixtureDir);
  const b = await analyzeProject(fixtureDir);
  assert.equal(JSON.stringify(a), JSON.stringify(b));
});

test("category rollups reflect severity per gate", async () => {
  const { categorize } = await import("../dist/report.js");
  const report = await analyzeProject(fixtureDir);
  const cats = categorize(report);

  assert.equal(cats.typeSafety, "warn"); // RT1001/RT1002 warnings present
  assert.equal(cats.runtimeCompatibility, "fail"); // RT2004 errors
  assert.equal(cats.jitSpecialization, "unlikely"); // RT2003 jit-stage error

  const empty = await analyzeProject(emptyDir);
  const emptyCats = categorize(empty);
  assert.equal(emptyCats.typeSafety, "pass");
  assert.equal(emptyCats.jitSpecialization, "likely");
});

test("pretty renderer includes gates and located findings", async () => {
  const { categorize } = await import("../dist/report.js");
  const { renderPretty } = await import("../dist/formats/pretty.js");
  const report = await analyzeProject(fixtureDir);
  const text = renderPretty(report, categorize(report));
  assert.ok(text.includes("conversion gates"));
  assert.ok(text.includes("main.ts:7:"));
  assert.ok(text.includes("FAIL"));
});
