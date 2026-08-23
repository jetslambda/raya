import { test } from "node:test";
import assert from "node:assert/strict";
import { analyzeProject } from "../dist/index.js";
import { fileURLToPath } from "node:url";
import path from "node:path";

const fixtureDir = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "fixtures",
  "empty-project",
);

test("empty valid project produces a v1 report with no findings", async () => {
  const report = await analyzeProject(fixtureDir);
  assert.equal(report.schemaVersion, 1);
  assert.ok(report.filesAnalyzed >= 1, "expected at least one source file");
  assert.deepEqual(report.findings, []);
});

test("missing tsconfig fails with a config error", async () => {
  await assert.rejects(
    () => analyzeProject(path.join(fixtureDir, "does-not-exist")),
    /config/,
  );
});
