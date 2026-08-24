import { test } from "node:test";
import assert from "node:assert/strict";
import { analyzeProject } from "../dist/index.js";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const fixtureDir = path.join(here, "fixtures", "node-compat");

test("compat rules classify imports by manifest status", async () => {
  const report = await analyzeProject(fixtureDir);
  const by = (code) => report.findings.filter((f) => f.code === code);

  const api = by("RT3001");
  assert.equal(api.length, 5, `expected 5 RT3001, got:\n${api.map(f => `${f.line} ${f.message}`).join("\n")}`);

  const lines = api.map((f) => f.line).sort((a, b) => a - b);
  assert.deepEqual(lines, [1, 2, 3, 4, 5]);

  assert.equal(api.filter((f) => f.severity === "error").length, 4);
  assert.equal(api.filter((f) => f.severity === "warning").length, 1);

  const misleading = api.find((f) => f.line === 2);
  assert.ok(misleading.message.includes("non-Node"), "http flagged as misleading");

  const bare = api.find((f) => f.line === 5);
  assert.ok(bare.message.includes("'node:path'"), "bare specifier suggests node: form");

  const addon = by("RT3003");
  assert.equal(addon.length, 1);
  assert.equal(addon[0].line, 7);

  // supported import (node:path) must not appear anywhere
  for (const f of report.findings) {
    assert.ok(!f.message.includes("'node:path' is"), "supported module flagged");
  }
});
