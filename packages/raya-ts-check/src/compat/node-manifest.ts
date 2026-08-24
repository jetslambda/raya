import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

export type NodeApiStatus = "supported" | "partial" | "stub" | "misleading";

export interface NodeModuleEntry {
  status: NodeApiStatus;
  note?: string;
}

export interface NodeManifest {
  schemaVersion: number;
  modules: Record<string, NodeModuleEntry>;
}

let cached: NodeManifest | undefined;

/** Load the checked-in compatibility manifest (cached). */
export function loadNodeManifest(): NodeManifest {
  if (cached) return cached;
  const manifestPath = path.join(
    path.dirname(fileURLToPath(import.meta.url)),
    "../../compat/node-manifest.json",
  );
  cached = JSON.parse(fs.readFileSync(manifestPath, "utf8")) as NodeManifest;
  return cached;
}

/** Test seam: replace the active manifest. */
export function setNodeManifest(manifest: NodeManifest): void {
  cached = manifest;
}
