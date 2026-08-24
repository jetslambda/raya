import * as path from "node:path";
import ts from "typescript";
import type { ReadinessFinding, ReadinessSeverity, ReadinessStage } from "./types.js";

/** Information a rule supplies; location fields are derived from the node. */
export interface FindingInfo {
  code: string;
  severity: ReadinessSeverity;
  stage: ReadinessStage;
  message: string;
  remediation?: string;
  rule: string;
}

/** Project-relative POSIX-style path for a source file. */
export function relativeFile(projectDir: string, sourceFile: ts.SourceFile): string {
  return path.relative(projectDir, sourceFile.fileName).split(path.sep).join("/");
}

/** Build a located finding from a syntax node. */
export function makeFinding(
  projectDir: string,
  sourceFile: ts.SourceFile,
  node: ts.Node,
  info: FindingInfo,
): ReadinessFinding {
  const start = node.getStart(sourceFile);
  const { line, character } = sourceFile.getLineAndCharacterOfPosition(start);
  return {
    ...info,
    file: relativeFile(projectDir, sourceFile),
    start,
    length: node.getEnd() - start,
    line: line + 1,
    column: character + 1,
  };
}

/** Depth-first collection of nodes matching a predicate. */
export function collectNodes(root: ts.Node, predicate: (node: ts.Node) => boolean): ts.Node[] {
  const out: ts.Node[] = [];
  function visit(node: ts.Node): void {
    if (predicate(node)) {
      out.push(node);
    }
    ts.forEachChild(node, visit);
  }
  ts.forEachChild(root, visit);
  return out;
}
