import ts from "typescript";
import type { ReadinessFinding } from "./types.js";
import { makeFinding, type FindingInfo } from "./source-location.js";

/** Everything a rule needs to inspect the program and report findings. */
export interface RuleContext {
  program: ts.Program;
  checker: ts.TypeChecker;
  options: ts.CompilerOptions;
  /** Absolute directory containing tsconfig.json; findings are relative to it. */
  projectDir: string;
  sourceFiles: readonly ts.SourceFile[];
  makeFinding(sourceFile: ts.SourceFile, node: ts.Node, info: FindingInfo): ReadinessFinding;
}

export function createContext(
  program: ts.Program,
  projectDir: string,
): RuleContext {
  return {
    program,
    checker: program.getTypeChecker(),
    options: program.getCompilerOptions(),
    projectDir,
    sourceFiles: program.getSourceFiles().filter((sf) => !sf.isDeclarationFile),
    makeFinding: (sf, node, info) => makeFinding(projectDir, sf, node, info),
  };
}
