// One fixture mixing all plan groups: a native addon import (RT3003 ->
// dependencyReplacements), eval (RT2004, message says "cannot be compiled"
// -> blockedByRaya), and plain manual-edit findings (RT1001, RT3002).
import addon from "./binary.node";

const loose: any = 1;

export function run(): void {
  eval("1 + 1");
}

const legacy = require("./legacy-util");

export function main(): number {
  void addon;
  void legacy;
  return loose ? 1 : 0;
}
