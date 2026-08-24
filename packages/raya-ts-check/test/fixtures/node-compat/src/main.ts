import { createHash } from "node:crypto";      // partial -> warning
import { createServer } from "node:http";       // misleading -> error
import { Script } from "node:vm";               // stub -> error
import { gzipSync } from "node:zlib";           // absent -> error
import { join } from "path";                    // bare builtin -> error
import { resolve } from "node:path";            // supported -> nothing
import native from "./binding.node";            // RT3003 -> error

export function hash(s: string): string {
  return createHash("sha256").update(s).digest("hex");
}

export function p(a: string, b: string): string {
  return resolve(join(a, b));
}
