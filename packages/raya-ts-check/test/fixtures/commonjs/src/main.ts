// CommonJS interop surface kept behind one module boundary.
const legacy: Record<string, unknown> = require("./legacy-util");

function combine(parts: string[]): string {
  return parts.join("+");
}

module.exports = { combine };
exports.combine = combine;
