import type { RuleContext } from "../context.js";
import type { ReadinessFinding } from "../types.js";

/**
 * A readiness rule. Rules are pure functions of the analysis context:
 * they report facts and never mutate program state. Scoring and exit
 * thresholds live in the report/CLI layers, not here.
 */
export interface ReadinessRule {
  /** Stable diagnostic code, e.g. "RT1001". */
  readonly code: string;
  /** Short human-readable name. */
  readonly name: string;
  run(ctx: RuleContext): readonly ReadinessFinding[];
}
