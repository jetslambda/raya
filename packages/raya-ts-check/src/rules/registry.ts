import type { ReadinessRule } from "./rule.js";
import { noExplicitAny } from "./no-explicit-any.js";
import { noImplicitAny } from "./no-implicit-any.js";
import { noUnsafeTypeAssertion } from "./no-unsafe-type-assertion.js";
import { validateJsonParse } from "./validate-json-parse.js";
import { noNonNullAssertion } from "./no-non-null-assertion.js";

/** All registered rules, in stable diagnostic-code order. */
export const rules: readonly ReadinessRule[] = [
  noExplicitAny,
  noImplicitAny,
  noUnsafeTypeAssertion,
  validateJsonParse,
  noNonNullAssertion,
];
