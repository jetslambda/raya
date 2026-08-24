import type { ReadinessRule } from "./rule.js";
import { noExplicitAny } from "./no-explicit-any.js";
import { noImplicitAny } from "./no-implicit-any.js";
import { noUnsafeTypeAssertion } from "./no-unsafe-type-assertion.js";
import { validateJsonParse } from "./validate-json-parse.js";
import { noNonNullAssertion } from "./no-non-null-assertion.js";
import { ambiguousNumber } from "./ambiguous-number.js";
import { dynamicPropertyAccess } from "./dynamic-property-access.js";
import { prototypeMutation } from "./prototype-mutation.js";
import { evalAndFunctionConstructor } from "./eval-and-function-constructor.js";

/** All registered rules, in stable diagnostic-code order. */
export const rules: readonly ReadinessRule[] = [
  noExplicitAny,
  noImplicitAny,
  noUnsafeTypeAssertion,
  validateJsonParse,
  noNonNullAssertion,

  // runtime/JIT portability (C4)
  ambiguousNumber,
  dynamicPropertyAccess,
  prototypeMutation,
  evalAndFunctionConstructor,
];
