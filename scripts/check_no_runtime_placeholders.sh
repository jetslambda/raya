#!/usr/bin/env bash
# Phase 0 gate (S3): known semantic placeholders must not re-enter runtime code.
# Each entry is a literal string that previously marked incorrect JIT/AOT/VM
# behavior. Extend the allowlist only with a link to an issue that tracks the
# legitimate occurrence.
set -uo pipefail
cd "$(dirname "$0")/.."

PATTERNS=(
  "just pass through as multiply"
  "push 0 as a placeholder"
  "placeholder — would call runtime"
  "would call runtime fmod"
  "GetArgCount support in JIT"
  "always returns false"
)

SCOPE=(
  crates/raya-engine/src/jit
  crates/raya-engine/src/aot
  crates/raya-engine/src/vm
)

fail=0
for pat in "${PATTERNS[@]}"; do
  hits=$(grep -RnF -- "$pat" "${SCOPE[@]}" 2>/dev/null || true)
  if [[ -n "$hits" ]]; then
    echo "PLACEHOLDER FOUND for pattern: $pat"
    echo "$hits"
    fail=1
  fi
done

if [[ $fail -ne 0 ]]; then
  echo ""
  echo "Runtime placeholder gate failed. Remove the placeholder or fix the"
  echo "semantics before merging. Do not add these strings back."
  exit 1
fi

echo "No runtime placeholders found."
