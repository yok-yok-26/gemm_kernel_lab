#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
MODE=${1:-user}
mkdir -p reports/memcheck
if ! command -v compute-sanitizer >/dev/null 2>&1; then echo "compute-sanitizer not found in PATH" | tee reports/memcheck/latest.log; exit 127; fi
compute-sanitizer --tool memcheck --error-exitcode 1 --log-file reports/memcheck/${MODE}_latest.log ./build-debug/test_gemm --mode "$MODE" --m "${M:-128}" --n "${N:-128}" --k "${K:-128}" --report reports/memcheck/${MODE}_correctness_latest.log
