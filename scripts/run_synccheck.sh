#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
MODE=${1:-user}
mkdir -p reports/synccheck
if ! command -v compute-sanitizer >/dev/null 2>&1; then echo "compute-sanitizer not found in PATH" | tee reports/synccheck/latest.log; exit 127; fi
compute-sanitizer --tool synccheck --error-exitcode 1 --log-file reports/synccheck/${MODE}_latest.log ./build-debug/test_gemm --mode "$MODE" --m "${M:-128}" --n "${N:-128}" --k "${K:-128}" --report reports/synccheck/${MODE}_correctness_latest.log
