#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
MODE=${1:-library_cublas_gemm}
BUILD=${BUILD:-build}
mkdir -p reports/correctness
LOG="reports/correctness/${MODE}_latest.log"
./${BUILD}/test_gemm --mode "$MODE" --m "${M:-128}" --n "${N:-128}" --k "${K:-128}" --report "$LOG" | tee "reports/correctness/${MODE}_stdout_latest.log"
echo "saved $LOG"
