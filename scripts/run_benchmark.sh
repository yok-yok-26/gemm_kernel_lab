#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
MODE=${1:-library_cublas_gemm}
mkdir -p reports/benchmark reports/benchmarks
CSV="reports/benchmark/${MODE}_latest.csv"
rm -f "$CSV"
./build/bench_gemm --mode "$MODE" --m "${M:-1024}" --n "${N:-1024}" --k "${K:-1024}" --warmup "${WARMUP:-10}" --iters "${ITERS:-50}" --csv "$CSV" | tee "reports/benchmark/${MODE}_latest.log"
cp "$CSV" "reports/benchmarks/${MODE}_latest.csv"
echo "saved $CSV"
