#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
MODE=${1:-user}
mkdir -p reports/ncu
if ! command -v ncu >/dev/null 2>&1; then echo "ncu not found in PATH" | tee reports/ncu/latest.log; exit 127; fi
ncu --force-overwrite --target-processes all --set full --export "reports/ncu/${MODE}_latest" ./build/bench_gemm --mode "$MODE" --m "${M:-1024}" --n "${N:-1024}" --k "${K:-1024}" --single-launch --csv "reports/ncu/${MODE}_single_launch.csv" 2>&1 | tee "reports/ncu/${MODE}_latest.log"
