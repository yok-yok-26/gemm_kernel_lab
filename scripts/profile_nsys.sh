#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
MODE=${1:-user}
mkdir -p reports/nsys
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt -o "reports/nsys/${MODE}_latest" ./build/bench_gemm --mode "$MODE" --m "${M:-1024}" --n "${N:-1024}" --k "${K:-1024}" --single-launch --csv "reports/nsys/${MODE}_single_launch.csv"
