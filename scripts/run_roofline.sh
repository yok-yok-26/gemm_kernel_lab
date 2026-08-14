#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
if [ "$#" -eq 0 ]; then
  python3 scripts/plot_roofline.py reports/benchmark/library_cublas_gemm_latest.csv
else
  python3 scripts/plot_roofline.py "$@"
fi
