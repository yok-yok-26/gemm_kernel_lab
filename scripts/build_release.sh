#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS="-O3 -lineinfo" -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j"$(nproc)"
