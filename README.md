# GEMM Kernel Lab

这是一个用于学习 CUDA GEMM kernel 优化的实验工程。工程同时保留两条路线：

- FP32 CUDA core 路线：`user`, `v1` 到 `v9`，对比 `library_cublas_gemm`。
- WMMA / Tensor Core 路线：`v10` 到 `v19`，半精度输入、FP32 累加和输出，对比 `library_cublas_gemm_half`。

当前工程默认面向远程 CUDA 服务器：`/home/silenceduke/vscode-project/gemm_kernel_lab`。发布到 GitHub 后，路径只作为原始实验位置记录，复现时按 clone 后的本地仓库路径执行命令。

## Hardware / Software

当前实验服务器环境：

- GPU: NVIDIA GeForce RTX 5070
- Driver: 580.173.02
- CUDA Toolkit: 使用 `/usr/local/cuda`
- Build system: CMake + Ninja
- C++/CUDA standard: C++17 / CUDA C++17
- 默认 CUDA 架构：`CMAKE_CUDA_ARCHITECTURES=120`

如果你的 GPU 架构不同，请在 CMake 配置时显式设置，例如：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
```

## Directory Layout

```text
benchmark/   Release benchmark harness
include/     public declarations and CUDA error-check helpers
kernels/     user-owned CUDA kernels and launch functions
reference/   CPU reference implementation
tests/       correctness harness
scripts/     build, correctness, benchmark, sanitizer, and profiler scripts
docs/        methodology and analysis documents
reports/     selected benchmark/profiling summaries and generated plots
```

## Build

Release build:

```bash
./scripts/build_release.sh
```

Debug build:

```bash
./scripts/build_debug.sh
```

Manual equivalent:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Correctness

FP32 baseline:

```bash
./scripts/run_correctness.sh library_cublas_gemm
```

FP32 user kernel example:

```bash
BUILD=build-debug M=128 N=128 K=128 ./scripts/run_correctness.sh v6
```

The current correctness test harness directly supports `library_cublas_gemm`, `user`, and `v1` to `v9`. WMMA modes are primarily exercised through the benchmark harness with legal-shape checks and through the recorded reports under `reports/`.

## Benchmark

FP32 baseline or FP32 user versions:

```bash
M=1024 N=1024 K=1024 WARMUP=10 ITERS=50 ./scripts/run_benchmark.sh library_cublas_gemm
M=1024 N=1024 K=1024 WARMUP=10 ITERS=50 ./scripts/run_benchmark.sh v6
```

WMMA / Tensor Core versions and same-precision half baseline:

```bash
M=1024 N=1024 K=1024 WARMUP=10 ITERS=50 ./scripts/run_benchmark.sh library_cublas_gemm_half
M=1024 N=1024 K=1024 WARMUP=10 ITERS=50 ./scripts/run_benchmark.sh v16
```

Benchmark CSV files are written under `reports/benchmark/` and mirrored to `reports/benchmarks/` by `scripts/run_benchmark.sh`.

## Profiling

Nsight Compute, single launch:

```bash
M=1024 N=1024 K=1024 ./scripts/profile_ncu.sh v16
```

Nsight Systems, single launch:

```bash
M=1024 N=1024 K=1024 ./scripts/profile_nsys.sh v16
```

Profiler timing is diagnostic timing and is kept separate from Release benchmark timing.

## Comparison Rules

- FP32 modes compare against `library_cublas_gemm`.
- WMMA / half-input modes compare against `library_cublas_gemm_half`.
- `library_cublas_gemm_half` uses half A/B, FP32 C, FP32 accumulation, and Tensor Core math through `cublasGemmEx`.
- FP32-to-half conversion for WMMA modes is input preparation and is not included in the timed benchmark loop.
- Illegal WMMA shapes are reported as `SKIP`, not correctness failures.
- Release benchmark latency is used for performance comparison. NCU metrics are used for hardware explanation.

See `docs/benchmark_methodology.md` for the full methodology.

## Documentation Map

Start from:

- `docs/index.md`
- `docs/reproduce.md`
- `docs/algorithm_versions.md`
- `docs/benchmark_methodology.md`
- `docs/profiling_metrics.md`

Historical experiment reports are kept under `reports/`. Raw profiler files such as `.ncu-rep` and `.nsys-rep` are ignored by Git; publish summarized Markdown/CSV/plot artifacts instead.

## License

MIT License. See `LICENSE`.
