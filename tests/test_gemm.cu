#include "cuda_check.cuh"
#include "gemm.cuh"
#include <algorithm>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

struct Args {
  std::string mode = "library_cublas_gemm";
  GemmShape shape{128, 128, 128};
  std::string report = "reports/correctness/latest.log";
};

static Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string s(argv[i]);
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); std::exit(2); }
      return argv[++i];
    };
    auto value_after_equals = [&](const char* name) -> const char* {
      const std::string prefix = std::string(name) + "=";
      return s.rfind(prefix, 0) == 0 ? s.c_str() + prefix.size() : nullptr;
    };
    if (s == "--mode") args.mode = need("--mode");
    else if (const char* v = value_after_equals("--mode")) args.mode = v;
    else if (s == "--m") args.shape.m = std::atoi(need("--m"));
    else if (const char* v = value_after_equals("--m")) args.shape.m = std::atoi(v);
    else if (s == "--n") args.shape.n = std::atoi(need("--n"));
    else if (const char* v = value_after_equals("--n")) args.shape.n = std::atoi(v);
    else if (s == "--k") args.shape.k = std::atoi(need("--k"));
    else if (const char* v = value_after_equals("--k")) args.shape.k = std::atoi(v);
    else if (s == "--report") args.report = need("--report");
    else if (const char* v = value_after_equals("--report")) args.report = v;
  }
  return args;
}

static cudaError_t launch_mode(const std::string& mode, const float* a, const float* b, float* c,
                               GemmShape shape, cudaStream_t stream) {
  if (mode == "library_cublas_gemm") return launch_cublas_gemm(a, b, c, shape, stream);
  if (mode == "user") return launch_user_gemm(a, b, c, shape, stream);
  if (mode == "v1") return launch_user_gemm_v1(a, b, c, shape, stream);
  if (mode == "v2") return launch_user_gemm_v2(a, b, c, shape, stream);
  if (mode == "v3") return launch_user_gemm_v3(a, b, c, shape, stream);
  if (mode == "v4") return launch_user_gemm_v4(a, b, c, shape, stream);
  if (mode == "v5") return launch_user_gemm_v5(a, b, c, shape, stream);
  if (mode == "v6") return launch_user_gemm_v6(a, b, c, shape, stream);
  if (mode == "v7") return launch_user_gemm_v7(a, b, c, shape, stream);
  if (mode == "v8") return launch_user_gemm_v8(a, b, c, shape, stream);
  if (mode == "v9") return launch_user_gemm_v9(a, b, c, shape, stream);
  std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
  return cudaErrorInvalidValue;
}

static cudaError_t launch_v10_mode(const float* a, const float* b, float* c,
                                   GemmShape shape, cudaStream_t stream) {
  const int64_t a_count = static_cast<int64_t>(shape.m) * shape.k;
  const int64_t b_count = static_cast<int64_t>(shape.k) * shape.n;
  half* d_a_half = nullptr;
  half* d_b_half = nullptr;
  CUDA_CHECK(cudaMallocAsync(&d_a_half, a_count * sizeof(half), stream));
  CUDA_CHECK(cudaMallocAsync(&d_b_half, b_count * sizeof(half), stream));

  cudaError_t st = launch_user_gemm_v10_convert_inputs(a, b, d_a_half, d_b_half, shape, stream);
  if (st == cudaSuccess) {
    st = launch_user_gemm_v10_wmma_only(d_a_half, d_b_half, c, shape, stream);
  }

  cudaError_t free_a = cudaFreeAsync(d_a_half, stream);
  cudaError_t free_b = cudaFreeAsync(d_b_half, stream);
  if (st != cudaSuccess) return st;
  if (free_a != cudaSuccess) return free_a;
  return free_b;
}

static cudaError_t launch_cublas_half_mode(const float* a, const float* b, float* c,
                                           GemmShape shape, cudaStream_t stream) {
  const int64_t a_count = static_cast<int64_t>(shape.m) * shape.k;
  const int64_t b_count = static_cast<int64_t>(shape.k) * shape.n;
  half* d_a_half = nullptr;
  half* d_b_half = nullptr;
  CUDA_CHECK(cudaMallocAsync(&d_a_half, a_count * sizeof(half), stream));
  CUDA_CHECK(cudaMallocAsync(&d_b_half, b_count * sizeof(half), stream));

  cudaError_t st = launch_user_gemm_v12_convert_inputs(a, b, d_a_half, d_b_half, shape, stream);
  if (st == cudaSuccess) {
    st = launch_cublas_gemm_half(d_a_half, d_b_half, c, shape, stream);
  }

  cudaError_t free_a = cudaFreeAsync(d_a_half, stream);
  cudaError_t free_b = cudaFreeAsync(d_b_half, stream);
  if (st != cudaSuccess) return st;
  if (free_a != cudaSuccess) return free_a;
  return free_b;
}

static cudaError_t launch_v11_mode(const float* a, const float* b, float* c,
                                   GemmShape shape, cudaStream_t stream) {
  const int64_t a_count = static_cast<int64_t>(shape.m) * shape.k;
  const int64_t b_count = static_cast<int64_t>(shape.k) * shape.n;
  half* d_a_half = nullptr;
  half* d_b_half = nullptr;
  CUDA_CHECK(cudaMallocAsync(&d_a_half, a_count * sizeof(half), stream));
  CUDA_CHECK(cudaMallocAsync(&d_b_half, b_count * sizeof(half), stream));

  cudaError_t st = launch_user_gemm_v11_convert_inputs(a, b, d_a_half, d_b_half, shape, stream);
  if (st == cudaSuccess) {
    st = launch_user_gemm_v11_wmma_only(d_a_half, d_b_half, c, shape, stream);
  }

  cudaError_t free_a = cudaFreeAsync(d_a_half, stream);
  cudaError_t free_b = cudaFreeAsync(d_b_half, stream);
  if (st != cudaSuccess) return st;
  if (free_a != cudaSuccess) return free_a;
  return free_b;
}

static cudaError_t launch_v12_mode(const float* a, const float* b, float* c,
                                   GemmShape shape, cudaStream_t stream) {
  const int64_t a_count = static_cast<int64_t>(shape.m) * shape.k;
  const int64_t b_count = static_cast<int64_t>(shape.k) * shape.n;
  half* d_a_half = nullptr;
  half* d_b_half = nullptr;
  CUDA_CHECK(cudaMallocAsync(&d_a_half, a_count * sizeof(half), stream));
  CUDA_CHECK(cudaMallocAsync(&d_b_half, b_count * sizeof(half), stream));

  cudaError_t st = launch_user_gemm_v12_convert_inputs(a, b, d_a_half, d_b_half, shape, stream);
  if (st == cudaSuccess) {
    st = launch_user_gemm_v12_wmma_only(d_a_half, d_b_half, c, shape, stream);
  }

  cudaError_t free_a = cudaFreeAsync(d_a_half, stream);
  cudaError_t free_b = cudaFreeAsync(d_b_half, stream);
  if (st != cudaSuccess) return st;
  if (free_a != cudaSuccess) return free_a;
  return free_b;
}

static cudaError_t launch_v13_mode(const float* a, const float* b, float* c,
                                   GemmShape shape, cudaStream_t stream) {
  const int64_t a_count = static_cast<int64_t>(shape.m) * shape.k;
  const int64_t b_count = static_cast<int64_t>(shape.k) * shape.n;
  half* d_a_half = nullptr;
  half* d_b_half = nullptr;
  CUDA_CHECK(cudaMallocAsync(&d_a_half, a_count * sizeof(half), stream));
  CUDA_CHECK(cudaMallocAsync(&d_b_half, b_count * sizeof(half), stream));

  cudaError_t st = launch_user_gemm_v13_convert_inputs(a, b, d_a_half, d_b_half, shape, stream);
  if (st == cudaSuccess) {
    st = launch_user_gemm_v13_wmma_only(d_a_half, d_b_half, c, shape, stream);
  }

  cudaError_t free_a = cudaFreeAsync(d_a_half, stream);
  cudaError_t free_b = cudaFreeAsync(d_b_half, stream);
  if (st != cudaSuccess) return st;
  if (free_a != cudaSuccess) return free_a;
  return free_b;
}

static cudaError_t launch_v14_mode(const float* a, const float* b, float* c,
                                   GemmShape shape, cudaStream_t stream) {
  const int64_t a_count = static_cast<int64_t>(shape.m) * shape.k;
  const int64_t b_count = static_cast<int64_t>(shape.k) * shape.n;
  half* d_a_half = nullptr;
  half* d_b_half = nullptr;
  CUDA_CHECK(cudaMallocAsync(&d_a_half, a_count * sizeof(half), stream));
  CUDA_CHECK(cudaMallocAsync(&d_b_half, b_count * sizeof(half), stream));

  cudaError_t st = launch_user_gemm_v14_convert_inputs(a, b, d_a_half, d_b_half, shape, stream);
  if (st == cudaSuccess) {
    st = launch_user_gemm_v14_wmma_only(d_a_half, d_b_half, c, shape, stream);
  }

  cudaError_t free_a = cudaFreeAsync(d_a_half, stream);
  cudaError_t free_b = cudaFreeAsync(d_b_half, stream);
  if (st != cudaSuccess) return st;
  if (free_a != cudaSuccess) return free_a;
  return free_b;
}


static cudaError_t launch_v15_mode(const float* a, const float* b, float* c,
                                   GemmShape shape, cudaStream_t stream) {
  const int64_t a_count = static_cast<int64_t>(shape.m) * shape.k;
  const int64_t b_count = static_cast<int64_t>(shape.k) * shape.n;
  half* d_a_half = nullptr;
  half* d_b_half = nullptr;
  CUDA_CHECK(cudaMallocAsync(&d_a_half, a_count * sizeof(half), stream));
  CUDA_CHECK(cudaMallocAsync(&d_b_half, b_count * sizeof(half), stream));

  cudaError_t st = launch_user_gemm_v15_convert_inputs(a, b, d_a_half, d_b_half, shape, stream);
  if (st == cudaSuccess) {
    st = launch_user_gemm_v15_wmma_only(d_a_half, d_b_half, c, shape, stream);
  }

  cudaError_t free_a = cudaFreeAsync(d_a_half, stream);
  cudaError_t free_b = cudaFreeAsync(d_b_half, stream);
  if (st != cudaSuccess) return st;
  if (free_a != cudaSuccess) return free_a;
  return free_b;
}

static cudaError_t launch_v16_mode(const float* a, const float* b, float* c,
                                   GemmShape shape, cudaStream_t stream) {
  const int64_t a_count = static_cast<int64_t>(shape.m) * shape.k;
  const int64_t b_count = static_cast<int64_t>(shape.k) * shape.n;
  half* d_a_half = nullptr;
  half* d_b_half = nullptr;
  CUDA_CHECK(cudaMallocAsync(&d_a_half, a_count * sizeof(half), stream));
  CUDA_CHECK(cudaMallocAsync(&d_b_half, b_count * sizeof(half), stream));

  cudaError_t st = launch_user_gemm_v16_convert_inputs(a, b, d_a_half, d_b_half, shape, stream);
  if (st == cudaSuccess) {
    st = launch_user_gemm_v16_wmma_only(d_a_half, d_b_half, c, shape, stream);
  }

  cudaError_t free_a = cudaFreeAsync(d_a_half, stream);
  cudaError_t free_b = cudaFreeAsync(d_b_half, stream);
  if (st != cudaSuccess) return st;
  if (free_a != cudaSuccess) return free_a;
  return free_b;
}

static cudaError_t launch_v17_mode(const float* a, const float* b, float* c,
                                   GemmShape shape, cudaStream_t stream) {
  const int64_t a_count = static_cast<int64_t>(shape.m) * shape.k;
  const int64_t b_count = static_cast<int64_t>(shape.k) * shape.n;
  half* d_a_half = nullptr;
  half* d_b_half = nullptr;
  CUDA_CHECK(cudaMallocAsync(&d_a_half, a_count * sizeof(half), stream));
  CUDA_CHECK(cudaMallocAsync(&d_b_half, b_count * sizeof(half), stream));

  cudaError_t st = launch_user_gemm_v17_convert_inputs(a, b, d_a_half, d_b_half, shape, stream);
  if (st == cudaSuccess) {
    st = launch_user_gemm_v17_wmma_only(d_a_half, d_b_half, c, shape, stream);
  }

  cudaError_t free_a = cudaFreeAsync(d_a_half, stream);
  cudaError_t free_b = cudaFreeAsync(d_b_half, stream);
  if (st != cudaSuccess) return st;
  if (free_a != cudaSuccess) return free_a;
  return free_b;
}

static cudaError_t launch_v18_mode(const float* a, const float* b, float* c,
                                   GemmShape shape, cudaStream_t stream) {
  const int64_t a_count = static_cast<int64_t>(shape.m) * shape.k;
  const int64_t b_count = static_cast<int64_t>(shape.k) * shape.n;
  half* d_a_half = nullptr;
  half* d_b_half = nullptr;
  CUDA_CHECK(cudaMallocAsync(&d_a_half, a_count * sizeof(half), stream));
  CUDA_CHECK(cudaMallocAsync(&d_b_half, b_count * sizeof(half), stream));

  cudaError_t st = launch_user_gemm_v18_convert_inputs(a, b, d_a_half, d_b_half, shape, stream);
  if (st == cudaSuccess) {
    st = launch_user_gemm_v18_wmma_only(d_a_half, d_b_half, c, shape, stream);
  }

  cudaError_t free_a = cudaFreeAsync(d_a_half, stream);
  cudaError_t free_b = cudaFreeAsync(d_b_half, stream);
  if (st != cudaSuccess) return st;
  if (free_a != cudaSuccess) return free_a;
  return free_b;
}

static cudaError_t launch_v19_mode(const float* a, const float* b, float* c,
                                   GemmShape shape, cudaStream_t stream) {
  const int64_t a_count = static_cast<int64_t>(shape.m) * shape.k;
  const int64_t b_count = static_cast<int64_t>(shape.k) * shape.n;
  half* d_a_half = nullptr;
  half* d_b_half = nullptr;
  CUDA_CHECK(cudaMallocAsync(&d_a_half, a_count * sizeof(half), stream));
  CUDA_CHECK(cudaMallocAsync(&d_b_half, b_count * sizeof(half), stream));

  cudaError_t st = launch_user_gemm_v19_convert_inputs(a, b, d_a_half, d_b_half, shape, stream);
  if (st == cudaSuccess) {
    st = launch_user_gemm_v19_wmma_only(d_a_half, d_b_half, c, shape, stream);
  }

  cudaError_t free_a = cudaFreeAsync(d_a_half, stream);
  cudaError_t free_b = cudaFreeAsync(d_b_half, stream);
  if (st != cudaSuccess) return st;
  if (free_a != cudaSuccess) return free_a;
  return free_b;
}

static bool is_wmma_half_mode(const std::string& mode) {
  return mode == "library_cublas_gemm_half" || mode == "v10" || mode == "v11" || mode == "v12" || mode == "v13" || mode == "v14" || mode == "v15" || mode == "v16" || mode == "v17" || mode == "v18" || mode == "v19";
}

static bool is_legal_mode_shape(const std::string& mode, GemmShape shape, std::string* reason) {
  if (mode == "v10" && (shape.m % 16 != 0 || shape.n % 16 != 0 || shape.k % 16 != 0)) {
    if (reason) *reason = "v10 WMMA learning kernel requires M, N, and K to be multiples of 16";
    return false;
  }
  if (mode == "v11" && (shape.m % 64 != 0 || shape.n % 64 != 0 || shape.k % 16 != 0)) {
    if (reason) *reason = "v11 WMMA learning kernel requires M and N to be multiples of 64, and K to be a multiple of 16";
    return false;
  }
  if (mode == "v12" && (shape.m % 64 != 0 || shape.n % 64 != 0 || shape.k % 16 != 0)) {
    if (reason) *reason = "v12 WMMA learning kernel requires M and N to be multiples of 64, and K to be a multiple of 16";
    return false;
  }
  if (mode == "v13" && (shape.m % 64 != 0 || shape.n % 64 != 0 || shape.k % 16 != 0)) {
    if (reason) *reason = "v13 WMMA learning kernel requires M and N to be multiples of 64, and K to be a multiple of 16";
    return false;
  }
  if (mode == "v14" && (shape.m % 64 != 0 || shape.n % 64 != 0 || shape.k % 16 != 0)) {
    if (reason) *reason = "v14 WMMA learning kernel requires M and N to be multiples of 64, and K to be a multiple of 16";
    return false;
  }
  if (mode == "v15" && (shape.m % 64 != 0 || shape.n % 64 != 0 || shape.k % 16 != 0)) {
    if (reason) *reason = "v15 WMMA learning kernel requires M and N to be multiples of 64, and K to be a multiple of 16";
    return false;
  }
  if (mode == "v16" && (shape.m % 64 != 0 || shape.n % 64 != 0 || shape.k % 16 != 0)) {
    if (reason) *reason = "v16 WMMA learning kernel requires M and N to be multiples of 64, and K to be a multiple of 16";
    return false;
  }
  if (mode == "v17" && (shape.m % 64 != 0 || shape.n % 64 != 0 || shape.k % 16 != 0)) {
    if (reason) *reason = "v17 WMMA learning kernel requires M and N to be multiples of 64, and K to be a multiple of 16";
    return false;
  }
  if (mode == "v18" && (shape.m % 64 != 0 || shape.n % 64 != 0 || shape.k % 16 != 0)) {
    if (reason) *reason = "v18 WMMA learning kernel requires M and N to be multiples of 64, and K to be a multiple of 16";
    return false;
  }
  if (mode == "v19" && (shape.m % 64 != 0 || shape.n % 64 != 0 || shape.k % 16 != 0)) {
    if (reason) *reason = "v19 WMMA learning kernel requires M and N to be multiples of 64, and K to be a multiple of 16";
    return false;
  }
  return true;
}

static void quantize_to_half_float(std::vector<float>& values) {
  for (float& value : values) {
    value = __half2float(__float2half_rn(value));
  }
}

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  const int64_t a_count = static_cast<int64_t>(args.shape.m) * args.shape.k;
  const int64_t b_count = static_cast<int64_t>(args.shape.k) * args.shape.n;
  const int64_t c_count = static_cast<int64_t>(args.shape.m) * args.shape.n;
  if (args.shape.m <= 0 || args.shape.n <= 0 || args.shape.k <= 0) {
    std::fprintf(stderr, "shape must be positive\n");
    return 2;
  }
  std::string skip_reason;
  if (!is_legal_mode_shape(args.mode, args.shape, &skip_reason)) {
    std::ofstream report(args.report);
    report << "status=SKIP\nmode=" << args.mode << "\n";
    report << "m=" << args.shape.m << " n=" << args.shape.n << " k=" << args.shape.k << "\n";
    report << "reason=" << skip_reason << "\n";
    std::printf("SKIP mode=%s m=%d n=%d k=%d reason=%s report=%s\n",
                args.mode.c_str(), args.shape.m, args.shape.n, args.shape.k,
                skip_reason.c_str(), args.report.c_str());
    return 0;
  }

  std::vector<float> h_a(a_count), h_b(b_count), h_c(c_count, -777.0f), h_ref(c_count);
  std::mt19937 rng(20260604);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : h_a) x = dist(rng);
  for (auto& x : h_b) x = dist(rng);

  std::vector<float> h_ref_a = h_a;
  std::vector<float> h_ref_b = h_b;
  if (is_wmma_half_mode(args.mode)) {
    quantize_to_half_float(h_ref_a);
    quantize_to_half_float(h_ref_b);
  }
  cpu_gemm_reference(h_ref_a.data(), h_ref_b.data(), h_ref.data(), args.shape);

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, h_c.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c, 0, h_c.size() * sizeof(float)));

  cudaError_t launch_status = cudaSuccess;
  if (args.mode == "library_cublas_gemm_half") {
    launch_status = launch_cublas_half_mode(d_a, d_b, d_c, args.shape, 0);
  } else if (args.mode == "v10") {
    launch_status = launch_v10_mode(d_a, d_b, d_c, args.shape, 0);
  } else if (args.mode == "v11") {
    launch_status = launch_v11_mode(d_a, d_b, d_c, args.shape, 0);
  } else if (args.mode == "v12") {
    launch_status = launch_v12_mode(d_a, d_b, d_c, args.shape, 0);
  } else if (args.mode == "v13") {
    launch_status = launch_v13_mode(d_a, d_b, d_c, args.shape, 0);
  } else if (args.mode == "v14") {
    launch_status = launch_v14_mode(d_a, d_b, d_c, args.shape, 0);
  } else if (args.mode == "v15") {
    launch_status = launch_v15_mode(d_a, d_b, d_c, args.shape, 0);
  } else if (args.mode == "v16") {
    launch_status = launch_v16_mode(d_a, d_b, d_c, args.shape, 0);
  } else if (args.mode == "v17") {
    launch_status = launch_v17_mode(d_a, d_b, d_c, args.shape, 0);
  } else if (args.mode == "v18") {
    launch_status = launch_v18_mode(d_a, d_b, d_c, args.shape, 0);
  } else if (args.mode == "v19") {
    launch_status = launch_v19_mode(d_a, d_b, d_c, args.shape, 0);
  } else {
    launch_status = launch_mode(args.mode, d_a, d_b, d_c, args.shape, 0);
  }
  if (launch_status != cudaSuccess) {
    std::ofstream report(args.report);
    report << "status=FAIL\nmode=" << args.mode << "\nreason=launch returned "
           << cudaGetErrorString(launch_status) << "\n";
    std::fprintf(stderr, "launch failed for mode %s: %s\n", args.mode.c_str(), cudaGetErrorString(launch_status));
    return 1;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float), cudaMemcpyDeviceToHost));

  GemmTolerance tol;
  if (is_wmma_half_mode(args.mode)) {
    tol.atol = 1.0e-2f;
    tol.rtol = 1.0e-2f;
  }
  float max_abs = 0.0f, max_rel = 0.0f;
  int64_t bad_count = 0, worst_idx = -1;
  for (int64_t idx = 0; idx < c_count; ++idx) {
    const float got = h_c[idx];
    const float ref = h_ref[idx];
    const float abs_err = std::abs(got - ref);
    const float rel_err = abs_err / std::max(1.0f, std::abs(ref));
    if (abs_err > max_abs || rel_err > max_rel) { max_abs = abs_err; max_rel = rel_err; worst_idx = idx; }
    if (!(abs_err <= tol.atol + tol.rtol * std::abs(ref))) ++bad_count;
  }

  std::ofstream report(args.report);
  report << "status=" << (bad_count == 0 ? "PASS" : "FAIL") << "\n";
  report << "mode=" << args.mode << "\n";
  report << "m=" << args.shape.m << " n=" << args.shape.n << " k=" << args.shape.k << "\n";
  report << "atol=" << tol.atol << " rtol=" << tol.rtol << "\n";
  report << "max_abs=" << max_abs << " max_rel=" << max_rel << " bad_count=" << bad_count
         << " worst_idx=" << worst_idx << "\n";
  if (worst_idx >= 0) {
    report << "worst_got=" << h_c[worst_idx] << " worst_ref=" << h_ref[worst_idx] << "\n";
  }

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  std::printf("%s mode=%s m=%d n=%d k=%d max_abs=%g max_rel=%g bad_count=%ld report=%s\n",
              bad_count == 0 ? "PASS" : "FAIL", args.mode.c_str(), args.shape.m, args.shape.n, args.shape.k,
              max_abs, max_rel, static_cast<long>(bad_count), args.report.c_str());
  return bad_count == 0 ? 0 : 1;
}
