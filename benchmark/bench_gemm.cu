#include "cuda_check.cuh"
#include "gemm.cuh"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

struct Args {
  std::string mode = "library_cublas_gemm";
  GemmShape shape{1024, 1024, 1024};
  int warmup = 10;
  int iters = 50;
  bool single_launch = false;
  std::string csv = "reports/benchmark/latest.csv";
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
    else if (s == "--warmup") args.warmup = std::atoi(need("--warmup"));
    else if (const char* v = value_after_equals("--warmup")) args.warmup = std::atoi(v);
    else if (s == "--iters") args.iters = std::atoi(need("--iters"));
    else if (const char* v = value_after_equals("--iters")) args.iters = std::atoi(v);
    else if (s == "--csv") args.csv = need("--csv");
    else if (const char* v = value_after_equals("--csv")) args.csv = v;
    else if (s == "--single-launch") args.single_launch = true;
  }
  if (args.single_launch) { args.warmup = 0; args.iters = 1; }
  return args;
}

class CublasGemmRunner {
 public:
  explicit CublasGemmRunner(cudaStream_t stream) {
    CUBLAS_CHECK(cublasCreate(&handle_));
    CUBLAS_CHECK(cublasSetStream(handle_, stream));
  }
  ~CublasGemmRunner() { cublasDestroy(handle_); }
  cudaError_t run(const float* a, const float* b, float* c, GemmShape shape) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                    shape.n, shape.m, shape.k,
                                    &alpha,
                                    b, shape.n,
                                    a, shape.k,
                                    &beta,
                                    c, shape.n);
    return st == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
  }
 private:
  cublasHandle_t handle_{};
};

class CublasHalfGemmRunner {
 public:
  explicit CublasHalfGemmRunner(cudaStream_t stream) {
    CUBLAS_CHECK(cublasCreate(&handle_));
    CUBLAS_CHECK(cublasSetStream(handle_, stream));
    CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH));
  }
  ~CublasHalfGemmRunner() { cublasDestroy(handle_); }
  cudaError_t run(const half* a, const half* b, float* c, GemmShape shape) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(handle_,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     shape.n, shape.m, shape.k,
                                     &alpha,
                                     b, CUDA_R_16F, shape.n,
                                     a, CUDA_R_16F, shape.k,
                                     &beta,
                                     c, CUDA_R_32F, shape.n,
                                     CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return st == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
  }
 private:
  cublasHandle_t handle_{};
};

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  const int64_t a_count = static_cast<int64_t>(args.shape.m) * args.shape.k;
  const int64_t b_count = static_cast<int64_t>(args.shape.k) * args.shape.n;
  const int64_t c_count = static_cast<int64_t>(args.shape.m) * args.shape.n;
  std::vector<float> h_a(a_count), h_b(b_count);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : h_a) x = dist(rng);
  for (auto& x : h_b) x = dist(rng);

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  half *d_a_wmma_half = nullptr, *d_b_wmma_half = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, c_count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(float), cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  CublasGemmRunner cublas_runner(stream);
  CublasHalfGemmRunner cublas_half_runner(stream);
  const bool is_cublas_half_mode = args.mode == "library_cublas_gemm_half";
  const bool is_wmma_mode = args.mode == "v10" || args.mode == "v11" || args.mode == "v12" || args.mode == "v13" || args.mode == "v14" || args.mode == "v15" || args.mode == "v16" || args.mode == "v17" || args.mode == "v18" || args.mode == "v19";
  const bool is_v10_legal = args.mode == "v10" &&
      (args.shape.m % 16 == 0 && args.shape.n % 16 == 0 && args.shape.k % 16 == 0);
  const bool is_v11_legal = args.mode == "v11" &&
      (args.shape.m % 64 == 0 && args.shape.n % 64 == 0 && args.shape.k % 16 == 0);
  const bool is_v12_legal = args.mode == "v12" &&
      (args.shape.m % 64 == 0 && args.shape.n % 64 == 0 && args.shape.k % 16 == 0);
  const bool is_v13_legal = args.mode == "v13" &&
      (args.shape.m % 64 == 0 && args.shape.n % 64 == 0 && args.shape.k % 16 == 0);
  const bool is_v14_legal = args.mode == "v14" &&
      (args.shape.m % 64 == 0 && args.shape.n % 64 == 0 && args.shape.k % 16 == 0);
  const bool is_v15_legal = args.mode == "v15" &&
      (args.shape.m % 64 == 0 && args.shape.n % 64 == 0 && args.shape.k % 16 == 0);
  const bool is_v16_legal = args.mode == "v16" &&
      (args.shape.m % 64 == 0 && args.shape.n % 64 == 0 && args.shape.k % 16 == 0);
  const bool is_v17_legal = args.mode == "v17" &&
      (args.shape.m % 64 == 0 && args.shape.n % 64 == 0 && args.shape.k % 16 == 0);
  const bool is_v18_legal = args.mode == "v18" &&
      (args.shape.m % 64 == 0 && args.shape.n % 64 == 0 && args.shape.k % 16 == 0);
  const bool is_v19_legal = args.mode == "v19" &&
      (args.shape.m % 64 == 0 && args.shape.n % 64 == 0 && args.shape.k % 16 == 0);
  if (is_cublas_half_mode || is_v10_legal || is_v11_legal || is_v12_legal || is_v13_legal || is_v14_legal || is_v15_legal || is_v16_legal || is_v17_legal || is_v18_legal || is_v19_legal) {
    CUDA_CHECK(cudaMalloc(&d_a_wmma_half, a_count * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b_wmma_half, b_count * sizeof(half)));
    cudaError_t convert_status = cudaSuccess;
    if (is_cublas_half_mode) {
      convert_status = launch_user_gemm_v12_convert_inputs(d_a, d_b, d_a_wmma_half, d_b_wmma_half, args.shape, stream);
    } else if (args.mode == "v10") {
      convert_status = launch_user_gemm_v10_convert_inputs(d_a, d_b, d_a_wmma_half, d_b_wmma_half, args.shape, stream);
    } else if (args.mode == "v11") {
      convert_status = launch_user_gemm_v11_convert_inputs(d_a, d_b, d_a_wmma_half, d_b_wmma_half, args.shape, stream);
    } else if (args.mode == "v12") {
      convert_status = launch_user_gemm_v12_convert_inputs(d_a, d_b, d_a_wmma_half, d_b_wmma_half, args.shape, stream);
    } else if (args.mode == "v13") {
      convert_status = launch_user_gemm_v13_convert_inputs(d_a, d_b, d_a_wmma_half, d_b_wmma_half, args.shape, stream);
    } else if (args.mode == "v14") {
      convert_status = launch_user_gemm_v14_convert_inputs(d_a, d_b, d_a_wmma_half, d_b_wmma_half, args.shape, stream);
    } else if (args.mode == "v15") {
      convert_status = launch_user_gemm_v15_convert_inputs(d_a, d_b, d_a_wmma_half, d_b_wmma_half, args.shape, stream);
    } else if (args.mode == "v16") {
      convert_status = launch_user_gemm_v16_convert_inputs(d_a, d_b, d_a_wmma_half, d_b_wmma_half, args.shape, stream);
    } else if (args.mode == "v17") {
      convert_status = launch_user_gemm_v17_convert_inputs(d_a, d_b, d_a_wmma_half, d_b_wmma_half, args.shape, stream);
    } else if (args.mode == "v18") {
      convert_status = launch_user_gemm_v18_convert_inputs(d_a, d_b, d_a_wmma_half, d_b_wmma_half, args.shape, stream);
    } else {
      convert_status = launch_user_gemm_v19_convert_inputs(d_a, d_b, d_a_wmma_half, d_b_wmma_half, args.shape, stream);
    }
    if (convert_status != cudaSuccess) {
      std::fprintf(stderr, "%s input conversion failed: %s\n", args.mode.c_str(), cudaGetErrorString(convert_status));
      return 1;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  auto run_once = [&]() -> cudaError_t {
    if (args.mode == "library_cublas_gemm") return cublas_runner.run(d_a, d_b, d_c, args.shape);
    if (args.mode == "library_cublas_gemm_half") return cublas_half_runner.run(d_a_wmma_half, d_b_wmma_half, d_c, args.shape);
    if (args.mode == "user") return launch_user_gemm(d_a, d_b, d_c, args.shape, stream);
    if (args.mode == "v1") return launch_user_gemm_v1(d_a, d_b, d_c, args.shape, stream);
    if (args.mode == "v2") return launch_user_gemm_v2(d_a, d_b, d_c, args.shape, stream);
    if (args.mode == "v3") return launch_user_gemm_v3(d_a, d_b, d_c, args.shape, stream);
    if (args.mode == "v4") return launch_user_gemm_v4(d_a, d_b, d_c, args.shape, stream);
    if (args.mode == "v5") return launch_user_gemm_v5(d_a, d_b, d_c, args.shape, stream);
    if (args.mode == "v6") return launch_user_gemm_v6(d_a, d_b, d_c, args.shape, stream);
    if (args.mode == "v7") return launch_user_gemm_v7(d_a, d_b, d_c, args.shape, stream);
    if (args.mode == "v8") return launch_user_gemm_v8(d_a, d_b, d_c, args.shape, stream);
    if (args.mode == "v9") return launch_user_gemm_v9(d_a, d_b, d_c, args.shape, stream);
    if (args.mode == "v10") return launch_user_gemm_v10_wmma_only(d_a_wmma_half, d_b_wmma_half, d_c, args.shape, stream);
    if (args.mode == "v11") return launch_user_gemm_v11_wmma_only(d_a_wmma_half, d_b_wmma_half, d_c, args.shape, stream);
    if (args.mode == "v12") return launch_user_gemm_v12_wmma_only(d_a_wmma_half, d_b_wmma_half, d_c, args.shape, stream);
    if (args.mode == "v13") return launch_user_gemm_v13_wmma_only(d_a_wmma_half, d_b_wmma_half, d_c, args.shape, stream);
    if (args.mode == "v14") return launch_user_gemm_v14_wmma_only(d_a_wmma_half, d_b_wmma_half, d_c, args.shape, stream);
    if (args.mode == "v15") return launch_user_gemm_v15_wmma_only(d_a_wmma_half, d_b_wmma_half, d_c, args.shape, stream);
    if (args.mode == "v16") return launch_user_gemm_v16_wmma_only(d_a_wmma_half, d_b_wmma_half, d_c, args.shape, stream);
    if (args.mode == "v17") return launch_user_gemm_v17_wmma_only(d_a_wmma_half, d_b_wmma_half, d_c, args.shape, stream);
    if (args.mode == "v18") return launch_user_gemm_v18_wmma_only(d_a_wmma_half, d_b_wmma_half, d_c, args.shape, stream);
    if (args.mode == "v19") return launch_user_gemm_v19_wmma_only(d_a_wmma_half, d_b_wmma_half, d_c, args.shape, stream);
    return cudaErrorInvalidValue;
  };

  if (is_wmma_mode && !(is_v10_legal || is_v11_legal || is_v12_legal || is_v13_legal || is_v14_legal || is_v15_legal || is_v16_legal || is_v17_legal || is_v18_legal || is_v19_legal)) {
    std::ofstream csv(args.csv, std::ios::app);
    if (csv.tellp() == 0) csv << "mode,m,n,k,warmup,iters,ms,tflops,logical_gbs,status,note\n";
    const char* note = args.mode == "v10"
        ? "v10 WMMA learning kernel requires M/N/K multiples of 16"
        : "WMMA learning kernel requires M/N multiples of 64 and K multiple of 16";
    csv << args.mode << ',' << args.shape.m << ',' << args.shape.n << ',' << args.shape.k << ','
        << args.warmup << ',' << args.iters << ",nan,nan,nan,SKIP,"
        << note << '\n';
    std::printf("SKIP mode=%s m=%d n=%d k=%d reason=%s csv=%s\n",
                args.mode.c_str(), args.shape.m, args.shape.n, args.shape.k, note, args.csv.c_str());
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
  }

  for (int i = 0; i < args.warmup; ++i) {
    cudaError_t st = run_once();
    if (st != cudaSuccess) { std::fprintf(stderr, "launch failed: %s\n", cudaGetErrorString(st)); return 1; }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < args.iters; ++i) {
    cudaError_t st = run_once();
    if (st != cudaSuccess) { std::fprintf(stderr, "launch failed: %s\n", cudaGetErrorString(st)); return 1; }
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms_total = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_total, start, stop));
  const double ms = ms_total / args.iters;
  const double flops = 2.0 * args.shape.m * args.shape.n * args.shape.k;
  const double tflops = flops / (ms * 1.0e-3) / 1.0e12;
  const double logical_bytes = 4.0 * (static_cast<double>(a_count) + b_count + c_count);
  const double logical_gbs = logical_bytes / (ms * 1.0e-3) / 1.0e9;

  std::ofstream csv(args.csv, std::ios::app);
  if (csv.tellp() == 0) csv << "mode,m,n,k,warmup,iters,ms,tflops,logical_gbs\n";
  csv << args.mode << ',' << args.shape.m << ',' << args.shape.n << ',' << args.shape.k << ','
      << args.warmup << ',' << args.iters << ',' << ms << ',' << tflops << ',' << logical_gbs << '\n';
  std::printf("mode=%s m=%d n=%d k=%d ms=%g tflops=%g logical_gbs=%g csv=%s\n",
              args.mode.c_str(), args.shape.m, args.shape.n, args.shape.k, ms, tflops, logical_gbs, args.csv.c_str());

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  if (d_a_wmma_half) CUDA_CHECK(cudaFree(d_a_wmma_half));
  if (d_b_wmma_half) CUDA_CHECK(cudaFree(d_b_wmma_half));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return 0;
}
