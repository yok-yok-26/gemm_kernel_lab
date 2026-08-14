#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <string>

struct GemmShape {
  int m = 128;
  int n = 128;
  int k = 128;
};

struct GemmTolerance {
  float atol = 1.0e-3f;
  float rtol = 1.0e-3f;
};

// Current lab contract: FP32 row-major A[m,k], B[k,n], C[m,n], C = A * B.
// User owns this launch function internals, including grid/block/smem/stream policy.
cudaError_t launch_user_gemm(const float* d_a, const float* d_b, float* d_c,
                             GemmShape shape, cudaStream_t stream);


cudaError_t launch_user_gemm_v1(const float* d_a, const float* d_b, float* d_c,
                                GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v2(const float* d_a, const float* d_b, float* d_c,
                                GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v3(const float* d_a, const float* d_b, float* d_c,
                                GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v4(const float* d_a, const float* d_b, float* d_c,
                                GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v5(const float* d_a, const float* d_b, float* d_c,
                                GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v6(const float* d_a, const float* d_b, float* d_c,
                                GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v7(const float* d_a, const float* d_b, float* d_c,
                                GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v8(const float* d_a, const float* d_b, float* d_c,
                                GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v9(const float* d_a, const float* d_b, float* d_c,
                                GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v10_convert_inputs(const float* d_a, const float* d_b,
                                                half* d_a_half, half* d_b_half,
                                                GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v10_wmma_only(const half* d_a_half, const half* d_b_half,
                                           float* d_c, GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v11_convert_inputs(const float* d_a, const float* d_b,
                                                half* d_a_half, half* d_b_half,
                                                GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v11_wmma_only(const half* d_a_half, const half* d_b_half,
                                           float* d_c, GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v12_convert_inputs(const float* d_a, const float* d_b,
                                                half* d_a_half, half* d_b_half,
                                                GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v12_wmma_only(const half* d_a_half, const half* d_b_half,
                                           float* d_c, GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v13_convert_inputs(const float* d_a, const float* d_b,
                                                half* d_a_half, half* d_b_half,
                                                GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v13_wmma_only(const half* d_a_half, const half* d_b_half,
                                           float* d_c, GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v14_convert_inputs(const float* d_a, const float* d_b,
                                                half* d_a_half, half* d_b_half,
                                                GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v14_wmma_only(const half* d_a_half, const half* d_b_half,
                                           float* d_c, GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v15_convert_inputs(const float* d_a, const float* d_b,
                                                half* d_a_half, half* d_b_half,
                                                GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v15_wmma_only(const half* d_a_half, const half* d_b_half,
                                           float* d_c, GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v16_convert_inputs(const float* d_a, const float* d_b,
                                                half* d_a_half, half* d_b_half,
                                                GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v16_wmma_only(const half* d_a_half, const half* d_b_half,
                                           float* d_c, GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v17_convert_inputs(const float* d_a, const float* d_b,
                                                half* d_a_half, half* d_b_half,
                                                GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v17_wmma_only(const half* d_a_half, const half* d_b_half,
                                           float* d_c, GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v18_convert_inputs(const float* d_a, const float* d_b,
                                                half* d_a_half, half* d_b_half,
                                                GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v18_wmma_only(const half* d_a_half, const half* d_b_half,
                                           float* d_c, GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v19_convert_inputs(const float* d_a, const float* d_b,
                                                half* d_a_half, half* d_b_half,
                                                GemmShape shape, cudaStream_t stream);

cudaError_t launch_user_gemm_v19_wmma_only(const half* d_a_half, const half* d_b_half,
                                           float* d_c, GemmShape shape, cudaStream_t stream);

cudaError_t launch_cublas_gemm(const float* d_a, const float* d_b, float* d_c,
                               GemmShape shape, cudaStream_t stream);

cudaError_t launch_cublas_gemm_half(const half* d_a, const half* d_b, float* d_c,
                                    GemmShape shape, cudaStream_t stream);

void cpu_gemm_reference(const float* a, const float* b, float* c, GemmShape shape);
