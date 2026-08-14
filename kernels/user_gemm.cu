#include "gemm.cuh"
#include "cuda_check.cuh"
#include <cublas_v2.h>

// Exercise starting point: implement your GEMM kernel and launch policy here.
// Keep the public launch signature stable unless we deliberately change the lab contract.
// Contract: FP32 row-major A[m,k], B[k,n], C[m,n], C = A * B.

cudaError_t launch_user_gemm(const float* d_a, const float* d_b, float* d_c,
                             GemmShape shape, cudaStream_t stream) {
  (void)d_a;
  (void)d_b;
  (void)d_c;
  (void)shape;
  (void)stream;
  return cudaErrorNotSupported;
}

cudaError_t launch_cublas_gemm(const float* d_a, const float* d_b, float* d_c,
                               GemmShape shape, cudaStream_t stream) {
  cublasHandle_t handle;
  cublasStatus_t st = cublasCreate(&handle);
  if (st != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
  st = cublasSetStream(handle, stream);
  if (st != CUBLAS_STATUS_SUCCESS) {
    cublasDestroy(handle);
    return cudaErrorUnknown;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  // cuBLAS is column-major. Row-major C=A*B is equivalent to column-major C^T=B^T*A^T.
  st = cublasSgemm(handle,
                   CUBLAS_OP_N, CUBLAS_OP_N,
                   shape.n, shape.m, shape.k,
                   &alpha,
                   d_b, shape.n,
                   d_a, shape.k,
                   &beta,
                   d_c, shape.n);
  cublasStatus_t destroy_st = cublasDestroy(handle);
  return (st == CUBLAS_STATUS_SUCCESS && destroy_st == CUBLAS_STATUS_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}

cudaError_t launch_cublas_gemm_half(const half* d_a, const half* d_b, float* d_c,
                                    GemmShape shape, cudaStream_t stream) {
  cublasHandle_t handle;
  cublasStatus_t st = cublasCreate(&handle);
  if (st != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
  st = cublasSetStream(handle, stream);
  if (st != CUBLAS_STATUS_SUCCESS) {
    cublasDestroy(handle);
    return cudaErrorUnknown;
  }
  st = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  if (st != CUBLAS_STATUS_SUCCESS) {
    cublasDestroy(handle);
    return cudaErrorUnknown;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  st = cublasGemmEx(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    shape.n, shape.m, shape.k,
                    &alpha,
                    d_b, CUDA_R_16F, shape.n,
                    d_a, CUDA_R_16F, shape.k,
                    &beta,
                    d_c, CUDA_R_32F, shape.n,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cublasStatus_t destroy_st = cublasDestroy(handle);
  return (st == CUBLAS_STATUS_SUCCESS && destroy_st == CUBLAS_STATUS_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}
