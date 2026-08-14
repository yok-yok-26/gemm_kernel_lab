#include "gemm.cuh"

void cpu_gemm_reference(const float* a, const float* b, float* c, GemmShape shape) {
  for (int i = 0; i < shape.m; ++i) {
    for (int j = 0; j < shape.n; ++j) {
      double acc = 0.0;
      for (int p = 0; p < shape.k; ++p) {
        acc += static_cast<double>(a[i * shape.k + p]) * static_cast<double>(b[p * shape.n + j]);
      }
      c[i * shape.n + j] = static_cast<float>(acc);
    }
  }
}
