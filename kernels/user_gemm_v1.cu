#include "gemm.cuh"
#include "cuda_check.cuh"

// Exercise starting point: implement your GEMM kernel and launch policy here.
// Keep the public launch signature stable unless we deliberately change the lab contract.
// Contract: FP32 row-major A[m,k], B[k,n], C[m,n], C = A * B.



// simple shared memory gemm
template < size_t BLOCKSIZE>
__global__ void simple_seme_gemm(float* a, float* b, float* c, int M, int K, int N){

  int idx_n = threadIdx.x + blockDim.x * blockIdx.x;
  int idx_m = threadIdx.y + blockDim.y * blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  __shared__ float a_seme[BLOCKSIZE][BLOCKSIZE];
  __shared__ float b_seme[BLOCKSIZE][BLOCKSIZE];
  float acc = 0.0f;

  for (size_t istart = 0; istart < K; istart+=blockDim.x)
  {
    if ((idx_m < M) && (istart + tx < K))
    {
      a_seme[ty][tx] = a[K * idx_m + istart + tx];
    }
    else
    {
      a_seme[ty][tx] = 0.0f;
    }
    
    if ((idx_n < N) && (istart + ty < K))
    {
      b_seme[ty][tx] = b[N * (istart + ty) + idx_n];
    }
    else
    {
      b_seme[ty][tx] = 0.0f;
    }
    __syncthreads();


    for (int i = 0; i < BLOCKSIZE; i++)
    {
      acc += a_seme[ty][i] * b_seme[i][tx];
    }
    __syncthreads();

  }

  if (idx_m < M && idx_n < N)
  {
    c[idx_m * N + idx_n] = acc;
  }
  
  
}






cudaError_t launch_user_gemm_v1(const float* d_a, const float* d_b, float* d_c,
                             GemmShape shape, cudaStream_t stream) {
  constexpr int blockszie = 16;
  dim3 block{blockszie, blockszie};            
  dim3 grid{(shape.n + block.x - 1) / block.x, (shape.m + block.y - 1) / block.y};

  simple_seme_gemm<blockszie><<<grid, block, 0, stream>>>(
    const_cast<float*>(d_a), const_cast<float*>(d_b), const_cast<float*>(d_c), shape.m, shape.k, shape.n
  );

  CUDA_KERNEL_CHECK();
  return cudaSuccess;
}


