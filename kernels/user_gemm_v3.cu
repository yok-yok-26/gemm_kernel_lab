#include "gemm.cuh"
#include "cuda_check.cuh"

// Exercise starting point: implement your GEMM kernel and launch policy here.
// Keep the public launch signature stable unless we deliberately change the lab contract.
// Contract: FP32 row-major A[m,k], B[k,n], C[m,n], C = A * B.



// simple shared memory gemm
template <size_t BLOCKSIZE, size_t TILESIZE>
__global__ void tile_seme_gemm_v3(float* a, float* b, float* c, int M, int K, int N){

  int idx_n = (threadIdx.x + blockDim.x * blockIdx.x) * TILESIZE;
  int idx_m = (threadIdx.y + blockDim.y * blockIdx.y) * TILESIZE;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  __shared__ float a_seme[TILESIZE * BLOCKSIZE][BLOCKSIZE + 4];
  __shared__ float b_seme[BLOCKSIZE][BLOCKSIZE * (TILESIZE + 1)];
  int PAD_TILESIZE = TILESIZE + 1;
  float acc[TILESIZE][TILESIZE] = {{0.0f}};     // !!!!! 等价于 float acc[TILESIZE][TILESIZE] = {0.0f};  !!!!! 注意：不能写成 float acc[TILESIZE][TILESIZE] = {}; 

  for (size_t istart = 0; istart < K; istart+=blockDim.x)
  {

    for (int itile = 0; itile < TILESIZE; itile++)
    {
      if (((idx_m + itile) < M) && (istart + tx < K))
      {
        a_seme[ty * TILESIZE + itile][tx] = a[K * (idx_m + itile) + istart + tx];
      }
      else
      {
        a_seme[ty * TILESIZE + itile][tx] = 0.0f;
      }
    }
    
    for (int itile = 0; itile < TILESIZE; itile++)
    {
      if ((idx_n + itile < N) && (istart + ty < K))
      {
        b_seme[ty][tx * PAD_TILESIZE + itile] = b[N * (istart + ty) + idx_n + itile];
      }
      else
      {
        b_seme[ty][tx * PAD_TILESIZE + itile] = 0.0f;
      }
    }
    __syncthreads();    
    

    for (int iy = 0; iy < TILESIZE; iy++)
    {
      for (int ix = 0; ix < TILESIZE; ix++)
      {
        for (int ib = 0; ib < BLOCKSIZE; ib++){
          acc[iy][ix] += a_seme[ty * TILESIZE + iy][ib] * b_seme[ib][tx * PAD_TILESIZE + ix];
        }
      }
    }
    __syncthreads();


  }


  for (int iy = 0; iy < TILESIZE; iy++)
  {
    for (int ix = 0; ix < TILESIZE; ix++)
    {
      if ((idx_m + iy < M) && (idx_n + ix < N))
      {
        c[(idx_m + iy) * N + (idx_n + ix)] = acc[iy][ix];
      }
    }
  }
  
}






cudaError_t launch_user_gemm_v3(const float* d_a, const float* d_b, float* d_c,
                             GemmShape shape, cudaStream_t stream) {
  constexpr int blockszie = 16;
  constexpr int tilesize = 4;
  dim3 block{blockszie, blockszie};            
  dim3 grid{
    (shape.n + block.x * tilesize - 1) / (block.x * tilesize), 
    (shape.m + block.y * tilesize - 1) / (block.y * tilesize)
  };

  tile_seme_gemm_v3<blockszie, tilesize><<<grid, block, 0, stream>>>(
    const_cast<float*>(d_a), const_cast<float*>(d_b), const_cast<float*>(d_c), shape.m, shape.k, shape.n
  );

  CUDA_KERNEL_CHECK();
  return cudaSuccess;
}


