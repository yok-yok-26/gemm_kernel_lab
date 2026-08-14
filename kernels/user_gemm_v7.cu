#include "gemm.cuh"
#include "cuda_check.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;


static __device__ __forceinline__ unsigned cvta_to_shared_u32(const void* ptr)
{
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

static __device__ __forceinline__ void cp_async_ca_zfill_16(
    void* dst_shared,
    const void* src_global,
    int src_bytes)
{
#if __CUDA_ARCH__ >= 800
    unsigned smem_addr = cvta_to_shared_u32(dst_shared);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::
            "r"(smem_addr), "l"(src_global), "r"(src_bytes));
#else
    if (src_bytes == 16)
    {
        *reinterpret_cast<float4*>(dst_shared) =
            *reinterpret_cast<const float4*>(src_global);
    }
    else
    {
        char* dst = reinterpret_cast<char*>(dst_shared);
        const char* src = reinterpret_cast<const char*>(src_global);
        #pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            dst[i] = (i < src_bytes) ? src[i] : 0;
        }
    }
#endif
}

static __device__ __forceinline__ void cp_async_commit_group()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

static __device__ __forceinline__ void cp_async_wait_group0()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}









// Exercise starting point: implement your GEMM kernel and launch policy here.
// Keep the public launch signature stable unless we deliberately change the lab contract.
// Contract: FP32 row-major A[m,k], B[k,n], C[m,n], C = A * B.

// 先不考虑 M N 越界的情况
template <size_t BLOCKSIZE, size_t TILESIZE>
__global__ void tile_seme_gemm_v7(float* a, float* b, float* c, int M, int K, int N){

   if (M % (TILESIZE * BLOCKSIZE) != 0) return;
   if (N % (TILESIZE * BLOCKSIZE) != 0) return;


  constexpr int STAGES = 2;

  int idx_n = (threadIdx.x + blockDim.x * blockIdx.x) * TILESIZE;
  int idx_m = (threadIdx.y + blockDim.y * blockIdx.y) * TILESIZE;
  int idx_n_block = (blockDim.x * blockIdx.x) * TILESIZE;
  int idx_m_block = (blockDim.y * blockIdx.y) * TILESIZE;
  uint8_t tx = threadIdx.x;
  uint8_t ty = threadIdx.y;
  int tid = tx + ty * blockDim.x;

  extern __shared__ float seme[];
  constexpr int STAGE_ASMEM_LEN = (TILESIZE * BLOCKSIZE) * (BLOCKSIZE + 4);
  constexpr int STAGE_BSMEM_LEN = ((TILESIZE + 0) * BLOCKSIZE) * (BLOCKSIZE);
  float* a_seme = seme;
  float* b_seme = &seme[STAGES * STAGE_ASMEM_LEN];

  float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f;
  float c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
  float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f;
  float c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;
  uint8_t read_flag = 0, comp_flag = 1;

 

  auto tileA_cur = [&](int stage, int row_idx, int col_idx) -> float& {
    return a_seme[stage * STAGE_ASMEM_LEN + row_idx * (BLOCKSIZE + 4) + col_idx];
  };
  auto tileB_cur = [&](int stage, int row_idx, int col_idx) -> float& {
    return b_seme[stage * STAGE_BSMEM_LEN + row_idx * (TILESIZE + 0) * BLOCKSIZE + col_idx];
  };


  auto pred_data = [&](int istart_cur, int stage_cur){

    int irow_a = tid / (BLOCKSIZE / TILESIZE);
    int icol_a = tid % (BLOCKSIZE / TILESIZE);

    // src_bys：有效复制的字节数
    int src_bys = 4 * TILESIZE;
    if ((istart_cur + icol_a * TILESIZE) >= K)
    {
        src_bys = 0;
    }
    else if ((istart_cur + icol_a * TILESIZE + TILESIZE) > K)
    {
        src_bys = (K - (istart_cur + icol_a * TILESIZE )) * 4;
    }

    cp_async_ca_zfill_16(
        &tileA_cur(stage_cur, irow_a, icol_a * TILESIZE),
        &a[(idx_m_block + irow_a) * K + istart_cur + icol_a * TILESIZE],
        src_bys
    );


    int irow_b = ty;
    int icol_b = tx;

    src_bys = 4 * TILESIZE;
    if ((idx_n_block + icol_b * TILESIZE) >= N)
    {
        src_bys = 0;
    }
    else if ((idx_n_block + icol_b * TILESIZE + TILESIZE) > N)
    {
        src_bys = (N - (idx_n_block + icol_b * TILESIZE)) * 4;
    }

    cp_async_ca_zfill_16(
        &tileB_cur(stage_cur, irow_b, icol_b * TILESIZE),
        &b[(istart_cur + irow_b) * N + idx_n_block + icol_b * TILESIZE],
        src_bys
    );

    cp_async_commit_group();

  };

  pred_data(0, read_flag);
  cp_async_wait_group0();
  __syncthreads();


  for (size_t istart = blockDim.x; istart < (K + blockDim.x); istart+=blockDim.x)
  {
    if (istart < K)
    {
        pred_data(istart, comp_flag);
    }
    

    #pragma unroll
    for (int ib = 0; ib < BLOCKSIZE; ++ib) {
        float b0 = tileB_cur(read_flag, ib, tx * TILESIZE + 0);
        float b1 = tileB_cur(read_flag, ib, tx * TILESIZE + 1);
        float b2 = tileB_cur(read_flag, ib, tx * TILESIZE + 2);
        float b3 = tileB_cur(read_flag, ib, tx * TILESIZE + 3);

        float a0 = tileA_cur(read_flag, ty * TILESIZE + 0, ib);
        float a1 = tileA_cur(read_flag, ty * TILESIZE + 1, ib);
        float a2 = tileA_cur(read_flag, ty * TILESIZE + 2, ib);
        float a3 = tileA_cur(read_flag, ty * TILESIZE + 3, ib);

        float a_reg = tileA_cur(read_flag, ty * TILESIZE + 0, ib);
        c00 = fmaf(a_reg, b0, c00);
        c01 = fmaf(a_reg, b1, c01);
        c02 = fmaf(a_reg, b2, c02);
        c03 = fmaf(a_reg, b3, c03);

        a_reg = tileA_cur(read_flag, ty * TILESIZE + 1, ib);
        c10 = fmaf(a_reg, b0, c10);
        c11 = fmaf(a_reg, b1, c11);
        c12 = fmaf(a_reg, b2, c12);
        c13 = fmaf(a_reg, b3, c13);

        a_reg = tileA_cur(read_flag, ty * TILESIZE + 2, ib);
        c20 = fmaf(a_reg, b0, c20);
        c21 = fmaf(a_reg, b1, c21);
        c22 = fmaf(a_reg, b2, c22);
        c23 = fmaf(a_reg, b3, c23);

        a_reg = tileA_cur(read_flag, ty * TILESIZE + 3, ib);
        c30 = fmaf(a_reg, b0, c30);
        c31 = fmaf(a_reg, b1, c31);
        c32 = fmaf(a_reg, b2, c32);
        c33 = fmaf(a_reg, b3, c33);
    }


    // 等待
    if (istart < K)
    {
        cp_async_wait_group0();
    }
    __syncthreads();

    read_flag ^= 1;
    comp_flag ^= 1;
  }



//   for (int iy = 0; iy < TILESIZE; iy++)
//   {
//     for (int ix = 0; ix < TILESIZE; ix++)
//     {
//       if ((idx_m + iy < M) && (idx_n + ix < N))
//       {
//         c[(idx_m + iy) * N + (idx_n + ix)] = acc[iy][ix];
//       }
//     }
//   }
  
  *reinterpret_cast<float4*>(&c[(idx_m + 0) * N + (idx_n)]) = make_float4(c00, c01, c02, c03);
  *reinterpret_cast<float4*>(&c[(idx_m + 1) * N + (idx_n)]) = make_float4(c10, c11, c12, c13);
  *reinterpret_cast<float4*>(&c[(idx_m + 2) * N + (idx_n)]) = make_float4(c20, c21, c22, c23);
  *reinterpret_cast<float4*>(&c[(idx_m + 3) * N + (idx_n)]) = make_float4(c30, c31, c32, c33);
}





cudaError_t launch_user_gemm_v7(const float* d_a, const float* d_b, float* d_c,
                             GemmShape shape, cudaStream_t stream) {
  constexpr int blockszie = 32;
  constexpr int tilesize = 4;
  dim3 block{blockszie, blockszie};            
  dim3 grid{
    (shape.n + block.x * tilesize - 1) / (block.x * tilesize), 
    (shape.m + block.y * tilesize - 1) / (block.y * tilesize)
  };

  int total_cnt = 2 * tilesize * blockszie * (blockszie + 4) + 
  2 * tilesize * blockszie * (blockszie + 0);

    cudaFuncSetAttribute(
        tile_seme_gemm_v7<blockszie, tilesize>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(total_cnt * sizeof(float))
    );

  tile_seme_gemm_v7<blockszie, tilesize><<<grid, block, total_cnt * sizeof(float), stream>>>(
    const_cast<float*>(d_a), const_cast<float*>(d_b), const_cast<float*>(d_c), shape.m, shape.k, shape.n
  );

  CUDA_KERNEL_CHECK();
  return cudaSuccess;
}
