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
__global__ void tile_seme_gemm_v8(float* a, float* b, float* c, int M, int K, int N){

   if (M % (TILESIZE * BLOCKSIZE) != 0) return;
   if (N % (TILESIZE * BLOCKSIZE) != 0) return;


  constexpr int STAGES = 2;

  int idx_n = (threadIdx.x + blockDim.x * blockIdx.x) * TILESIZE;
  int idx_m = (threadIdx.y + blockDim.y * blockIdx.y) * TILESIZE;
  int idx_n_block = (blockDim.x * blockIdx.x) * TILESIZE;
  int idx_m_block = (blockDim.y * blockIdx.y) * TILESIZE;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = tx + ty * blockDim.x;

  extern __shared__ float seme[];
  constexpr int PAD_TILESIZE_A = BLOCKSIZE + 4;
  constexpr int PAD_TILESIZE_B = TILESIZE + 0;
  size_t STAGE_ASMEM_LEN = (TILESIZE * BLOCKSIZE) * (BLOCKSIZE + 4);
  size_t STAGE_BSMEM_LEN = ((TILESIZE + 0) * BLOCKSIZE) * (BLOCKSIZE);
  float* a_seme = seme;
  float* b_seme = &seme[STAGES * STAGE_ASMEM_LEN];

  float acc[TILESIZE][TILESIZE] = {{0.0f}};     // !!!!! 等价于 float acc[TILESIZE][TILESIZE] = {0.0f};  !!!!! 注意：不能写成 float acc[TILESIZE][TILESIZE] = {}; 
  uint8_t read_flag = 0, comp_flag = 1;

 

  auto tileA_cur = [&](int stage, int row_idx, int col_idx) -> float& {
    return a_seme[stage * STAGE_ASMEM_LEN + row_idx * PAD_TILESIZE_A + col_idx];
  };
  auto tileB_cur = [&](int stage, int row_idx, int col_idx) -> float& {
    return b_seme[stage * STAGE_BSMEM_LEN + row_idx * PAD_TILESIZE_B * BLOCKSIZE + col_idx];
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
        src_bys <= 16 ? src_bys : 16
    );
    if (src_bys > 16)
    {
        cp_async_ca_zfill_16(
            &tileA_cur(stage_cur, irow_a, icol_a * TILESIZE + TILESIZE / 2),
            &a[(idx_m_block + irow_a) * K + istart_cur + icol_a * TILESIZE + TILESIZE / 2],
            src_bys - 16
        );
    }
    
    
    
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
        src_bys <= 16 ? src_bys : 16
    );
    if (src_bys > 16)
    {
        cp_async_ca_zfill_16(
            &tileB_cur(stage_cur, irow_b, icol_b * TILESIZE + TILESIZE / 2),
            &b[(istart_cur + irow_b) * N + idx_n_block + icol_b * TILESIZE + TILESIZE / 2],
            src_bys - 16
        );
    }
    
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
        float b4 = tileB_cur(read_flag, ib, tx * TILESIZE + 4);
        float b5 = tileB_cur(read_flag, ib, tx * TILESIZE + 5);
        float b6 = tileB_cur(read_flag, ib, tx * TILESIZE + 6);
        float b7 = tileB_cur(read_flag, ib, tx * TILESIZE + 7);

        float a0 = tileA_cur(read_flag, ty * TILESIZE + 0, ib);
        float a1 = tileA_cur(read_flag, ty * TILESIZE + 1, ib);
        float a2 = tileA_cur(read_flag, ty * TILESIZE + 2, ib);
        float a3 = tileA_cur(read_flag, ty * TILESIZE + 3, ib);
        float a4 = tileA_cur(read_flag, ty * TILESIZE + 4, ib);
        float a5 = tileA_cur(read_flag, ty * TILESIZE + 5, ib);
        float a6 = tileA_cur(read_flag, ty * TILESIZE + 6, ib);
        float a7 = tileA_cur(read_flag, ty * TILESIZE + 7, ib);

        acc[0][0] += a0 * b0;
        acc[0][1] += a0 * b1;
        acc[0][2] += a0 * b2;
        acc[0][3] += a0 * b3;
        acc[0][4] += a0 * b4;
        acc[0][5] += a0 * b5;
        acc[0][6] += a0 * b6;
        acc[0][7] += a0 * b7;

        acc[1][0] += a1 * b0;
        acc[1][1] += a1 * b1;
        acc[1][2] += a1 * b2;
        acc[1][3] += a1 * b3;
        acc[1][4] += a1 * b4;
        acc[1][5] += a1 * b5;
        acc[1][6] += a1 * b6;
        acc[1][7] += a1 * b7;

        acc[2][0] += a2 * b0;
        acc[2][1] += a2 * b1;
        acc[2][2] += a2 * b2;
        acc[2][3] += a2 * b3;
        acc[2][4] += a2 * b4;
        acc[2][5] += a2 * b5;
        acc[2][6] += a2 * b6;
        acc[2][7] += a2 * b7;

        acc[3][0] += a3 * b0;
        acc[3][1] += a3 * b1;
        acc[3][2] += a3 * b2;
        acc[3][3] += a3 * b3;
        acc[3][4] += a3 * b4;
        acc[3][5] += a3 * b5;
        acc[3][6] += a3 * b6;
        acc[3][7] += a3 * b7;

        acc[4][0] += a4 * b0;
        acc[4][1] += a4 * b1;
        acc[4][2] += a4 * b2;
        acc[4][3] += a4 * b3;
        acc[4][4] += a4 * b4;
        acc[4][5] += a4 * b5;
        acc[4][6] += a4 * b6;
        acc[4][7] += a4 * b7;

        acc[5][0] += a5 * b0;
        acc[5][1] += a5 * b1;
        acc[5][2] += a5 * b2;
        acc[5][3] += a5 * b3;
        acc[5][4] += a5 * b4;
        acc[5][5] += a5 * b5;
        acc[5][6] += a5 * b6;
        acc[5][7] += a5 * b7;

        acc[6][0] += a6 * b0;
        acc[6][1] += a6 * b1;
        acc[6][2] += a6 * b2;
        acc[6][3] += a6 * b3;
        acc[6][4] += a6 * b4;
        acc[6][5] += a6 * b5;
        acc[6][6] += a6 * b6;
        acc[6][7] += a6 * b7;

        acc[7][0] += a7 * b0;
        acc[7][1] += a7 * b1;
        acc[7][2] += a7 * b2;
        acc[7][3] += a7 * b3;
        acc[7][4] += a7 * b4;
        acc[7][5] += a7 * b5;
        acc[7][6] += a7 * b6;
        acc[7][7] += a7 * b7;
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





cudaError_t launch_user_gemm_v8(const float* d_a, const float* d_b, float* d_c,
                             GemmShape shape, cudaStream_t stream) {
  constexpr int blockszie = 16;
  constexpr int tilesize = 8;
  dim3 block{blockszie, blockszie};            
  dim3 grid{
    (shape.n + block.x * tilesize - 1) / (block.x * tilesize), 
    (shape.m + block.y * tilesize - 1) / (block.y * tilesize)
  };

  int total_cnt = 2 * tilesize * blockszie * (blockszie + 4) + 
  2 * tilesize * blockszie * (blockszie + 0);

  tile_seme_gemm_v8<blockszie, tilesize><<<grid, block, total_cnt * sizeof(float), stream>>>(
    const_cast<float*>(d_a), const_cast<float*>(d_b), const_cast<float*>(d_c), shape.m, shape.k, shape.n
  );

  CUDA_KERNEL_CHECK();
  return cudaSuccess;
}
