#include "gemm.cuh"
#include "cuda_check.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;





// Exercise starting point: implement your GEMM kernel and launch policy here.
// Keep the public launch signature stable unless we deliberately change the lab contract.
// Contract: FP32 row-major A[m,k], B[k,n], C[m,n], C = A * B.



template <size_t BLOCKSIZE, size_t TILESIZE>
__global__ void tile_seme_gemm_v5_error1(float* a, float* b, float* c, int M, int K, int N){

  if (M % (BLOCKSIZE*TILESIZE) != 0) return;
  if (N % (BLOCKSIZE*TILESIZE) != 0) return;
  if (K % (BLOCKSIZE) != 0) return;

  constexpr int STAGES = 2;

  __shared__ cuda::pipeline_shared_state<
      cuda::thread_scope_block,
      STAGES
  > pipe_state;

  auto block = cg::this_thread_block();
  auto pipe = cuda::make_pipeline(block, &pipe_state);


  int idx_n = (threadIdx.x + blockDim.x * blockIdx.x) * TILESIZE;
  int idx_m = (threadIdx.y + blockDim.y * blockIdx.y) * TILESIZE;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  constexpr int PAD_TILESIZE = TILESIZE + 0;
  __shared__ float a_seme[STAGES][TILESIZE * BLOCKSIZE][BLOCKSIZE + 4];
  __shared__ float b_seme[STAGES][BLOCKSIZE][BLOCKSIZE * PAD_TILESIZE];
  float acc[TILESIZE][TILESIZE] = {{0.0f}};     // !!!!! 等价于 float acc[TILESIZE][TILESIZE] = {0.0f};  !!!!! 注意：不能写成 float acc[TILESIZE][TILESIZE] = {}; 
  uint8_t read_flag = 0, comp_flag = 1;

 

  pipe.producer_acquire();
  for (int itile = 0; itile < TILESIZE; itile++)
  {
    if (((idx_m + itile) < M) && (0 + tx < K)){
      cuda::memcpy_async(
        &a_seme[comp_flag][ty * TILESIZE + itile][tx], 
        &a[K * (idx_m + itile) + 0 + tx], 
        sizeof(float), 
        pipe
      );
    }
    else
    {
      a_seme[comp_flag][ty * TILESIZE + itile][tx] = 0.0f;
    }
    
    if ((idx_n + itile < N) && (0 + ty < K))
    {
      cuda::memcpy_async(
        &b_seme[comp_flag][ty][tx * PAD_TILESIZE + itile], 
        &b[N * (0 + ty) + idx_n + itile], 
        sizeof(float), 
        pipe
      );
    }
    else
    {
      b_seme[comp_flag][ty][tx * PAD_TILESIZE + itile] = 0.0f;
    }

  }
  pipe.producer_commit();


  for (size_t istart = blockDim.x; istart < K; istart+=blockDim.x)
  {

    pipe.producer_acquire();
    for (int itile = 0; itile < TILESIZE; itile++)
    {
      if (((idx_m + itile) < M) && (istart + tx < K)){
        cuda::memcpy_async(
          &a_seme[read_flag][ty * TILESIZE + itile][tx], 
          &a[K * (idx_m + itile) + istart + tx], 
          sizeof(float), 
          pipe
        );
      }
      else
      {
        a_seme[read_flag][ty * TILESIZE + itile][tx] = 0.0f;
      }
      
      if ((idx_n + itile < N) && (istart + ty < K))
      {
        cuda::memcpy_async(
          &b_seme[read_flag][ty][tx * PAD_TILESIZE + itile], 
          &b[N * (istart + ty) + idx_n + itile], 
          sizeof(float), 
          pipe
        );
      }
      else
      {
        b_seme[read_flag][ty][tx * PAD_TILESIZE + itile] = 0.0f;
      }

    }
    pipe.producer_commit();



    pipe.consumer_wait();
    #pragma unroll
    for (int ib = 0; ib < BLOCKSIZE; ++ib) {
        float b0 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 0];
        float b1 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 1];
        float b2 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 2];
        float b3 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 3];

        float a0 = a_seme[comp_flag][ty * TILESIZE + 0][ib];
        float a1 = a_seme[comp_flag][ty * TILESIZE + 1][ib];
        float a2 = a_seme[comp_flag][ty * TILESIZE + 2][ib];
        float a3 = a_seme[comp_flag][ty * TILESIZE + 3][ib];

        acc[0][0] += a0 * b0;
        acc[0][1] += a0 * b1;
        acc[0][2] += a0 * b2;
        acc[0][3] += a0 * b3;

        acc[1][0] += a1 * b0;
        acc[1][1] += a1 * b1;
        acc[1][2] += a1 * b2;
        acc[1][3] += a1 * b3;

        acc[2][0] += a2 * b0;
        acc[2][1] += a2 * b1;
        acc[2][2] += a2 * b2;
        acc[2][3] += a2 * b3;

        acc[3][0] += a3 * b0;
        acc[3][1] += a3 * b1;
        acc[3][2] += a3 * b2;
        acc[3][3] += a3 * b3;
    }
    pipe.consumer_release();

    read_flag = (read_flag + 1) % 2;
    comp_flag = (comp_flag + 1) % 2;
  }

  pipe.consumer_wait();
  #pragma unroll
  for (int ib = 0; ib < BLOCKSIZE; ++ib) {
      float b0 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 0];
      float b1 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 1];
      float b2 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 2];
      float b3 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 3];

      float a0 = a_seme[comp_flag][ty * TILESIZE + 0][ib];
      float a1 = a_seme[comp_flag][ty * TILESIZE + 1][ib];
      float a2 = a_seme[comp_flag][ty * TILESIZE + 2][ib];
      float a3 = a_seme[comp_flag][ty * TILESIZE + 3][ib];

      acc[0][0] += a0 * b0;
      acc[0][1] += a0 * b1;
      acc[0][2] += a0 * b2;
      acc[0][3] += a0 * b3;

      acc[1][0] += a1 * b0;
      acc[1][1] += a1 * b1;
      acc[1][2] += a1 * b2;
      acc[1][3] += a1 * b3;

      acc[2][0] += a2 * b0;
      acc[2][1] += a2 * b1;
      acc[2][2] += a2 * b2;
      acc[2][3] += a2 * b3;

      acc[3][0] += a3 * b0;
      acc[3][1] += a3 * b1;
      acc[3][2] += a3 * b2;
      acc[3][3] += a3 * b3;
  }
  pipe.consumer_release();


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







template <size_t BLOCKSIZE, size_t TILESIZE>
__global__ void tile_seme_gemm_v51(float* a, float* b, float* c, int M, int K, int N) {
    static_assert(TILESIZE == 4, "This kernel currently assumes TILESIZE == 4.");

    constexpr int STAGES = 2;
    constexpr int PAD_TILESIZE = TILESIZE + 0;


    auto block = cg::this_thread_block();
    auto pipe = cuda::make_pipeline();

    int idx_n = (threadIdx.x + blockDim.x * blockIdx.x) * TILESIZE;
    int idx_m = (threadIdx.y + blockDim.y * blockIdx.y) * TILESIZE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (blockDim.x != BLOCKSIZE || blockDim.y != BLOCKSIZE) {
        return;
    }

    __shared__ float a_seme[STAGES][TILESIZE * BLOCKSIZE][BLOCKSIZE + 4];
    __shared__ float b_seme[STAGES][BLOCKSIZE][BLOCKSIZE * PAD_TILESIZE];

    float acc[TILESIZE][TILESIZE] = {{0.0f}};

    uint8_t read_flag = 0;
    uint8_t comp_flag = 1;

    // ============================================================
    // preload tile 0 into comp_flag
    // ============================================================
    pipe.producer_acquire();

    #pragma unroll
    for (int itile = 0; itile < TILESIZE; ++itile) {
        if (((idx_m + itile) < M) && (tx < K)) {
            cuda::memcpy_async(
                &a_seme[comp_flag][ty * TILESIZE + itile][tx],
                &a[K * (idx_m + itile) + tx],
                sizeof(float),
                pipe
            );
        } else {
            a_seme[comp_flag][ty * TILESIZE + itile][tx] = 0.0f;
        }

        if ((idx_n + itile < N) && (ty < K)) {
            cuda::memcpy_async(
                &b_seme[comp_flag][ty][tx * PAD_TILESIZE + itile],
                &b[N * ty + idx_n + itile],
                sizeof(float),
                pipe
            );
        } else {
            b_seme[comp_flag][ty][tx * PAD_TILESIZE + itile] = 0.0f;
        }
    }

    pipe.producer_commit();

    // ============================================================
    // main loop:
    // consume current tile first, then produce next tile
    // ============================================================
    for (size_t istart = BLOCKSIZE; istart < static_cast<size_t>(K); istart += BLOCKSIZE) {
        // --------------------------------------------------------
        // consume current tile: comp_flag
        // --------------------------------------------------------
        pipe.consumer_wait();
        block.sync();

        #pragma unroll
        for (int ib = 0; ib < BLOCKSIZE; ++ib) {
            float b0 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 0];
            float b1 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 1];
            float b2 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 2];
            float b3 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 3];

            float a0 = a_seme[comp_flag][ty * TILESIZE + 0][ib];
            float a1 = a_seme[comp_flag][ty * TILESIZE + 1][ib];
            float a2 = a_seme[comp_flag][ty * TILESIZE + 2][ib];
            float a3 = a_seme[comp_flag][ty * TILESIZE + 3][ib];

            acc[0][0] += a0 * b0;
            acc[0][1] += a0 * b1;
            acc[0][2] += a0 * b2;
            acc[0][3] += a0 * b3;

            acc[1][0] += a1 * b0;
            acc[1][1] += a1 * b1;
            acc[1][2] += a1 * b2;
            acc[1][3] += a1 * b3;

            acc[2][0] += a2 * b0;
            acc[2][1] += a2 * b1;
            acc[2][2] += a2 * b2;
            acc[2][3] += a2 * b3;

            acc[3][0] += a3 * b0;
            acc[3][1] += a3 * b1;
            acc[3][2] += a3 * b2;
            acc[3][3] += a3 * b3;
        }

        block.sync();
        pipe.consumer_release();

        // --------------------------------------------------------
        // produce next tile into read_flag
        // --------------------------------------------------------
        pipe.producer_acquire();

        #pragma unroll
        for (int itile = 0; itile < TILESIZE; ++itile) {
            if (((idx_m + itile) < M) && (istart + tx < static_cast<size_t>(K))) {
                cuda::memcpy_async(
                    &a_seme[read_flag][ty * TILESIZE + itile][tx],
                    &a[K * (idx_m + itile) + istart + tx],
                    sizeof(float),
                    pipe
                );
            } else {
                a_seme[read_flag][ty * TILESIZE + itile][tx] = 0.0f;
            }

            if ((idx_n + itile < N) && (istart + ty < static_cast<size_t>(K))) {
                cuda::memcpy_async(
                    &b_seme[read_flag][ty][tx * PAD_TILESIZE + itile],
                    &b[N * (istart + ty) + idx_n + itile],
                    sizeof(float),
                    pipe
                );
            } else {
                b_seme[read_flag][ty][tx * PAD_TILESIZE + itile] = 0.0f;
            }
        }

        pipe.producer_commit();

        read_flag = static_cast<uint8_t>((read_flag + 1) & 1);
        comp_flag = static_cast<uint8_t>((comp_flag + 1) & 1);
    }

    // ============================================================
    // consume last tile: comp_flag
    // ============================================================
    pipe.consumer_wait();
    block.sync();

    #pragma unroll
    for (int ib = 0; ib < BLOCKSIZE; ++ib) {
        float b0 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 0];
        float b1 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 1];
        float b2 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 2];
        float b3 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 3];

        float a0 = a_seme[comp_flag][ty * TILESIZE + 0][ib];
        float a1 = a_seme[comp_flag][ty * TILESIZE + 1][ib];
        float a2 = a_seme[comp_flag][ty * TILESIZE + 2][ib];
        float a3 = a_seme[comp_flag][ty * TILESIZE + 3][ib];

        acc[0][0] += a0 * b0;
        acc[0][1] += a0 * b1;
        acc[0][2] += a0 * b2;
        acc[0][3] += a0 * b3;

        acc[1][0] += a1 * b0;
        acc[1][1] += a1 * b1;
        acc[1][2] += a1 * b2;
        acc[1][3] += a1 * b3;

        acc[2][0] += a2 * b0;
        acc[2][1] += a2 * b1;
        acc[2][2] += a2 * b2;
        acc[2][3] += a2 * b3;

        acc[3][0] += a3 * b0;
        acc[3][1] += a3 * b1;
        acc[3][2] += a3 * b2;
        acc[3][3] += a3 * b3;
    }

    block.sync();
    pipe.consumer_release();

    // ============================================================
    // write back
    // ============================================================
    #pragma unroll
    for (int iy = 0; iy < TILESIZE; ++iy) {
        #pragma unroll
        for (int ix = 0; ix < TILESIZE; ++ix) {
            if ((idx_m + iy < M) && (idx_n + ix < N)) {
                c[(idx_m + iy) * N + (idx_n + ix)] = acc[iy][ix];
            }
        }
    }
}







template <size_t BLOCKSIZE, size_t TILESIZE>
__global__ void tile_seme_gemm_v5_error2(float* a, float* b, float* c, int M, int K, int N) {
    static_assert(TILESIZE == 4, "This kernel currently assumes TILESIZE == 4.");

    constexpr int STAGES = 2;
    constexpr int PAD_TILESIZE = TILESIZE + 0;

    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope_block,
        STAGES
    > pipe_state;

    auto block = cg::this_thread_block();
    auto pipe = cuda::make_pipeline(block, &pipe_state);

    int idx_n = (threadIdx.x + blockDim.x * blockIdx.x) * TILESIZE;
    int idx_m = (threadIdx.y + blockDim.y * blockIdx.y) * TILESIZE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (blockDim.x != BLOCKSIZE || blockDim.y != BLOCKSIZE) {
        return;
    }

    __shared__ float a_seme[STAGES][TILESIZE * BLOCKSIZE][BLOCKSIZE + 4];
    __shared__ float b_seme[STAGES][BLOCKSIZE][BLOCKSIZE * PAD_TILESIZE];

    float acc[TILESIZE][TILESIZE] = {{0.0f}};

    uint8_t read_flag = 0;
    uint8_t comp_flag = 1;

    // ============================================================
    // preload tile 0 into comp_flag
    // ============================================================
    pipe.producer_acquire();

    #pragma unroll
    for (int itile = 0; itile < TILESIZE; ++itile) {
        if (((idx_m + itile) < M) && (tx < K)) {
            cuda::memcpy_async(
                block,
                &a_seme[comp_flag][ty * TILESIZE + itile][tx],
                &a[K * (idx_m + itile) + tx],
                sizeof(float),
                pipe
            );
        } else {
            a_seme[comp_flag][ty * TILESIZE + itile][tx] = 0.0f;
        }

        if ((idx_n + itile < N) && (ty < K)) {
            cuda::memcpy_async(
                block,
                &b_seme[comp_flag][ty][tx * PAD_TILESIZE + itile],
                &b[N * ty + idx_n + itile],
                sizeof(float),
                pipe
            );
        } else {
            b_seme[comp_flag][ty][tx * PAD_TILESIZE + itile] = 0.0f;
        }
    }

    pipe.producer_commit();

    // ============================================================
    // main loop:
    // consume current tile first, then produce next tile
    // ============================================================
    for (size_t istart = BLOCKSIZE; istart < static_cast<size_t>(K); istart += BLOCKSIZE) {
        // --------------------------------------------------------
        // consume current tile: comp_flag
        // --------------------------------------------------------
        pipe.consumer_wait();
        block.sync();

        #pragma unroll
        for (int ib = 0; ib < BLOCKSIZE; ++ib) {
            float b0 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 0];
            float b1 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 1];
            float b2 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 2];
            float b3 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 3];

            float a0 = a_seme[comp_flag][ty * TILESIZE + 0][ib];
            float a1 = a_seme[comp_flag][ty * TILESIZE + 1][ib];
            float a2 = a_seme[comp_flag][ty * TILESIZE + 2][ib];
            float a3 = a_seme[comp_flag][ty * TILESIZE + 3][ib];

            acc[0][0] += a0 * b0;
            acc[0][1] += a0 * b1;
            acc[0][2] += a0 * b2;
            acc[0][3] += a0 * b3;

            acc[1][0] += a1 * b0;
            acc[1][1] += a1 * b1;
            acc[1][2] += a1 * b2;
            acc[1][3] += a1 * b3;

            acc[2][0] += a2 * b0;
            acc[2][1] += a2 * b1;
            acc[2][2] += a2 * b2;
            acc[2][3] += a2 * b3;

            acc[3][0] += a3 * b0;
            acc[3][1] += a3 * b1;
            acc[3][2] += a3 * b2;
            acc[3][3] += a3 * b3;
        }

        block.sync();
        pipe.consumer_release();

        // --------------------------------------------------------
        // produce next tile into read_flag
        // --------------------------------------------------------
        pipe.producer_acquire();

        #pragma unroll
        for (int itile = 0; itile < TILESIZE; ++itile) {
            if (((idx_m + itile) < M) && (istart + tx < static_cast<size_t>(K))) {
                cuda::memcpy_async(
                    block,
                    &a_seme[read_flag][ty * TILESIZE + itile][tx],
                    &a[K * (idx_m + itile) + istart + tx],
                    sizeof(float),
                    pipe
                );
            } else {
                a_seme[read_flag][ty * TILESIZE + itile][tx] = 0.0f;
            }

            if ((idx_n + itile < N) && (istart + ty < static_cast<size_t>(K))) {
                cuda::memcpy_async(
                    block,
                    &b_seme[read_flag][ty][tx * PAD_TILESIZE + itile],
                    &b[N * (istart + ty) + idx_n + itile],
                    sizeof(float),
                    pipe
                );
            } else {
                b_seme[read_flag][ty][tx * PAD_TILESIZE + itile] = 0.0f;
            }
        }

        pipe.producer_commit();

        read_flag = static_cast<uint8_t>((read_flag + 1) & 1);
        comp_flag = static_cast<uint8_t>((comp_flag + 1) & 1);
    }

    // ============================================================
    // consume last tile: comp_flag
    // ============================================================
    pipe.consumer_wait();
    block.sync();

    #pragma unroll
    for (int ib = 0; ib < BLOCKSIZE; ++ib) {
        float b0 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 0];
        float b1 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 1];
        float b2 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 2];
        float b3 = b_seme[comp_flag][ib][tx * PAD_TILESIZE + 3];

        float a0 = a_seme[comp_flag][ty * TILESIZE + 0][ib];
        float a1 = a_seme[comp_flag][ty * TILESIZE + 1][ib];
        float a2 = a_seme[comp_flag][ty * TILESIZE + 2][ib];
        float a3 = a_seme[comp_flag][ty * TILESIZE + 3][ib];

        acc[0][0] += a0 * b0;
        acc[0][1] += a0 * b1;
        acc[0][2] += a0 * b2;
        acc[0][3] += a0 * b3;

        acc[1][0] += a1 * b0;
        acc[1][1] += a1 * b1;
        acc[1][2] += a1 * b2;
        acc[1][3] += a1 * b3;

        acc[2][0] += a2 * b0;
        acc[2][1] += a2 * b1;
        acc[2][2] += a2 * b2;
        acc[2][3] += a2 * b3;

        acc[3][0] += a3 * b0;
        acc[3][1] += a3 * b1;
        acc[3][2] += a3 * b2;
        acc[3][3] += a3 * b3;
    }

    block.sync();
    pipe.consumer_release();

    // ============================================================
    // write back
    // ============================================================
    #pragma unroll
    for (int iy = 0; iy < TILESIZE; ++iy) {
        #pragma unroll
        for (int ix = 0; ix < TILESIZE; ++ix) {
            if ((idx_m + iy < M) && (idx_n + ix < N)) {
                c[(idx_m + iy) * N + (idx_n + ix)] = acc[iy][ix];
            }
        }
    }
}






template <size_t BLOCKSIZE, size_t TILESIZE>
__global__ void tile_seme_gemm_v5(
    float* a,
    float* b,
    float* c,
    int M,
    int K,
    int N
) {
    static_assert(TILESIZE == 4, "This kernel currently assumes TILESIZE == 4.");

    if (blockDim.x != BLOCKSIZE || blockDim.y != BLOCKSIZE) {
        return;
    }

    constexpr int STAGES = 2;
    constexpr int PAD_TILESIZE = TILESIZE + 0;

    int idx_n = (threadIdx.x + blockDim.x * blockIdx.x) * TILESIZE;
    int idx_m = (threadIdx.y + blockDim.y * blockIdx.y) * TILESIZE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float a_seme[STAGES][TILESIZE * BLOCKSIZE][BLOCKSIZE + 4];
    __shared__ float b_seme[STAGES][BLOCKSIZE][(BLOCKSIZE + 1) * TILESIZE];

    float acc[TILESIZE][TILESIZE] = {{0.0f}};

    auto pipe = cuda::make_pipeline();

    auto load_tile = [&](int stage, size_t istart) {
        pipe.producer_acquire();

        #pragma unroll
        for (int itile = 0; itile < TILESIZE; ++itile) {
            if (((idx_m + itile) < M) && (istart + tx < static_cast<size_t>(K))) {
                cuda::memcpy_async(
                    &a_seme[stage][ty * TILESIZE + itile][tx],
                    &a[K * (idx_m + itile) + istart + tx],
                    sizeof(float),
                    pipe
                );
            } else {
                a_seme[stage][ty * TILESIZE + itile][tx] = 0.0f;
            }
        }

        #pragma unroll
        for (int itile = 0; itile < TILESIZE; ++itile) {
            if ((idx_n + itile < N) && (istart + ty < static_cast<size_t>(K))) {
                cuda::memcpy_async(
                    &b_seme[stage][ty][tx * PAD_TILESIZE + itile],
                    &b[N * (istart + ty) + idx_n + itile],
                    sizeof(float),
                    pipe
                );
            } else {
                b_seme[stage][ty][tx * PAD_TILESIZE + itile] = 0.0f;
            }
        }

        pipe.producer_commit();
    };

    auto compute_tile = [&](int stage) {
        #pragma unroll
        for (int ib = 0; ib < BLOCKSIZE; ++ib) {
            float b0 = b_seme[stage][ib][tx * PAD_TILESIZE + 0];
            float b1 = b_seme[stage][ib][tx * PAD_TILESIZE + 1];
            float b2 = b_seme[stage][ib][tx * PAD_TILESIZE + 2];
            float b3 = b_seme[stage][ib][tx * PAD_TILESIZE + 3];

            float a0 = a_seme[stage][ty * TILESIZE + 0][ib];
            float a1 = a_seme[stage][ty * TILESIZE + 1][ib];
            float a2 = a_seme[stage][ty * TILESIZE + 2][ib];
            float a3 = a_seme[stage][ty * TILESIZE + 3][ib];

            acc[0][0] += a0 * b0;
            acc[0][1] += a0 * b1;
            acc[0][2] += a0 * b2;
            acc[0][3] += a0 * b3;

            acc[1][0] += a1 * b0;
            acc[1][1] += a1 * b1;
            acc[1][2] += a1 * b2;
            acc[1][3] += a1 * b3;

            acc[2][0] += a2 * b0;
            acc[2][1] += a2 * b1;
            acc[2][2] += a2 * b2;
            acc[2][3] += a2 * b3;

            acc[3][0] += a3 * b0;
            acc[3][1] += a3 * b1;
            acc[3][2] += a3 * b2;
            acc[3][3] += a3 * b3;
        }
    };

    // ============================================================
    // preload tile 0
    // ============================================================
    load_tile(0, 0);

    // 等 tile 0 到 shared
    pipe.consumer_wait();
    __syncthreads();

    int curr_stage = 0;
    int next_stage = 1;

    // ============================================================
    // main pipeline loop
    //
    // next tile async copy 和 current tile compute 重叠
    // ============================================================
    for (size_t istart = BLOCKSIZE; istart < static_cast<size_t>(K); istart += BLOCKSIZE) {
        // 1. 发起下一块异步拷贝
        load_tile(next_stage, istart);

        // 2. 计算当前块
        // 理想情况下，当前 compute 期间 next tile 的 async copy 在后台推进
        compute_tile(curr_stage);

        // 3. 所有线程都结束读取 curr_stage
        __syncthreads();

        // 4. 释放当前消费 stage
        pipe.consumer_release();

        // 5. 等 next tile 真的准备好，然后进入下一轮
        pipe.consumer_wait();
        __syncthreads();

        curr_stage ^= 1;
        next_stage ^= 1;
    }

    // ============================================================
    // compute last tile
    // ============================================================
    compute_tile(curr_stage);

    __syncthreads();
    pipe.consumer_release();

    // ============================================================
    // write back
    // ============================================================
    #pragma unroll
    for (int iy = 0; iy < TILESIZE; ++iy) {
        #pragma unroll
        for (int ix = 0; ix < TILESIZE; ++ix) {
            if ((idx_m + iy < M) && (idx_n + ix < N)) {
                c[(idx_m + iy) * N + (idx_n + ix)] = acc[iy][ix];
            }
        }
    }
}




cudaError_t launch_user_gemm_v5(const float* d_a, const float* d_b, float* d_c,
                             GemmShape shape, cudaStream_t stream) {
  constexpr int blockszie = 16;
  constexpr int tilesize = 4;
  dim3 block{blockszie, blockszie};            
  dim3 grid{
    (shape.n + block.x * tilesize - 1) / (block.x * tilesize), 
    (shape.m + block.y * tilesize - 1) / (block.y * tilesize)
  };

  tile_seme_gemm_v5<blockszie, tilesize><<<grid, block, 0, stream>>>(
    const_cast<float*>(d_a), const_cast<float*>(d_b), const_cast<float*>(d_c), shape.m, shape.k, shape.n
  );


  CUDA_KERNEL_CHECK();
  return cudaSuccess;
}
