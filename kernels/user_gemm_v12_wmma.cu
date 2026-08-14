#include "gemm.cuh"
#include "cuda_check.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;



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

__global__ void float_to_half_kernel_v12(const float* src, half* dst, int64_t count) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = __float2half_rn(src[idx]);
    }
}









// Exercise starting point: implement your GEMM kernel and launch policy here.
// Keep the public launch signature stable unless we deliberately change the lab contract.
// Contract: FP32 row-major A[m,k], B[k,n], C[m,n], C = A * B.
template <size_t BLOCKSIZE, size_t TILESIZE_N, size_t TILESIZE_M>
__global__ void tile_seme_gemm_v12(
    const half* A,
    const half* B,
    float* C,
    int M, int N, int K
) {
    int tx = threadIdx.x;
    int warp_id = threadIdx.x >> 5;
    int idx_warp_m = warp_id / 4;
    int idx_warp_n = warp_id % 4;
    int idx_t_m = tx / (16 / 2);    // [0, 16) [16, 32) [32, 48) [48, 64) 
    int idx_t_n = tx % (16 / 2);    // [0, 8)

    int tile_m = blockIdx.y;
    int tile_n = blockIdx.x;

    // 2 * 16 * 16 个线程，每个线程负责读取 2 个数。前 128 个线程，读取 256 个数，一个16x16 
    __shared__ half seme_a[4 * 16 * 16];    // 32 * 16 = 16 * 16 * 2
    __shared__ half seme_b[4 * 16 * 16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);


    for (int k0 = 0; k0 < K; k0 += 16) {

        const half* a_tile = A + (tile_m * 4) * 16 * K + k0;
        const half* b_tile = B + k0 * N + (tile_n * 4) * 16;

        seme_a[idx_t_m * 16 + idx_t_n * 2 + 0] = a_tile[idx_t_m * K + idx_t_n * 2 + 0];
        seme_a[idx_t_m * 16 + idx_t_n * 2 + 1] = a_tile[idx_t_m * K + idx_t_n * 2 + 1];

        seme_b[
            (idx_t_m / 16) * 16 * 16 + (idx_t_m % 16) * 16 + idx_t_n * 2 + 0
        ] = b_tile[
            (idx_t_m % 16) * N + idx_t_n * 2 + (idx_t_m / 16) * 16 + 0
        ];
        seme_b[
            (idx_t_m / 16) * 16 * 16 + (idx_t_m % 16) * 16 + idx_t_n * 2 + 1
        ] = b_tile[
            (idx_t_m % 16) * N + idx_t_n * 2 + (idx_t_m / 16) * 16 + 1
        ];
        __syncthreads();

        const half* a_seme_tile = static_cast<const half*>(seme_a + idx_warp_m * 16 * 16);
        const half* b_seme_tile = static_cast<const half*>(seme_b + idx_warp_n * 16 * 16);

        wmma::load_matrix_sync(a_frag, a_seme_tile, 16);
        wmma::load_matrix_sync(b_frag, b_seme_tile, 16 * 1);
        __syncthreads();

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* c_tile = C + (tile_m * 4 + idx_warp_m) * 16 * N + (tile_n * 4 + idx_warp_n) * 16;
    wmma::store_matrix_sync(c_tile, c_frag, N, wmma::mem_row_major);
}






cudaError_t launch_user_gemm_v12_convert_inputs(const float* d_a, const float* d_b,
                                                half* d_a_half, half* d_b_half,
                                                GemmShape shape, cudaStream_t stream) {
    const int64_t a_count = static_cast<int64_t>(shape.m) * shape.k;
    const int64_t b_count = static_cast<int64_t>(shape.k) * shape.n;
    constexpr int convert_block = 256;
    float_to_half_kernel_v12<<<static_cast<int>((a_count + convert_block - 1) / convert_block),
                               convert_block, 0, stream>>>(d_a, d_a_half, a_count);
    CUDA_KERNEL_CHECK();
    float_to_half_kernel_v12<<<static_cast<int>((b_count + convert_block - 1) / convert_block),
                               convert_block, 0, stream>>>(d_b, d_b_half, b_count);
    CUDA_KERNEL_CHECK();
    return cudaSuccess;
}

cudaError_t launch_user_gemm_v12_wmma_only(const half* d_a_half, const half* d_b_half,
                                           float* d_c, GemmShape shape, cudaStream_t stream) {
    constexpr int blockszie = 16;
    constexpr int tilesize_n = 4;
    constexpr int tilesize_m = 4;

    assert(shape.m % (blockszie * 4) == 0);
    assert(shape.n % (blockszie * 4) == 0);
    assert(shape.k % blockszie == 0);

    dim3 block(32 * 16);     // 算 4x4 块小矩阵
    dim3 grid(shape.n / (blockszie * 4), shape.m / (blockszie * 4));

    tile_seme_gemm_v12<blockszie, tilesize_n, tilesize_m><<<grid, block, 0, stream>>>(
        d_a_half,
        d_b_half,
        d_c,
        shape.m, shape.n, shape.k
    );

    CUDA_KERNEL_CHECK();
    return cudaSuccess;
}
