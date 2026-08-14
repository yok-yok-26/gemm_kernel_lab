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

/// @brief 16个大B，4个float，8个half
/// @param dst_shared 
/// @param src_global 
/// @param src_bytes 
/// @return 
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

__global__ void float_to_half_kernel_v19(const float* src, half* dst, int64_t count) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = __float2half_rn(src[idx]);
    }
}






// Exercise starting point: implement your GEMM kernel and launch policy here.
// Keep the public launch signature stable unless we deliberately change the lab contract.
// Contract: FP32 row-major A[m,k], B[k,n], C[m,n], C = A * B.
template <size_t BLOCKSIZE, size_t TILESIZE_N, size_t TILESIZE_M>
__global__ void tile_seme_gemm_v19(
    const half* A,
    const half* B,
    float* C,
    int M, int N, int K
) {
    int tx = threadIdx.x;           // [0, 32*4)
    int warp_id = threadIdx.x >> 5;
    int warp_id_n = warp_id % 2, warp_id_m = warp_id / 2;
    int idx_t_m = tx / (4);    // [0, 16) [16, 32)
    int idx_t_n = tx % (4);    // [0, 4)
    int idx_t_m_b = tx / (8);    // [0, 16) 
    int idx_t_n_b = tx % (8);    // [0, 8)
    constexpr int SEME_LEN_B = 64 + 8;

    int tile_m = blockIdx.y;
    int tile_n = blockIdx.x;

    int read_flag = 0;
    int com_flag = 1;

    // 32 * 4 个线程，每个线程负责读取 8 个数。前 32 个线程，读取 256 个数
    __shared__ half seme_a[2][4 * 16 * (16 * 2)];           // 32 * 16 = 16 * 16 * 2
    __shared__ half seme_b[2][SEME_LEN_B * (16 * 2)];       // 32 * 16 = 16 * 16 * 2

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    // wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag_2;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[4];

    wmma::fill_fragment(c_frag[0], 0.0f);
    wmma::fill_fragment(c_frag[1], 0.0f);
    wmma::fill_fragment(c_frag[2], 0.0f);
    wmma::fill_fragment(c_frag[3], 0.0f);


    const half* a_tile = A + (tile_m * 4) * 16 * K + 0;
    const half* b_tile = B + 0 * N + (tile_n * 4) * 16;

    cp_async_ca_zfill_16(
        &seme_a[read_flag][
            idx_t_m * 32 + idx_t_n * 8
        ],
        a_tile + idx_t_m * K + idx_t_n * 8, 
        16
    );

    cp_async_ca_zfill_16(
        &seme_a[read_flag][
            (idx_t_m + 32) * 32 + idx_t_n * 8
        ],
        a_tile + (idx_t_m + 32) * K + idx_t_n * 8, 
        16
    );

    cp_async_ca_zfill_16(
        &seme_b[read_flag][
            idx_t_m_b * SEME_LEN_B + idx_t_n_b * 8
        ], 
        b_tile + idx_t_m_b * N + idx_t_n_b * 8, 
        16
    );

    cp_async_ca_zfill_16(
        &seme_b[read_flag][
            (idx_t_m_b + 16) * SEME_LEN_B + idx_t_n_b * 8
        ], 
        b_tile + (idx_t_m_b + 16) * N + idx_t_n_b * 8, 
        16
    );

    cp_async_commit_group();
    cp_async_wait_group0();
    __syncthreads();



    for (int k0 = 32; k0 < (K + 32); k0 += 32) {

        if (k0 < K)
        {
            const half* a_tile_loop = A + (tile_m * 4) * 16 * K + k0;
            const half* b_tile_loop = B + k0 * N + (tile_n * 4) * 16;

            cp_async_ca_zfill_16(
                &seme_a[com_flag][
                    idx_t_m * 32 + idx_t_n * 8
                ],
                a_tile_loop + idx_t_m * K + idx_t_n * 8, 
                16
            );

            cp_async_ca_zfill_16(
                &seme_a[com_flag][
                    (idx_t_m + 32) * 32 + idx_t_n * 8
                ],
                a_tile_loop + (idx_t_m + 32) * K + idx_t_n * 8, 
                16
            );

            cp_async_ca_zfill_16(
                &seme_b[com_flag][
                    idx_t_m_b * SEME_LEN_B + idx_t_n_b * 8
                ], 
                b_tile_loop + idx_t_m_b * N + idx_t_n_b * 8, 
                16
            );

            cp_async_ca_zfill_16(
                &seme_b[com_flag][
                    (idx_t_m_b + 16) * SEME_LEN_B + idx_t_n_b * 8
                ], 
                b_tile_loop + (idx_t_m_b + 16) * N + idx_t_n_b * 8, 
                16
            );


            cp_async_commit_group();
        }
        

        half* a_seme_tile = (
            &seme_a[read_flag][(warp_id_m * 2 + 0) * 32 * 16]
        );
        half* a_seme_tile_2 = (
            &seme_a[read_flag][(warp_id_m * 2 + 0) * 32 * 16 + 16]
        );
        half* b_seme_tile = (
            &seme_b[read_flag][(warp_id_n * 2 + 0) * 16]
        );
        half* b_seme_tile_2 = (
            &seme_b[read_flag][(warp_id_n * 2 + 0) * 16 + SEME_LEN_B * 16]
        );

        wmma::load_matrix_sync(a_frag, a_seme_tile, 32);
        wmma::load_matrix_sync(b_frag, b_seme_tile, SEME_LEN_B);
        wmma::mma_sync(
            c_frag[0], a_frag, b_frag, c_frag[0]
        );

        wmma::load_matrix_sync(a_frag, a_seme_tile_2, 32);
        wmma::load_matrix_sync(b_frag, b_seme_tile_2, SEME_LEN_B);
        wmma::mma_sync(
            c_frag[0], a_frag, b_frag, c_frag[0]
        );


        b_seme_tile = (
            &seme_b[read_flag][(warp_id_n * 2 + 1) * 16]
        );
        b_seme_tile_2 = (
            &seme_b[read_flag][(warp_id_n * 2 + 1) * 16 + SEME_LEN_B * 16]
        );

        wmma::load_matrix_sync(b_frag, b_seme_tile_2, SEME_LEN_B);
        wmma::mma_sync(
            c_frag[1], a_frag, b_frag, c_frag[1]
        );

        wmma::load_matrix_sync(a_frag, a_seme_tile, 32);
        wmma::load_matrix_sync(b_frag, b_seme_tile, SEME_LEN_B);
        wmma::mma_sync(
            c_frag[1], a_frag, b_frag, c_frag[1]
        );


        a_seme_tile = (
            &seme_a[read_flag][(warp_id_m * 2 + 1) * 32 * 16]
        );
        a_seme_tile_2 = (
            &seme_a[read_flag][(warp_id_m * 2 + 1) * 32 * 16 + 16]
        );

        wmma::load_matrix_sync(a_frag, a_seme_tile, 32);
        wmma::mma_sync(
            c_frag[3], a_frag, b_frag, c_frag[3]
        );

        wmma::load_matrix_sync(a_frag, a_seme_tile_2, 32);
        wmma::load_matrix_sync(b_frag, b_seme_tile_2, SEME_LEN_B);
        wmma::mma_sync(
            c_frag[3], a_frag, b_frag, c_frag[3]
        );


        b_seme_tile = (
            &seme_b[read_flag][(warp_id_n * 2 + 0) * 16]
        );
        b_seme_tile_2 = (
            &seme_b[read_flag][(warp_id_n * 2 + 0) * 16 + SEME_LEN_B * 16]
        );

        wmma::load_matrix_sync(b_frag, b_seme_tile_2, SEME_LEN_B);
        wmma::mma_sync(
            c_frag[2], a_frag, b_frag, c_frag[2]
        );

        wmma::load_matrix_sync(a_frag, a_seme_tile, 32);
        wmma::load_matrix_sync(b_frag, b_seme_tile, SEME_LEN_B);
        wmma::mma_sync(
            c_frag[2], a_frag, b_frag, c_frag[2]
        );



        if (k0 < K)
        {
            cp_async_wait_group0();   
        }
        __syncthreads();


        read_flag ^= 1;
        com_flag ^= 1;
        
    }


    float* c_tile = C + (tile_m * 4 + warp_id_m * 2 + 0) * 16 * N + (tile_n * 4 + warp_id_n * 2 + 0) * 16;
    wmma::store_matrix_sync(c_tile, c_frag[0], N, wmma::mem_row_major);

    c_tile = C + (tile_m * 4 + warp_id_m * 2 + 0) * 16 * N + (tile_n * 4 + warp_id_n * 2 + 1) * 16;
    wmma::store_matrix_sync(c_tile, c_frag[1], N, wmma::mem_row_major);

    c_tile = C + (tile_m * 4 + warp_id_m * 2 + 1) * 16 * N + (tile_n * 4 + warp_id_n * 2 + 0) * 16;
    wmma::store_matrix_sync(c_tile, c_frag[2], N, wmma::mem_row_major);
    
    c_tile = C + (tile_m * 4 + warp_id_m * 2 + 1) * 16 * N + (tile_n * 4 + warp_id_n * 2 + 1) * 16;
    wmma::store_matrix_sync(c_tile, c_frag[3], N, wmma::mem_row_major);

}






cudaError_t launch_user_gemm_v19_convert_inputs(const float* d_a, const float* d_b,
                                                half* d_a_half, half* d_b_half,
                                                GemmShape shape, cudaStream_t stream) {
    const int64_t a_count = static_cast<int64_t>(shape.m) * shape.k;
    const int64_t b_count = static_cast<int64_t>(shape.k) * shape.n;
    constexpr int convert_block = 256;
    float_to_half_kernel_v19<<<static_cast<int>((a_count + convert_block - 1) / convert_block),
                               convert_block, 0, stream>>>(d_a, d_a_half, a_count);
    CUDA_KERNEL_CHECK();
    float_to_half_kernel_v19<<<static_cast<int>((b_count + convert_block - 1) / convert_block),
                               convert_block, 0, stream>>>(d_b, d_b_half, b_count);
    CUDA_KERNEL_CHECK();
    return cudaSuccess;
}

cudaError_t launch_user_gemm_v19_wmma_only(const half* d_a_half, const half* d_b_half,
                                           float* d_c, GemmShape shape, cudaStream_t stream) {
    constexpr int blockszie = 16;
    constexpr int tilesize_n = 4;
    constexpr int tilesize_m = 4;

    assert(shape.m % (blockszie * 4) == 0);
    assert(shape.n % (blockszie * 4) == 0);
    assert(shape.k % blockszie == 0);

    dim3 block(32 * 4);     // 算 4x4 块小矩阵， 1 个warp负责算 4 块小矩阵
    dim3 grid(shape.n / (blockszie * 4), shape.m / (blockszie * 4));

    tile_seme_gemm_v19<blockszie, tilesize_n, tilesize_m><<<grid, block, 0, stream>>>(
        d_a_half,
        d_b_half,
        d_c,
        shape.m, shape.n, shape.k
    );

    CUDA_KERNEL_CHECK();
    return cudaSuccess;
}
