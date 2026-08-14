cmake_minimum_required(VERSION 3.20)
project(gemm_kernel_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

add_library(gemm_lab STATIC
  kernels/user_gemm.cu
  kernels/user_gemm_v1.cu
  kernels/user_gemm_v2.cu
  kernels/user_gemm_v3.cu
  kernels/user_gemm_v4.cu
  kernels/user_gemm_v5.cu
  kernels/user_gemm_v6.cu
  kernels/user_gemm_v7.cu
  kernels/user_gemm_v8.cu
  kernels/user_gemm_v9.cu
  kernels/user_gemm_v10_wmma.cu
  kernels/user_gemm_v11_wmma.cu
  kernels/user_gemm_v12_wmma.cu
  kernels/user_gemm_v13_wmma.cu
  kernels/user_gemm_v14_wmma.cu
  kernels/user_gemm_v15_wmma.cu
  kernels/user_gemm_v16_wmma.cu
  kernels/user_gemm_v17_wmma.cu
  kernels/user_gemm_v18_wmma.cu
  kernels/user_gemm_v19_wmma.cu
  reference/cpu_gemm.cpp
)
target_include_directories(gemm_lab PUBLIC include reference)
target_link_libraries(gemm_lab PUBLIC CUDA::cudart CUDA::cublas)
target_compile_options(gemm_lab PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>
)

add_executable(test_gemm tests/test_gemm.cu)
target_link_libraries(test_gemm PRIVATE gemm_lab)

add_executable(bench_gemm benchmark/bench_gemm.cu)
target_link_libraries(bench_gemm PRIVATE gemm_lab)
