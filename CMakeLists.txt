cmake_minimum_required(VERSION 3.18)
project(MkLLM LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# — use old CUDA find for CMake ≤ 3.20 —
find_package(CUDA REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/engine/include
    ${CUDA_INCLUDE_DIRS}
)

# Main 
file(GLOB_RECURSE APP_SOURCES
    ${CMAKE_SOURCE_DIR}/engine/src/*.cu
)

add_executable(mkllm
    ${APP_SOURCES}
)

set_target_properties(mkllm PROPERTIES
    CUDA_ARCHITECTURES 80
    CUDA_SEPARABLE_COMPILATION ON
)

# MatMul benchmarking
file(GLOB KERNEL_SOURCES
    ${CMAKE_SOURCE_DIR}/engine/kernels/*.cu
    ${CMAKE_SOURCE_DIR}/engine/src/cuda_utils.cu
)

add_executable(matmul_bench
    ${KERNEL_SOURCES}
)

set_target_properties(matmul_bench PROPERTIES
    CUDA_ARCHITECTURES 80
    CUDA_SEPARABLE_COMPILATION ON
)

# link in cuBLAS
target_link_libraries(matmul_bench
    PRIVATE
      ${CUDA_CUBLAS_LIBRARIES}
)
