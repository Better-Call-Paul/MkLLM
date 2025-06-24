cmake_minimum_required(VERSION 3.18)
project(MkLLM LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# locate the CUDA Toolkit (provides CUDA::cudart, CUDA::cublas, etc.)
find_package(CUDAToolkit REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/engine/include
)

# ── Main executable ────────────────────────────────────────────────────────────
file(GLOB_RECURSE APP_SOURCES
    ${CMAKE_SOURCE_DIR}/engine/src/*.cu
)

add_executable(mkllm
    ${APP_SOURCES}
)

set_target_properties(mkllm PROPERTIES
    CUDA_ARCHITECTURES         "80-real"
    CUDA_SEPARABLE_COMPILATION  ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ── MatMul benchmarking executable ─────────────────────────────────────────────
file(GLOB BENCH_SOURCES
    ${CMAKE_SOURCE_DIR}/engine/kernels/*.cu
    ${CMAKE_SOURCE_DIR}/engine/src/cuda_utils.cu
    ${CMAKE_SOURCE_DIR}/engine/src/matmul_runner.cu
)

add_executable(matmul_bench
    ${BENCH_SOURCES}
)

set_target_properties(matmul_bench PROPERTIES
    CUDA_ARCHITECTURES         "80-real"
    CUDA_SEPARABLE_COMPILATION  ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# link cuBLAS for your SGEMM calls
target_link_libraries(matmul_bench
    PRIVATE CUDA::cublas
)
