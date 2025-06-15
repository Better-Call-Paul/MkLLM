cmake_minimum_required(VERSION 3.18)
project(MkLLM LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(
    ${CMAKE_SOURCE_DIR}/engine/include
)

file(GLOB_RECURSE CUDA_SOURCES
    ${CMAKE_SOURCE_DIR}/engine/src/*.cu
    ${CMAKE_SOURCE_DIR}/engine/kernels/*.cu
)

add_executable(mkllm
    ${CUDA_SOURCES}
)

set_target_properties(mkllm PROPERTIES
    CUDA_ARCHITECTURES 80
    CUDA_SEPARABLE_COMPILATION ON
)
