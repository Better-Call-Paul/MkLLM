cmake_minimum_required(VERSION 3.19)
project(MkLLM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD       20)
set(CMAKE_CUDA_STANDARD      20)
set(CMAKE_CUDA_ARCHITECTURES 90) 
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)       
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}/engine/include
  ${PROJECT_SOURCE_DIR}/engine/kernels
  ${PROJECT_SOURCE_DIR}/benchmarks
)

add_executable(sgemm_benchmark
  benchmarks/sgemm/sgemm.cu
)
target_link_libraries(sgemm_benchmark
  CUDA::cudart
  CUDA::cublas
)

add_executable(engine_main
  engine/src/main.cu
)
target_link_libraries(engine_main
  CUDA::cudart
)
