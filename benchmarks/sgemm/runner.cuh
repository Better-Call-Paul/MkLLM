#pragma once

#include "cuda_common.cuh"
#include "sgemm_kernels.cuh"

inline void run_cublas_sgemm(cublasHandle_t handle,
                      int M, int N, int K,
                      float alpha,
                      const float *A, const float *B,
                      float beta, float *C) {
    size_t sizeA = sizeof(float) * M * K;
    size_t sizeB = sizeof(float) * K * N;
    size_t sizeC = sizeof(float) * M * N;
    float *dA, *dB, *dC;

    CUDA_CHECK(cudaMalloc((void**)&dA, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&dB, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&dC, sizeC));

    CUDA_CHECK(cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, C, sizeC, cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              dB, CUDA_R_32F, N,
                              dA, CUDA_R_32F, K,
                              &beta,
                              dC, CUDA_R_32F, N,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C, dC, sizeC, cudaMemcpyDeviceToHost));

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

inline void run_sgemm_naive(int M, int N, int K,
                     float alpha, const float *A,
                     const float *B, float beta, float *C) 
{
    dim3 blk(32,32), grid(CEIL_DIV(N,32), CEIL_DIV(M,32));
    sgemm_naive<<<grid,blk>>>(M,N,K,alpha,A,B,beta,C);
}

inline void run_sgemm_coalesced(
    int M,
    int N,
    int K,
    float alpha,
    const float *A,
    const float *B,
    float beta,
    float *C)
{
    constexpr uint blockSize = 32;
    dim3 blockDim(blockSize * blockSize, 1);
    dim3 gridDim(CEIL_DIV(M, blockSize), CEIL_DIV(N, blockSize));
    sgemm_global_mem_coalesce<blockSize><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

inline void run_sgemm_shmem_cache_blocking(int M, int N, int K,
                      float alpha,
                      const float *A, const float *B,
                      float beta, float *C)
{
    constexpr uint block_size = 32;
    dim3 blockDim(block_size * block_size, 1);

    dim3 gridDim(CEIL_DIV(M, block_size), CEIL_DIV(N, block_size));

    cudaFuncSetAttribute(sgemm_shmem_cache_blocking<32>, cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);

    sgemm_shmem_cache_blocking<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemm1DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) 
{
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / TM);
  sgemm_1d_block_tiling<BM, BN, BK, TM>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

