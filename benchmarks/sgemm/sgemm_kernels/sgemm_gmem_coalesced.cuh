#pragma once

#include "../cuda_common.cuh"

template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(
    int M,
    int N,
    int K,
    float alpha,
    const float *A,
    const float *B,
    float beta,
    float *C)
{
    const uint cRow = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const uint cCol = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (cRow < M && cCol < N) 
    {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) 
        {
            sum += A[cRow * K + i] * B[i * N + cCol];
        }
        C[cRow * N + cCol] = alpha * sum + beta * C[cRow * N + cCol];
    }
}