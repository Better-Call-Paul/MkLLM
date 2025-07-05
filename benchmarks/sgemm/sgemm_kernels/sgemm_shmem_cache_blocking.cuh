#pragma once

#include "../cuda_common.cuh"

template <const uint BLOCKSIZE>
__global__ void sgemm_shmem_cache_blocking(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C)
{
    const uint threadRow = threadIdx.x / BLOCKSIZE;
    const uint threadCol = threadIdx.x % BLOCKSIZE;

    const uint blockRow = blockIdx.y;
    const uint blockCol = blockIdx.x;

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // move pointers to start
    A += blockRow * BLOCKSIZE * K;
    B += blockCol * BLOCKSIZE;
    C += blockRow * BLOCKSIZE * N + blockCol * BLOCKSIZE;

    float sum = 0.0f;
    for (uint i = 0; i < K; i += BLOCKSIZE)
    {
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol]; 
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        __syncthreads();

        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        for (uint dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
        {
            sum += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        __syncthreads();
    }
    C[threadRow * N + threadCol] = sum * alpha + beta * C[threadRow * N + threadCol];
}
