#pragma once

#include "../cuda_common.cuh"

__global__ void sgemm_naive(int M, int N, int K,
                     float alpha, const float *A,
                     const float *B, float beta, float *C) 
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M && col < N) 
    {
        float sum = 0.0f;

        for (int i = 0; i < K; ++i) 
        {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * sum + C[row * N + col] * beta;
    }
}
