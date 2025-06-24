#pragma once

#include "../cuda_common.cuh"
/*
__global__ void naive_sgemm(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col >= N || row >= M) 
    {
        return;
    }

    float sum = 0.0f;
    for (uint i = 0; i < K; ++i) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

void run_naive_sgemm(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
    dim3 blockDim(32, 32);
    naive_sgemm<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}*/
