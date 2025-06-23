#pragma once

#include "../cuda_common.cuh"


__global__ void basic_sgemm(float* A, float* B, float* C, int M, int N, int K)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= M || y >= N) {
        return;
    }

    float sum = 0;

    for (int i = 0; i < K; ++i) {
        sum += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = sum;
}
