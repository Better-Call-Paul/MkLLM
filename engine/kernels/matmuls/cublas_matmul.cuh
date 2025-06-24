#pragma once

#include "../cuda_common.cuh"
/*
void run_cublas_sgemm(
    float* A, float* B, float* C,
    float alpha, float beta,
    int M, int N, int K,
    cudaStream_t stream = 0)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B, N,
                A, K,
                &beta,
                C, N);

    cublasDestroy(handle);
}
*/