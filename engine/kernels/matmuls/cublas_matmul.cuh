#pragma once

#include "../cuda_common.cuh"

void run_cublas_sgemm(
    float* A, float* B, float* C,
    float alpha, float beta,
    int M, int N, int K,
    cudaStream_t stream = 0)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    cublasSgemm(
  handle,
  CUBLAS_OP_T,           // transpose “A” → we pass B here
  CUBLAS_OP_T,           // transpose “B” → we pass A here
  /* m */ N,             // # rows of the column-major C = N
  /* n */ M,             // # cols of C = M
  /* k */ K,
  &alpha,
  /* A */ B, /* lda=*/N,  // lda = original # columns of B (N)
  /* B */ A, /* ldb=*/K,  // ldb = original # columns of A (K)
  &beta,
  /* C */ C, /* ldc=*/N   // ldc = row-major stride = N
);


    cublasDestroy(handle);
}


