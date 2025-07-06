#pragma once 

#include "../cuda_common.cuh"

template<const uint BM, const uint BN, const uint BK, const uint TM>
__global__ void sgemm_1d_block_tiling(
    int M,
    int N,
    int K,
    float alpha,
    float *A,
    float *B,
    float beta,
    float *C
)
{
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float thread_results[TM] = {0.0};

    const uint threadRow = threadIdx.x / BN;
    const uint threadCol = threadIdx.x % BN;

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;

    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;

    assert(BM * BK == blockDim.x);
    assert(BK * BN == blockDim.x);

    for (uint i = 0; i < K; i += BK)
    {
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];

        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            float tempB = Bs[dotIdx * BN + threadCol];

            for (uint resIdx = 0; resIdx < TM; ++resIdx)
            {
                // local tile row then column within the tile
                thread_results[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * tempB;
            }
        }

        __syncthreads();
    }
    
    for (uint resIdx = 0; resIdx < TM; ++resIdx)
    {
        C[(threadRow * TM + resIdx) * N + threadCol] = alpha * thread_results[resIdx] + beta * C[(threadRow * TM + resIdx) * N + threadCol];
    }
}
