#include "cuda_common.cuh"
#include "runner.cuh"
#include "sgemm_kernels.cuh"
/*
int main()
{
    int deviceIdx = 0;
    if (const char* env = std::getenv("DEVICE"))
    {
        deviceIdx = std::atoi(env);
    }

    CUDA_CHECK(cudaSetDevice(deviceIdx));
    std::cout << "Running on device " << deviceIdx << "\n";
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    std::vector<int> sizes = {128,256,512,1024};
    int max_size = sizes.back();
    size_t max_bytes = sizeof(float) * max_size * max_size;
    std::vector<float> A(max_size * max_size);
    std::vector<float> B(max_size * max_size);
    std::vector<float> C0(max_size * max_size);
    std::vector<float> C_ref(max_size * max_size);
    std::vector<float> C_naive(max_size * max_size);
    std::vector<float> C_coal(max_size * max_size);
    std::vector<float> C_shared(max_size * max_size);
    std::vector<float> C_1D_block_tile(max_size * max_size);
    std::vector<flaot> C_2D_block_tile(max_size * max_size);

    randomizeMatrix(A.data(), A.size());
    randomizeMatrix(B.data(), B.size());
    rangeInitMatrix(C0.data(), C0.size());
    C_ref = C0;
    C_naive = C0;
    C_coal = C0;
    C_shared = C0;
    C_1D_block_tile = C0;
    C_2D_block_tile = C0;
    
    float* dA;
    float* dB;
    float* dC;
    CUDA_CHECK(cudaMalloc(&dA, max_bytes));
    CUDA_CHECK(cudaMalloc(&dB, max_bytes));
    CUDA_CHECK(cudaMalloc(&dC, max_bytes));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), max_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), max_bytes, cudaMemcpyHostToDevice));
    float alpha = 2.0f;
    float beta  = 3.0f;
    {
        int M = sizes[0];
        int N = M;
        int K = M;
        CUDA_CHECK(cudaMemcpy(dC, C_naive.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
        run_sgemm_naive(M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(C_naive.data(), dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
        run_cublas_sgemm(handle, M, N, K, alpha, A.data(), B.data(), beta, C_ref.data());
        if (!verifyMatrix(C_ref.data(), C_naive.data(), M, N, error_log_file))
        {  
            std::cout << "Error with naive\n";
            return 1;
        }
        CUDA_CHECK(cudaMemcpy(dC, C_coal.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
        run_sgemm_coalesced(M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(C_coal.data(), dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
        if (!verifyMatrix(C_ref.data(), C_coal.data(), M, N, error_log_file))
        {
            std::cout << "Error with coalesced\n";
            return 1;
        }
        CUDA_CHECK(cudaMemcpy(dC, C_shared.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
        run_sgemm_shmem_cache_blocking(M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(C_shared.data(), dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
        if (!verifyMatrix(C_ref.data(), C_shared.data(), M, N, error_log_file))
        {
            std::cout << "Issue with Shmem Cache Blocking Kernel\n";
            return 1;
        }
    }
    const int NUM_RUNS = 10;
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int size : sizes)
    {
        int M = size;
        int N = size;
        int K = size;
        float t_naive   = 0.0f;
        float t_coal    = 0.0f;
        float t_cublas  = 0.0f;
        float t_shared  = 0.0f;
        for (int i = 0; i < NUM_RUNS; ++i)
        {
            CUDA_CHECK(cudaMemcpy(dC, C_naive.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(start));
            run_sgemm_naive(M, N, K, alpha, dA, dB, beta, dC);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            {
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                t_naive += ms;
            }
            CUDA_CHECK(cudaMemcpy(dC, C_coal.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(start));
            run_sgemm_coalesced(M, N, K, alpha, dA, dB, beta, dC);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            {
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                t_coal += ms;
            }
            CUDA_CHECK(cudaMemcpy(dC, C_ref.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(start));
            run_cublas_sgemm(handle, M, N, K, alpha, A.data(), B.data(), beta, C_ref.data());
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            {
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                t_cublas += ms;
            }
            CUDA_CHECK(cudaMemcpy(dC, C_shared.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(start));
            run_sgemm_shmem_cache_blocking(M, N, K, alpha, dA, dB, beta, dC);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            {
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                t_shared += ms;
            }
        }
        std::cout
            << "dim " << size
            << " | naive: "   << (t_naive   / NUM_RUNS) << " ms"
            << " | coal: "    << (t_coal    / NUM_RUNS) << " ms"
            << " | cuBLAS: "  << (t_cublas  / NUM_RUNS) << " ms"
            << " | shared: "  << (t_shared  / NUM_RUNS) << " ms\n";
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
*/
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

int main()
{
    int deviceIdx = 0;

    if (const char* env = std::getenv("DEVICE"))
    {
        deviceIdx = std::atoi(env);
    }

    CUDA_CHECK(cudaSetDevice(deviceIdx));

    std::cout << "Running on device " << deviceIdx << "\n";

    cublasHandle_t handle;

    CUBLAS_CHECK(cublasCreate(&handle));

    std::vector<int> sizes = {128, 256, 512, 1024};

    int max_size = sizes.back();

    size_t max_bytes = sizeof(float) * max_size * max_size;

    std::vector<float> A(max_size * max_size);
    std::vector<float> B(max_size * max_size);
    std::vector<float> C0(max_size * max_size);
    std::vector<float> C_ref(max_size * max_size);
    std::vector<float> C_naive(max_size * max_size);
    std::vector<float> C_coal(max_size * max_size);
    std::vector<float> C_shared(max_size * max_size);
    std::vector<float> C_1D_block_tile(max_size * max_size);
    std::vector<float> C_2D_block_tile(max_size * max_size);

    randomizeMatrix(A.data(), A.size());
    randomizeMatrix(B.data(), B.size());
    rangeInitMatrix(C0.data(), C0.size());

    C_ref = C0;
    C_naive = C0;
    C_coal = C0;
    C_shared = C0;
    C_1D_block_tile = C0;
    C_2D_block_tile = C0;

    float* dA;
    float* dB;
    float* dC;

    CUDA_CHECK(cudaMalloc(&dA, max_bytes));
    CUDA_CHECK(cudaMalloc(&dB, max_bytes));
    CUDA_CHECK(cudaMalloc(&dC, max_bytes));

    CUDA_CHECK(cudaMemcpy(dA, A.data(), max_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), max_bytes, cudaMemcpyHostToDevice));

    float alpha = 2.0f;
    float beta  = 3.0f;

    {
        int M = sizes[0];
        int N = M;
        int K = M;

        CUDA_CHECK(cudaMemcpy(dC, C_naive.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
        run_sgemm_naive(M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(C_naive.data(), dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost));

        run_cublas_sgemm(handle, M, N, K, alpha, A.data(), B.data(), beta, C_ref.data());

        if (!verifyMatrix(C_ref.data(), C_naive.data(), M, N, error_log_file))
        {
            std::cout << "Error with naive\n";
            return 1;
        }

        CUDA_CHECK(cudaMemcpy(dC, C_coal.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
        run_sgemm_coalesced(M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(C_coal.data(), dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost));

        if (!verifyMatrix(C_ref.data(), C_coal.data(), M, N, error_log_file))
        {
            std::cout << "Error with coalesced\n";
            return 1;
        }

        CUDA_CHECK(cudaMemcpy(dC, C_shared.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
        run_sgemm_shmem_cache_blocking(M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(C_shared.data(), dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost));

        if (!verifyMatrix(C_ref.data(), C_shared.data(), M, N, error_log_file))
        {
            std::cout << "Issue with Shmem Cache Blocking Kernel\n";
            return 1;
        }
    }

    const int NUM_RUNS = 10;

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int size : sizes)
    {
        int M = size;
        int N = size;
        int K = size;

        float t_naive   = 0.0f;
        float t_coal    = 0.0f;
        float t_cublas  = 0.0f;
        float t_shared  = 0.0f;

        for (int i = 0; i < NUM_RUNS; ++i)
        {
            CUDA_CHECK(cudaMemcpy(dC, C_naive.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(start));
            run_sgemm_naive(M, N, K, alpha, dA, dB, beta, dC);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            {
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                t_naive += ms;
            }

            CUDA_CHECK(cudaMemcpy(dC, C_coal.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(start));
            run_sgemm_coalesced(M, N, K, alpha, dA, dB, beta, dC);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            {
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                t_coal += ms;
            }

            CUDA_CHECK(cudaMemcpy(dC, C_ref.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(start));
            run_cublas_sgemm(handle, M, N, K, alpha, A.data(), B.data(), beta, C_ref.data());
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            {
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                t_cublas += ms;
            }

            CUDA_CHECK(cudaMemcpy(dC, C_shared.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(start));
            run_sgemm_shmem_cache_blocking(M, N, K, alpha, dA, dB, beta, dC);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            {
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                t_shared += ms;
            }
        }

        std::cout
            << "dim " << size
            << " | naive: "   << (t_naive   / NUM_RUNS) << " ms"
            << " | coal: "    << (t_coal    / NUM_RUNS) << " ms"
            << " | cuBLAS: "  << (t_cublas  / NUM_RUNS) << " ms"
            << " | shared: "  << (t_shared  / NUM_RUNS) << " ms\n";
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUBLAS_CHECK(cublasDestroy(handle));

    return 0;
}
