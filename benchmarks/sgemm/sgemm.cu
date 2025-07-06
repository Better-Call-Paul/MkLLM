#include "cuda_common.cuh"
#include "runner.cuh"
#include "sgemm_kernels.cuh"

int main()
{
    int deviceIdx = 0;
    if (const char* env = std::getenv("DEVICE"))
        deviceIdx = std::atoi(env);

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

    randomizeMatrix(A.data(), A.size());
    randomizeMatrix(B.data(), B.size());
    rangeInitMatrix(C0.data(), C0.size());

    C_ref = C0;
    C_naive = C0;
    C_coal = C0;
    C_shared = C0;
    C_1D_block_tile = C0;

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, max_bytes));
    CUDA_CHECK(cudaMalloc(&dB, max_bytes));
    CUDA_CHECK(cudaMalloc(&dC, max_bytes));

    CUDA_CHECK(cudaMemcpy(dA, A.data(), max_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), max_bytes, cudaMemcpyHostToDevice));

    float alpha = 2.0f;
    float beta  = 3.0f;

    {
        int M = sizes[0], N = M, K = M;

        CUDA_CHECK(cudaMemcpy(dC, C_naive.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
        run_sgemm_naive(M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(C_naive.data(), dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost));

        run_cublas_sgemm(handle, M, N, K, alpha, A.data(), B.data(), beta, C_ref.data());
        if (!verifyMatrix(C_ref.data(), C_naive.data(), M, N, error_log_file)) {
            std::cout << "Error with naive" << std::endl;
            return 1;
        }

        CUDA_CHECK(cudaMemcpy(dC, C_coal.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
        run_sgemm_coalesced(M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(C_coal.data(), dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
        if (!verifyMatrix(C_ref.data(), C_coal.data(), M, N, error_log_file)) {
            std::cout << "Error with coalesced" << std::endl;
            return 1;
        }

        CUDA_CHECK(cudaMemcpy(dC, C_shared.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
        run_sgemm_shmem_cache_blocking(M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(C_shared.data(), dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
        if (!verifyMatrix(C_ref.data(), C_shared.data(), M, N, error_log_file)) {
            std::cout << "Error with shared blocking" << std::endl;
            return 1;
        }

        CUDA_CHECK(cudaMemcpy(dC, C_1D_block_tile.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
        runSgemm1DBlocktiling(M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(C_1D_block_tile.data(), dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
        if (!verifyMatrix(C_ref.data(), C_1D_block_tile.data(), M, N, error_log_file)) {
            std::cout << "Error with 1D block-tiling" << std::endl;
            return 1;
        }
    }

    const int NUM_RUNS = 10;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << std::left
              << std::setw(8)  << "Dim"
              << std::setw(12) << "Kernel"
              << std::setw(12) << "Time(ms)"
              << std::setw(12) << "GFLOP/s"
              << std::endl;

    for (int size : sizes) {
        int M = size, N = size, K = size;
        double flops = 2.0 * double(M) * N * K;

        float t_naive=0, t_coal=0, t_cublas=0, t_shared=0, t_1d=0;
        for (int i=0; i<NUM_RUNS; ++i) {
            // naive
            CUDA_CHECK(cudaMemcpy(dC, C_naive.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(start));
            run_sgemm_naive(M,N,K,alpha,dA,dB,beta,dC);
            CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&t_naive, start, stop));

            // coalesced
            CUDA_CHECK(cudaMemcpy(dC, C_coal.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(start));
            run_sgemm_coalesced(M,N,K,alpha,dA,dB,beta,dC);
            CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&t_coal, start, stop));

            // cuBLAS
            CUDA_CHECK(cudaMemcpy(dC, C_ref.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(start));
            run_cublas_sgemm(handle, M,N,K,alpha,A.data(),B.data(),beta,C_ref.data());
            CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&t_cublas, start, stop));

            // shared
            CUDA_CHECK(cudaMemcpy(dC, C_shared.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(start));
            run_sgemm_shmem_cache_blocking(M,N,K,alpha,dA,dB,beta,dC);
            CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&t_shared, start, stop));

            // 1D block-tiling
            CUDA_CHECK(cudaMemcpy(dC, C_1D_block_tile.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(start));
            runSgemm1DBlocktiling(M,N,K,alpha,dA,dB,beta,dC);
            CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&t_1d, start, stop));
        }

        auto avg = [&](double t){ return t / NUM_RUNS; };
        auto gflops = [&](double ms){ return flops / (ms * 1e-3) / 1e9; };

        struct Row { const char* name; double t; } rows[] = {
            {"Naive",   avg(t_naive)},
            {"Coalesced",avg(t_coal)},
            {"cuBLAS",  avg(t_cublas)},
            {"Shared",  avg(t_shared)},
            {"1DBlk",   avg(t_1d)},
        };

        for (auto &r : rows) {
            std::cout << std::left
                      << std::setw(8)  << (std::to_string(size)+"x"+std::to_string(size))
                      << std::setw(12) << r.name
                      << std::setw(12) << std::fixed << std::setprecision(2) << r.t
                      << std::setw(12) << std::fixed << std::setprecision(2) << gflops(r.t)
                      << std::endl;
        }

        std::cout << std::endl;
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
