#include "cuda_utils.cuh"
#include <string>
#include <vector>
#include <cstdlib>
#include <iostream>
#include "kernels.cuh"

#define CUDA_CHECK(err) Engine::cudaCheck(err, __FILE__, __LINE__)


const std::string error_log_file = "matmul_failure_file.txt";

int main()
{
    int device_idx = 0;

    if (const char* env_p = std::getenv("DEVICE"))
    {
        try
        {
            device_idx = std::stoi(env_p);
        }
        catch (const std::exception& e)
        {
            std::cerr << "Invalid DEVICE value: " << e.what() << '\n';
            return 1;
        }
    }
    CUDA_CHECK(cudaSetDevice(device_idx));

    CUDA_CHECK(cudaSetDevice(device_idx));

    const int M = 512, N = 512, K = 512;
    const size_t sizeA = size_t(M) * K;
    const size_t sizeB = size_t(K) * N;
    const size_t sizeC = size_t(M) * N;

    float* h_A = new float[sizeA];
    float* h_B = new float[sizeB];
    float* h_C = new float[sizeC];

    Engine::randomizeMatrix(h_A, sizeA);
    Engine::randomizeMatrix(h_B, sizeB);
    Engine::zeroInitMatrix(h_C, sizeC);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, sizeC * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    float ms = Engine::timeKernel([&]() {
        basic_sgemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    });

    std::cout << "basic_sgemm time: " << ms << " ms\n";

    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    delete[] h_A; delete[] h_B; delete[] h_C;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));


    return 0;
}