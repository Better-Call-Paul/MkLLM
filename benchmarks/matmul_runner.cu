#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CEIL_DIV(x,y) (((x)+(y)-1)/(y))

inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << file << ":" << line
                  << " CUDA Error: " << cudaGetErrorString(err)
                  << " (" << err << ")\n";
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(call) cudaCheck(call, __FILE__, __LINE__)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                         \
        cublasStatus_t status = (call);                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                   \
            std::cerr << __FILE__ << ":" << __LINE__                             \
                      << " cuBLAS error " << status << "\n";                     \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while(0)

const std::string error_log_file = "matmul_failure_file.txt";

void randomizeMatrix(float* mat, size_t size) {
    static std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) mat[i] = dist(rng);
}

void rangeInitMatrix(float* mat, size_t size) {
    for (size_t i = 0; i < size; ++i) mat[i] = float(i);
}

bool verifyMatrix(const float* ref, const float* out,
                  int M, int N, const std::string& logfile) {
    std::ofstream ofs(logfile, std::ios::app);
    const float tol = 1e-2f;
    bool any_error = false;
    int total = M * N;
    for (int i = 0; i < total; ++i) {
        float d = std::abs(ref[i] - out[i]);
        if (d > tol) {
            if (!any_error) {
                ofs << "=== MISMATCH for " << M << "x" << N << " ===\n";
                any_error = true;
            }
            ofs << "idx " << i
                << ": ref=" << std::setprecision(6) << ref[i]
                << ", out="  << out[i]
                << ", diff=" << d << "\n";
        }
    }
    if (!any_error) {
        ofs << "Verify passed for " << M << "x" << N << "\n";
        return true;
    }
    return false;
}

inline void checkKernelLaunch(int) { }
inline void checkKernelSync(int)  { }

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

template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(
    int M,
    int N,
    int K,
    float alpha,
    const float *A,
    const float *B,
    float beta,
    float *C)
{
    const uint cRow = blockDim.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const uint cCol = blockDim.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (cRow < M && cCol < N) 
    {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) 
        {
            sum += A[cRow * K + i] * B[i * N + cCol];
        }
        C[cRow * N + cCol] = alpha * sum + beta * C[cRow + N + cCol];
    }
}

inline void run_sgemm_coalesced(
    int M,
    int N,
    int K,
    float alpha,
    const float *A,
    const float *B,
    float beta,
    float *C)
{
    constexpr uint blockSize = 32;
    dim3 blockDim(blockSize * blockSize, 1);
    dim3 gridDim(CEIL_DIV(M, blockSize), CEIL_DIV(N, blockSize));
    sgemm_global_mem_coalesce<blockSize><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_naive(int M, int N, int K,
                     float alpha, const float *A,
                     const float *B, float beta, float *C) {
    dim3 blk(32,32), grid(CEIL_DIV(N,32), CEIL_DIV(M,32));
    sgemm_naive<<<grid,blk>>>(M,N,K,alpha,A,B,beta,C);
}

void run_cublas_sgemm(cublasHandle_t handle,
                      int M, int N, int K,
                      float alpha,
                      const float *A, const float *B,
                      float beta, float *C) {
    size_t sizeA = sizeof(float) * M * K;
    size_t sizeB = sizeof(float) * K * N;
    size_t sizeC = sizeof(float) * M * N;
    float *dA, *dB, *dC;

    CUDA_CHECK(cudaMalloc((void**)&dA, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&dB, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&dC, sizeC));

    CUDA_CHECK(cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, C, sizeC, cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              dB, CUDA_R_32F, N,
                              dA, CUDA_R_32F, K,
                              &beta,
                              dC, CUDA_R_32F, N,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C, dC, sizeC, cudaMemcpyDeviceToHost));

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

template <const int BLOCKSIZE>
__global__ void sgemm_shmem_cache_blocking(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C)
{
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    const uint threadCol = threadIdx.x % BLOCKSIZE;
    const uint threadRow = threadIdx.x / BLOCKSIZE;

    A += cRow * BLOCKSIZE * K;
    B += cCol * BLOCKSIZE;
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

    float tmp = 0.0f;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE)
    {
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        __syncthreads();
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
        {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                   Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        __syncthreads();
    }
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}

void run_sgemm_shmem_cache_blocking(int M, int N, int K,
                      float alpha,
                      const float *A, const float *B,
                      float beta, float *C)
{
    constexpr uint block_size = 32;
    dim3 blockDim(block_size * block_size, 1);

    dim3 gridDim(CEIL_DIV(M, block_size), CEIL_DIV(N, block_size));

    cudaFuncSetAttribute(sgemm_shmem_cache_blocking<32>, cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);

    sgemm_shmem_cache_blocking<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
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
    std::vector<float> C(max_size * max_size);
    std::vector<float> C_ref = C;

    randomizeMatrix(A.data(), A.size());
    randomizeMatrix(B.data(), B.size());
    rangeInitMatrix(C.data(), C.size());
    C_ref = C;

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

        CUDA_CHECK(cudaMemcpy(dC, C.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice));
        run_sgemm_naive(M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(C.data(), dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
        run_cublas_sgemm(handle, M, N, K, alpha, A.data(), B.data(), beta, C_ref.data());
        if (!verifyMatrix(C_ref.data(), C.data(), M, N, error_log_file))
        {
            return 1;
        }

        CUDA_CHECK(cudaMemcpy(dC, C.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice));
        run_sgemm_coalesced(M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(C.data(), dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
        if (!verifyMatrix(C_ref.data(), C.data(), M, N, error_log_file))
        {
            return 1;
        }

        CUDA_CHECK(cudaMemcpy(dC, C.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice));
        run_sgemm_shmem_cache_blocking(M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(C.data(), dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
        if (!verifyMatrix(C_ref.data(), C.data(), M, N, error_log_file))
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
            CUDA_CHECK(cudaMemcpy(dC, C.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(start));
            run_sgemm_naive(M, N, K, alpha, dA, dB, beta, dC);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            {
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                t_naive += ms;
            }

            CUDA_CHECK(cudaMemcpy(dC, C.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(start));
            run_sgemm_coalesced(M, N, K, alpha, dA, dB, beta, dC);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            {
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                t_coal += ms;
            }

            CUDA_CHECK(cudaMemcpy(dC, C.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(start));
            run_cublas_sgemm(handle, M, N, K, alpha, A.data(), B.data(), beta, C_ref.data());
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            {
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                t_cublas += ms;
            }

            CUDA_CHECK(cudaMemcpy(dC, C.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice));
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
