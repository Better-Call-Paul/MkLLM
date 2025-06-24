#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
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

void verifyMatrix(const float* ref, const float* out,
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
    if (!any_error)
        ofs << "Verify passed for " << M << "x" << N << "\n";
}

inline void checkKernelLaunch(int size) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "<<<KERNEL LAUNCH ERROR>>> size=" << size
                  << " : " << cudaGetErrorString(err)
                  << " (code " << err << ")\n";
    }
}

inline void checkKernelSync(int size) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "<<<KERNEL SYNC ERROR>>> size=" << size
                  << " : " << cudaGetErrorString(err)
                  << " (code " << err << ")\n";
        std::exit(EXIT_FAILURE);
    }
}

__global__ void sgemm_naive(int M, int N, int K,
                            float alpha, const float *A,
                            const float *B, float beta, float *C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < M && y < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i)
            tmp += A[x * K + i] * B[i * N + y];
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

void run_sgemm_naive(int M, int N, int K,
                     float alpha, float *A, float *B,
                     float beta, float *C) {
    dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blk(32, 32);
    sgemm_naive<<<grid, blk>>>(M, N, K, alpha, A, B, beta, C);
}

void run_cublas_sgemm(cublasHandle_t handle,
                      int M, int N, int K,
                      float alpha,
                      const float *A, const float *B,
                      float beta, float *C) {
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              B, CUDA_R_32F, N,
                              A, CUDA_R_32F, K,
                              &beta,
                              C, CUDA_R_32F, N,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

int main() {
    int deviceIdx = 0;
    if (const char* env = getenv("DEVICE")) deviceIdx = std::atoi(env);
    CUDA_CHECK(cudaSetDevice(deviceIdx));
    std::cout << "Running on device " << deviceIdx << "\n";

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    std::vector<int> sizes = {128};
    int max_size = sizes.back();
    size_t bytes = sizeof(float) * max_size * max_size;

    float *A   = (float*)malloc(bytes),
          *B   = (float*)malloc(bytes),
          *C   = (float*)malloc(bytes),
          *C_ref = (float*)malloc(bytes);
    float *dA, *dB, *dC, *dC_ref;

    randomizeMatrix(A, max_size*max_size);
    randomizeMatrix(B, max_size*max_size);
    rangeInitMatrix(C, max_size*max_size);

    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMalloc(&dC_ref, bytes));

    CUDA_CHECK(cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, C, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC_ref, C, bytes, cudaMemcpyHostToDevice));

    for (int size : sizes) {
        int M = size, N = size, K = size;
        std::cout << "dim " << size << ", α=2, β=3\n";

        run_sgemm_naive(M, N, K, 2.0f, dA, dB, 3.0f, dC);
        checkKernelLaunch(size);
        checkKernelSync(size);

        run_cublas_sgemm(handle, M, N, K, 2.0f, dA, dB, 3.0f, dC_ref);
        checkKernelSync(size);

        CUDA_CHECK(cudaMemcpy(C,     dC,     sizeof(float)*M*N, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(C_ref, dC_ref, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
        verifyMatrix(C_ref, C, M, N, error_log_file);
    }

    std::cout << "Stable\n";

    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
