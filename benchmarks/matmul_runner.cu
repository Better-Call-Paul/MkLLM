// matmul_check.cpp
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

// CPU fallback for naive SGEMM
void run_sgemm_naive(int M, int N, int K,
                     float alpha, const float *A,
                     const float *B, float beta, float *C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float tmp = 0.0f;
            for (int k = 0; k < K; ++k)
                tmp += A[i*K + k] * B[k*N + j];
            C[i*N + j] = alpha * tmp + beta * C[i*N + j];
        }
    }
}

// GPU cuBLAS for reference
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
int main() {
    // select device
    int deviceIdx = 0;
    if (const char* env = std::getenv("DEVICE"))
        deviceIdx = std::atoi(env);
    CUDA_CHECK(cudaSetDevice(deviceIdx));
    std::cout << "Running on device " << deviceIdx << "\n";

    // create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // problem sizes
    std::vector<int> sizes = {128, 256, 512, 1024};
    int max_size = sizes.back();
    size_t bytes = sizeof(float) * max_size * max_size;

    // host buffers
    std::vector<float> A(max_size*max_size),
                       B(max_size*max_size),
                       C(max_size*max_size),
                       C_ref(max_size*max_size);

    randomizeMatrix(A.data(), A.size());
    randomizeMatrix(B.data(), B.size());
    rangeInitMatrix(C.data(), C.size());
    std::copy(C.begin(), C.end(), C_ref.begin());

    // initial correctness check
    {
        int M = sizes[0], N = M, K = M;
        run_sgemm_naive(M, N, K, 2.0f, A.data(), B.data(), 3.0f, C.data());
        checkKernelSync(M);
        run_cublas_sgemm(handle, M, N, K, 2.0f, A.data(), B.data(), 3.0f, C_ref.data());
        checkKernelSync(M);
        if (!verifyMatrix(C_ref.data(), C.data(), M, N, error_log_file)) {
            std::cerr << "Initial verification FAILED\n";
            return 1;
        }
        std::cout << "Initial verify passed\n\n";
    }

    // timing setup
    const int NUM_RUNS = 10;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // benchmark loop
    for (int size : sizes) {
        int M = size, N = size, K = size;
        float total_naive_ms = 0.0f, total_cublas_ms = 0.0f;

        std::cout << "dim " << size << ", α=2, β=3\n";

        for (int i = 0; i < NUM_RUNS; ++i) {
            // naive
            CUDA_CHECK(cudaEventRecord(start, 0));
            run_sgemm_naive(M, N, K, 2.0f, A.data(), B.data(), 3.0f, C.data());
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            total_naive_ms += ms;

            // cuBLAS
            CUDA_CHECK(cudaEventRecord(start, 0));
            run_cublas_sgemm(handle, M, N, K, 2.0f, A.data(), B.data(), 3.0f, C_ref.data());
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            total_cublas_ms += ms;
        }

        std::cout
            << "  avg naive:  " << (total_naive_ms  / NUM_RUNS) << " ms"
            << " | avg cuBLAS: " << (total_cublas_ms / NUM_RUNS) << " ms\n";
    }

    // cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
