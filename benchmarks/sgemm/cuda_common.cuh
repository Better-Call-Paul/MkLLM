#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <string>
#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <nvrtc.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / N)

inline void cudaCheck(cudaError_t err, const char* file, int line) 
{
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

void randomizeMatrix(float* mat, size_t size) 
{
    static std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) mat[i] = dist(rng);
}

void rangeInitMatrix(float* mat, size_t size) 
{
    for (size_t i = 0; i < size; ++i) mat[i] = float(i);
}

bool verifyMatrix(const float* ref, const float* out,
                  int M, int N, const std::string& logfile) 
{
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
