#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>

#define CUDA_CHECK(err) engine::cudaCheck(err, __FILE__, __LINE__)

namespace Engine
{
// utils/include/utils.cuh
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <cmath>

namespace engine {

void cudaCheck(cudaError_t error, const char* file, int line);

void cudaDeviceInfo();

using CpuClock     = std::chrono::high_resolution_clock;
using CpuTimePoint = std::chrono::time_point<CpuClock>;

inline CpuTimePoint cpuNow() {
    return CpuClock::now();
}

inline double cpuElapsed(const CpuTimePoint& start,
                         const CpuTimePoint& end) {
    return std::chrono::duration<double>(end - start).count();
}

template<typename F>
inline float timeKernel(F kernelLaunch, cudaStream_t stream = 0) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));
    kernelLaunch();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}


template<typename T>
inline void rangeInitMatrix(T* mat, int n) {
    for (int i = 0; i < n; ++i) mat[i] = static_cast<T>(i);
}

template<typename T>
inline void randomizeMatrix(T* mat, int n) {
    std::srand(static_cast<unsigned>(cpuNow().time_since_epoch().count()));
    for (int i = 0; i < n; ++i) {
        float tmp = (std::rand() % 5) + 0.01f * (std::rand() % 5);
        mat[i] = static_cast<T>((std::rand()%2) ? tmp : -tmp);
    }
}

template<typename T>
inline void zeroInitMatrix(T* mat, int n) {
    std::fill(mat, mat + n, static_cast<T>(0));
}

template<typename T>
inline void copyMatrix(const T* src, T* dest, int n) {
    std::copy(src, src + n, dest);
}

template<typename T>
inline void printMatrix(const T* a, int m, int n, std::ofstream& fs) {
    fs << std::fixed << std::setprecision(2) << "[";
    for (int i = 0; i < m * n; ++i) {
        fs << std::setw(5) << a[i]
           << ((i % n == n - 1) ? "" : ", ");
        if ((i + 1) % n == 0 && i + 1 < m * n) fs << ";\n";
    }
    fs << "]\n";
}

template<typename T>
inline bool verifyMatrix(const T* ref, const T* out, int n) {
    for (int i = 0; i < n; ++i) {
        if (std::fabs(ref[i] - out[i]) > static_cast<T>(1e-2)) {
            std::cerr << "Mismatch at " << i
                      << ": " << ref[i]
                      << " vs "  << out[i] << "\n";
            return false;
        }
    }
    return true;
}

} 

}
