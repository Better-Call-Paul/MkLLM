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
#include <iomanip>

#define CUDA_CHECK(err) cudaCheck(err, __FILE__, __LINE__)

void cudaCheck(cudaError_t error, const char* file, int line);

void cudaDeviceInfo();

using CpuClock     = std::chrono::high_resolution_clock;
using CpuTimePoint = std::chrono::time_point<CpuClock>;

inline CpuTimePoint cpuNow()
{
    return CpuClock::now();
}

inline double cpuElapsed(const CpuTimePoint& start, const CpuTimePoint& end)
{
    return std::chrono::duration<double>(end - start).count();
}

template<typename F>
inline float timeKernel(F kernelLaunch, cudaStream_t stream = 0)
{
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
inline void rangeInitMatrix(T* mat, int n)
{
    for (int i = 0; i < n; ++i)
    {
        mat[i] = static_cast<T>(i);
    }
}

template<typename T>
inline void randomizeMatrix(T* mat, int n)
{
    std::srand(static_cast<unsigned>(cpuNow().time_since_epoch().count()));

    for (int i = 0; i < n; ++i)
    {
        auto tmp = static_cast<float>((std::rand() % 5) + 0.01f * (std::rand() % 5));
        mat[i] = static_cast<T>((std::rand() % 2) ? tmp : -tmp);
    }
}

template<typename T>
inline void zeroInitMatrix(T* mat, int n)
{
    std::fill(mat, mat + n, static_cast<T>(0));
}

template<typename T>
inline void copyMatrix(const T* src, T* dest, int n)
{
    std::copy(src, src + n, dest);
}

template<typename T>
inline void printMatrix(const T* a, int m, int n, std::ofstream& fs)
{
    fs << std::fixed << std::setprecision(2) << "[";

    for (int i = 0; i < m * n; ++i)
    {
        fs << std::setw(5) << a[i] << ((i % n == n - 1) ? "" : ", ");
        if ((i + 1) % n == 0 && i + 1 < m * n)
        {
            fs << ";\n";
        }
    }

    fs << "]\n";
}

template<typename T>
inline bool verifyMatrix(const T* ref, const T* out, int M, int N, const std::string& log_file = "")
{
    std::ofstream log;

    if (!log_file.empty()) 
    {
        log.open(log_file);
    }

    bool pass = true;

    for (int i = 0; i < M * N; ++i)
    {
        T diff = std::fabs(ref[i] - out[i]);
        if (diff > static_cast<T>(1e-2))
        {
            int row = i / N;
            int col = i % N;

            std::cerr << "Mismatch at (" << row << ", " << col << "): "
                      << ref[i] << " vs " << out[i] << " [diff=" << diff << "]\n";

            if (log.is_open()) {
                log << "Mismatch at (" << row << ", " << col << "): "
                    << ref[i] << " vs " << out[i] << " [diff=" << diff << "]\n";
            }

            pass = false;
        }
    }

    if (log.is_open()) log.close();

    return pass;
}
