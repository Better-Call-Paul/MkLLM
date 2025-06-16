#include "cuda_utils.cuh"

namespace Engine 
{

void cudaCheck(cudaError_t error, const char* file, int line)
{
    if (error != cudaSuccess) 
    {
        std::fprintf(stderr,
                     "[CUDA ERROR] %s:%d %s\n",
                     file, line, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

void cudaDeviceInfo() {
    int id; CUDA_CHECK(cudaGetDevice(&id));
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, id));
    std::cout
        << "Device ID: "          << id                    << "\n"
        << "Name: "               << p.name                << "\n"
        << "Compute Capability: " << p.major << "." << p.minor << "\n"
        << "Total Global Mem: "   << (p.totalGlobalMem/1024/1024)
        << " MB\n";
}

}