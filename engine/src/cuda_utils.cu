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

void cudaDeviceInfo()
{
    int deviceId = 0;
    CUDA_CHECK(cudaGetDevice(&deviceId));

    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, deviceId));

    std::cout
        << "Device ID: "                      << deviceId                                       << "\n"
        << "Name: "                           << props.name                                     << "\n"
        << "Compute Capability: "            << props.major << "." << props.minor               << "\n"
        << "Memory Bus Width: "              << props.memoryBusWidth                            << "\n"
        << "Max Threads Per Block: "         << props.maxThreadsPerBlock                        << "\n"
        << "Max Threads Per Multiprocessor: "<< props.maxThreadsPerMultiProcessor                << "\n"
        << "Regs Per Block: "                << props.regsPerBlock                              << "\n"
        << "Regs Per Multiprocessor: "       << props.regsPerMultiprocessor                     << "\n"
        << "Total Global Memory: "           << (props.totalGlobalMem   / 1024 / 1024) << " MB\n"
        << "Shared Mem Per Block: "          << (props.sharedMemPerBlock / 1024)      << " KB\n"
        << "Shared Mem Per Multiprocessor: " << (props.sharedMemPerMultiprocessor / 1024) << " KB\n"
        << "Total Constant Memory: "         << (props.totalConstMem     / 1024)      << " KB\n"
        << "Multiprocessor Count: "          << props.multiProcessorCount                       << "\n"
        << "Warp Size: "                     << props.warpSize                                  << "\n";
}

}
