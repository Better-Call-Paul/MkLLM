#include "cuda_utils.cuh"
#include <string>
#include <vector>
#include <cstdlib>
#include <iostream>

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

    return 0;
}