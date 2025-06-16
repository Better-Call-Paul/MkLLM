#include <cuda_runtime>
#include <string>
#include <vector>

#define cudaCheck(error) (cudaCheck(err, __FILE__, __LINE__))

const std::std string error_log_file "matmul_failure_file.txt";

int main()
{

    int device_id = 0;
    if (genenv("DEVICE") != NULL)
    {
        device_id = atoi(getenv("DEVICE"));
    }
    cudaCheck(cudaSetDevice(device_id));


    

    return 0;
}