#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <nvrtc.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / N)