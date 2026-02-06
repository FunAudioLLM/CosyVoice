#pragma once

#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <map>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/extrema.h>
#include <cublas_v2.h>

/**
 * LOG_CUDA string.
 * @ingroup cudaError
 */
#define LOG_CUDA "[cuda]   "

 /**
  * cudaCheckError
  * @ingroup cudaError
  */
inline cudaError_t cudaCheckError(cudaError_t retval, const char* txt, const char* file, int line)
{
#if !defined(CUDA_TRACE)
    if (retval == cudaSuccess)
        return cudaSuccess;
#endif

    int activeDevice = -1;
    cudaGetDevice(&activeDevice);

    printf("[cuda]   device %i  -  %s\n", activeDevice, txt);
    printf(LOG_CUDA "%s\n", txt);

    if (retval != cudaSuccess)
    {
        printf(LOG_CUDA "   %s (error %u) (hex 0x%02X)\n", cudaGetErrorString(retval), retval, retval);
        printf(LOG_CUDA "   %s:%i\n", file, line);
    }

    return retval;
}

/**
 * Execute a CUDA call and print out any errors
 * @return the original cudaError_t result
 * @ingroup cudaError
 */
#define CUDA(x) cudaCheckError((x), #x, __FILE__, __LINE__)

 /**
  * If a / b has a remainder, round up.  This function is commonly using when launching
  * CUDA kernels, to compute a grid size inclusive of the entire dataset if it's dimensions
  * aren't evenly divisible by the block size.
  *
  * For example:
  *
  *    const dim3 blockDim(8,8);
  *    const dim3 gridDim(iDivUp(imgWidth,blockDim.x), iDivUp(imgHeight,blockDim.y));
  *
  * Then inside the CUDA kernel, there is typically a check that thread index is in-bounds.
  *
  * Without the use of iDivUp(), if the data dimensions weren't evenly divisible by the
  * block size, parts of the data wouldn't be covered by the grid and not processed.
  *
  * @ingroup cuda
  */
inline __device__ __host__ int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }