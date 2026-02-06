#pragma once
#include "cuda_utils_custom.h"

constexpr float eps = 1e-6f;

#define WARP_SIZE 32
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

cudaError_t layernorm_fp32x4_ld_launch(float* inputs, float* outputs, float* gamma, float* beta,
    int batch, int in_shape1, int in_shape2, cudaStream_t stream);

cudaError_t layernorm_f16x8_pack_f16_acc_ld_launch(half* inputs, half* outputs, half* gamma, half* beta,
    int batch, int in_shape1, int in_shape2, cudaStream_t stream);
