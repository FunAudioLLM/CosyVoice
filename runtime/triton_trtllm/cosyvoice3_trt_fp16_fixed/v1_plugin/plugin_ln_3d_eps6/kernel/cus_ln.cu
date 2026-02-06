#include "cus_ln.h"

template<typename T, const int kWarpSize = WARP_SIZE>
DEVICE_INLINE T warp_reduce_sum(T val) 
{
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) 
    {
        val += __shfl_xor_sync(0xffffffff, val, mask, kWarpSize);
    }
    return val;
}

template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) 
{
    // always <= 32 warps per block (limited by 1024 threads per block)
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ float shared[NUM_WARPS];

    val = warp_reduce_sum<float, WARP_SIZE>(val);
    if (lane == 0)
        shared[warp] = val;
    __syncthreads();
    val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
    val = warp_reduce_sum<float, WARP_SIZE>(val);
    return val;
}

template <const int NUM_THREADS = 256>
__global__ void layernorm_fp32x4_kernel(float* inputs, float* outputs, float* gamma, float* beta,
    int batch, int in_shape1, int in_shape2)
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx4 = tx * 4;

    __shared__ float s_mean;                     // shared within block
    __shared__ float s_variance;                 // shared within block

    if (tx4 < in_shape2)
    {
        const int pos = by * in_shape1 * in_shape2 + bx * in_shape2 + tx4;

        // load 128bit
        float4 reg_inputs = LDST128BITS(inputs[pos]);

        float value = reg_inputs.x + reg_inputs.y + reg_inputs.z + reg_inputs.w;
        float sum = block_reduce_sum_f32<NUM_THREADS>(value);
        if (tx == 0)
            s_mean = sum / float(in_shape2);
        __syncthreads();

        float4 reg_val_cent;
        reg_val_cent.x = reg_inputs.x - s_mean;
        reg_val_cent.y = reg_inputs.y - s_mean;
        reg_val_cent.z = reg_inputs.z - s_mean;
        reg_val_cent.w = reg_inputs.w - s_mean;
        value = reg_val_cent.x * reg_val_cent.x + reg_val_cent.y * reg_val_cent.y + reg_val_cent.z * reg_val_cent.z + reg_val_cent.w * reg_val_cent.w;
        sum = block_reduce_sum_f32<NUM_THREADS>(value);
        if (tx == 0)
            s_variance = sum / float(in_shape2);
        __syncthreads();

        float4 reg_outputs;
        float4 reg_gamma = LDST128BITS(gamma[tx4]);
        float4 reg_beta = LDST128BITS(beta[tx4]);
        float tmp = rsqrtf(s_variance + eps);
        reg_outputs.x = (reg_val_cent.x * tmp) * reg_gamma.x + reg_beta.x;
        reg_outputs.y = (reg_val_cent.y * tmp) * reg_gamma.y + reg_beta.y;
        reg_outputs.z = (reg_val_cent.z * tmp) * reg_gamma.z + reg_beta.z;
        reg_outputs.w = (reg_val_cent.w * tmp) * reg_gamma.w + reg_beta.w;

        LDST128BITS(outputs[pos]) = reg_outputs;
    }
}

cudaError_t layernorm_fp32x4_ld_launch(float* inputs, float* outputs, float* gamma, float* beta,
    int batch, int in_shape1, int in_shape2, cudaStream_t stream)
{
    if (!inputs || !outputs)
        return cudaErrorInvalidDevicePointer;

    int num = int(in_shape2 * 0.25);
    const dim3 blockDim(num, 1, 1);
    const dim3 gridDim(in_shape1, batch, 1);
    layernorm_fp32x4_kernel <256> << <gridDim, blockDim, 0, stream >> > (inputs, outputs, gamma, beta, batch, in_shape1, in_shape2);

    return CUDA(cudaGetLastError());
}

template <const int NUM_THREADS = 256>
__global__ void layernorm_f16x8_pack_f16_acc_kernel(half* inputs, half* outputs, half* gamma, half* beta,
    int batch, int in_shape1, int in_shape2)
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx8 = tx * 8;

    __shared__ float s_mean;
    __shared__ float s_variance;

    if (tx8 < in_shape2)
    {
        const int pos = by * in_shape1 * in_shape2 + bx * in_shape2 + tx8;

        half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.
        LDST128BITS(pack_x[0]) = LDST128BITS(inputs[pos]);

        float pack_xf[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) 
            pack_xf[i] = __half2float(pack_x[i]);

        float value = pack_xf[0] + pack_xf[1] + pack_xf[2] + pack_xf[3] + pack_xf[4] + pack_xf[5] + pack_xf[6] + pack_xf[7];
        float sum = block_reduce_sum_f32<NUM_THREADS>(value);
        if (tx == 0)
            s_mean = sum / (float)in_shape2;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 8; ++i) 
            pack_xf[i] = pack_xf[i] - s_mean;
 
        value = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) 
            value += (pack_xf[i] * pack_xf[i]);

        sum = block_reduce_sum_f32<NUM_THREADS>(value);
        if (tx == 0)
            s_variance = sum / (float)in_shape2;
        __syncthreads();

        half pack_g[8], pack_b[8];
        LDST128BITS(pack_g[0]) = LDST128BITS(gamma[tx8]);
        LDST128BITS(pack_b[0]) = LDST128BITS(beta[tx8]);
        float tmp = rsqrtf(s_variance + eps);

        #pragma unroll
        for (int i = 0; i < 8; ++i) 
        {
            // pack_y[i] = __hfma(__float2half(pack_xf[i] * tmp), pack_g[i], pack_b[i]);
            pack_y[i] = __float2half(fmaf(pack_xf[i] * tmp, __half2float(pack_g[i]), __half2float(pack_b[i])));
        }
        LDST128BITS(outputs[pos]) = LDST128BITS(pack_y[0]);
    }
}

cudaError_t layernorm_f16x8_pack_f16_acc_ld_launch(half* inputs, half* outputs, half* gamma, half* beta,
    int batch, int in_shape1, int in_shape2, cudaStream_t stream)
{
    if (!inputs || !outputs)
        return cudaErrorInvalidDevicePointer;

    int num = int(in_shape2 * 0.125);
    const dim3 blockDim(num, 1, 1);
    const dim3 gridDim(in_shape1, batch, 1);
    layernorm_f16x8_pack_f16_acc_kernel <128> << <gridDim, blockDim, 0, stream >> > (inputs, outputs, gamma, beta, batch, in_shape1, in_shape2);

    return CUDA(cudaGetLastError());
}
