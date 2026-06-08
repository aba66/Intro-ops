#pragma once

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

namespace oprt::softmax::nvidia {

__global__ void softmax_rowwise_kernel(float *out, const float *in, int64_t rows, int64_t cols) {
    int64_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    extern __shared__ float shared[];
    int tid = threadIdx.x;
    const float *row_in = in + row * cols;
    float *row_out = out + row * cols;

    float local_max = -FLT_MAX;
    for (int64_t col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, row_in[col]);
    }
    shared[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = shared[0];

    float local_sum = 0.0f;
    for (int64_t col = tid; col < cols; col += blockDim.x) {
        float value = expf(row_in[col] - row_max);
        row_out[col] = value;
        local_sum += value;
    }
    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = shared[0];

    for (int64_t col = tid; col < cols; col += blockDim.x) {
        row_out[col] /= row_sum;
    }
}

} // namespace oprt::softmax::nvidia
