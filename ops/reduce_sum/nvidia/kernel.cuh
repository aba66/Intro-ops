#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace oprt::reduce_sum::nvidia {

__global__ void reduce_sum_rowwise_kernel(float *out, const float *in, int64_t rows, int64_t cols) {
    int64_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    extern __shared__ float shared[];
    int tid = threadIdx.x;
    float sum = 0.0f;
    const float *row_ptr = in + row * cols;
    for (int64_t col = tid; col < cols; col += blockDim.x) {
        sum += row_ptr[col];
    }

    shared[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[row] = shared[0];
    }
}

} // namespace oprt::reduce_sum::nvidia
