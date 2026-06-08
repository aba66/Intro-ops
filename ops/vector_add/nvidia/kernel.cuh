#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

namespace oprt::vector_add::nvidia {

template <typename T>
__device__ T add_values(T a, T b) {
    return a + b;
}

template <>
__device__ inline half add_values<half>(half a, half b) {
    return __hadd(a, b);
}

template <typename T>
__global__ void vector_add_contiguous_kernel(T * __restrict__ out,
                                             const T * __restrict__ a,
                                             const T * __restrict__ b,
                                             int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        out[i] = add_values(a[i], b[i]);
    }
}

} // namespace oprt::vector_add::nvidia
