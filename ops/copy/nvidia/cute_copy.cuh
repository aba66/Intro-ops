#pragma once

#include <cute/tensor.hpp>
#include <cuda_runtime.h>

#include <cstdint>

namespace oprt::copy::nvidia {

template <typename T, int Threads, int ElementsPerAccess>
__global__ void copy_contiguous_cute_l40_kernel(T *dst, const T *src, int64_t n) {
    using namespace cute;

    constexpr int elements_per_block = Threads * ElementsPerAccess;

    auto src_tensor = make_tensor(make_gmem_ptr(src), make_layout(make_shape(n)));
    auto dst_tensor = make_tensor(make_gmem_ptr(dst), make_layout(make_shape(n)));

    auto block_shape = make_shape(Int<elements_per_block>{});
    auto block_coord = make_coord(blockIdx.x);

    auto coords = make_identity_tensor(shape(src_tensor));
    auto src_tile = local_tile(src_tensor, block_shape, block_coord);
    auto dst_tile = local_tile(dst_tensor, block_shape, block_coord);
    auto coord_tile = local_tile(coords, block_shape, block_coord);

    auto thread_layout = make_layout(make_shape(Int<Threads>{}));
    auto value_layout = make_layout(make_shape(Int<ElementsPerAccess>{}));

    using AccessType = uint_byte_t<sizeof(T) * ElementsPerAccess>;
    using CopyOp = UniversalCopy<AccessType>;
    using Atom = Copy_Atom<CopyOp, T>;

    auto tiled_copy = make_tiled_copy(Atom{}, thread_layout, value_layout);
    auto thread_copy = tiled_copy.get_thread_slice(threadIdx.x);

    auto thread_src = thread_copy.partition_S(src_tile);
    auto thread_dst = thread_copy.partition_D(dst_tile);
    auto thread_coord = thread_copy.partition_S(coord_tile);

    CUTE_UNROLL
    for (int i = 0; i < size(thread_dst); ++i) {
        if (elem_less(thread_coord(i), shape(src_tensor))) {
            thread_dst(i) = thread_src(i);
        }
    }
}

template <typename T>
inline void launch_copy_contiguous_cute_l40(T *dst, const T *src, int64_t n, cudaStream_t stream) {
    constexpr int threads = 256;
    constexpr int bytes_per_access = 16;
    constexpr int elements_per_access = bytes_per_access / static_cast<int>(sizeof(T));
    static_assert(elements_per_access > 0, "copy element type is larger than the vector access");

    if (n <= 0) {
        return;
    }

    auto src_addr = reinterpret_cast<std::uintptr_t>(src);
    auto dst_addr = reinterpret_cast<std::uintptr_t>(dst);
    bool aligned = (src_addr % bytes_per_access == 0) && (dst_addr % bytes_per_access == 0);

    if (aligned) {
        constexpr int elements_per_block = threads * elements_per_access;
        int64_t blocks64 = (n + elements_per_block - 1) / elements_per_block;
        int blocks = static_cast<int>(blocks64 > 0 ? blocks64 : 1);
        copy_contiguous_cute_l40_kernel<T, threads, elements_per_access><<<blocks, threads, 0, stream>>>(
            dst, src, n);
    } else {
        int64_t blocks64 = (n + threads - 1) / threads;
        int blocks = static_cast<int>(blocks64 > 0 ? blocks64 : 1);
        copy_contiguous_cute_l40_kernel<T, threads, 1><<<blocks, threads, 0, stream>>>(
            dst, src, n);
    }
}

} // namespace oprt::copy::nvidia
