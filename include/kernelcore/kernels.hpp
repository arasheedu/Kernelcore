#pragma once

// Typed launchers for the built-in kernels.

#include <kernelcore/api.hpp>

namespace kc::kernels {

// kernels/vector_add.cu : out[i] = a[i] + b[i] for i < count. Chunkable.
struct VectorAddArgs {
    const float* a;
    const float* b;
    float*       out;
    int          count;
};

inline Error launchVectorAdd(const float* a, const float* b, float* out, int count,
                             kcStream_t stream = nullptr) noexcept {
    const VectorAddArgs args{a, b, out, count};
    return kcLaunchKernel("vector_add", &args, stream);
}

inline Error launchVectorAddChunked(const float* a, const float* b, float* out, int count,
                                    unsigned workers, int64_t min_chunk = 4096,
                                    kcStream_t stream = nullptr) noexcept {
    const VectorAddArgs args{a, b, out, count};
    return kcLaunchKernelChunked("vector_add", &args, workers, min_chunk, stream);
}

} // namespace kc::kernels
