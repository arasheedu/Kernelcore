#pragma once

// Host-callable launchers for the nvcc-compiled copies of the kernels in kernels/.
// Plain C++ interface so the comparison app is compiled by cl, not nvcc.
// All pointers are CUDA device pointers. Each launcher only enqueues the kernel.

#include <cuda_runtime_api.h>

namespace gpu {

constexpr int kBlockSize = 256;

cudaError_t vectorAdd(const float* a, const float* b, float* out, int count, cudaStream_t stream);
cudaError_t scaleAdd(const float* x, float* out, float scale, float bias, int count, cudaStream_t stream);
cudaError_t trigMix(const float* x, float* out, int count, cudaStream_t stream);

} // namespace gpu
