// The same kernel sources that Booth compiles for the CPU, compiled here by nvcc.
// They are #included unchanged so the two sides cannot drift apart.

#include "cuda_kernels.hpp"

#include <cuda_runtime.h>

#include "kernels/vector_add.cu"
#include "kernels/test/scale_add.cu"
#include "kernels/test/trig_mix.cu"

namespace gpu {
namespace {

unsigned blocksFor(int count) { return static_cast<unsigned>((count + kBlockSize - 1) / kBlockSize); }

} // namespace

cudaError_t vectorAdd(const float* a, const float* b, float* out, int count, cudaStream_t stream) {
    if (count <= 0) return cudaSuccess;
    vector_add<<<blocksFor(count), kBlockSize, 0, stream>>>(a, b, out, count);
    return cudaGetLastError();
}

cudaError_t scaleAdd(const float* x, float* out, float scale, float bias, int count, cudaStream_t stream) {
    if (count <= 0) return cudaSuccess;
    scale_add<<<blocksFor(count), kBlockSize, 0, stream>>>(x, out, scale, bias, count);
    return cudaGetLastError();
}

cudaError_t trigMix(const float* x, float* out, int count, cudaStream_t stream) {
    if (count <= 0) return cudaSuccess;
    trig_mix<<<blocksFor(count), kBlockSize, 0, stream>>>(x, out, count);
    return cudaGetLastError();
}

} // namespace gpu
