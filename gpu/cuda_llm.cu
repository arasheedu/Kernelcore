// The LLM decode kernels that Booth compiles for pcuda, compiled here by nvcc, unchanged.

#include "cuda_llm.hpp"

#include "cuda_kernels.hpp"  // kBlockSize

#include <cuda_runtime.h>

#include "kernels/llm/matvec_q8.cu"
#include "kernels/llm/matvec_q4.cu"
#include "kernels/llm/matvec_q4x8.cu"
#include "kernels/llm/rms_norm.cu"
#include "kernels/llm/rope.cu"
#include "kernels/llm/attn_scores.cu"
#include "kernels/llm/softmax.cu"
#include "kernels/llm/attn_values.cu"
#include "kernels/llm/silu_mul.cu"

namespace gpu::llm {
namespace {

unsigned blocksFor(int count) { return static_cast<unsigned>((count + kBlockSize - 1) / kBlockSize); }

} // namespace

cudaError_t matvecQ8(const int8_t* q, const float* scales, const float* x, float* y, const int* dims, int rows,
                     cudaStream_t s) {
    if (rows <= 0) return cudaSuccess;
    matvec_q8<<<blocksFor(rows), kBlockSize, 0, s>>>(reinterpret_cast<const signed char*>(q), scales, x, y, dims);
    return cudaGetLastError();
}

cudaError_t matvecQ4(const uint8_t* qs, const float* scales, const float* x, float* y, const int* dims, int rows,
                     cudaStream_t s) {
    if (rows <= 0) return cudaSuccess;
    matvec_q4<<<blocksFor(rows), kBlockSize, 0, s>>>(qs, scales, x, y, dims);
    return cudaGetLastError();
}

cudaError_t matvecQ4x8(const uint8_t* qs, const float* scales, const float* x, float* y, const int* dims, int rows,
                       cudaStream_t s) {
    if (rows <= 0) return cudaSuccess;
    matvec_q4x8<<<blocksFor(rows), kBlockSize, 0, s>>>(qs, scales, x, y, dims);
    return cudaGetLastError();
}

cudaError_t rmsNorm(const float* x, const float* w, float* y, int n, int rows, float eps, cudaStream_t s) {
    if (rows <= 0) return cudaSuccess;
    rms_norm<<<blocksFor(rows), kBlockSize, 0, s>>>(x, w, y, n, rows, eps);
    return cudaGetLastError();
}

cudaError_t rope(float* x, int pairs_per_head, int pos, int count, float log2_base, cudaStream_t s) {
    if (count <= 0) return cudaSuccess;
    ::rope<<<blocksFor(count), kBlockSize, 0, s>>>(x, pairs_per_head, pos, count, log2_base);
    return cudaGetLastError();
}

cudaError_t attnScores(const float* q, const float* k, float* scores, const int* dims, int total, float scale,
                       cudaStream_t s) {
    if (total <= 0) return cudaSuccess;
    attn_scores<<<blocksFor(total), kBlockSize, 0, s>>>(q, k, scores, dims, 0, scale);
    return cudaGetLastError();
}

cudaError_t softmax(float* x, int n, int rows, cudaStream_t s) {
    if (rows <= 0) return cudaSuccess;
    ::softmax<<<blocksFor(rows), kBlockSize, 0, s>>>(x, n, rows);
    return cudaGetLastError();
}

cudaError_t attnValues(const float* p, const float* v, float* out, const int* dims, int total, cudaStream_t s) {
    if (total <= 0) return cudaSuccess;
    attn_values<<<blocksFor(total), kBlockSize, 0, s>>>(p, v, out, dims, 0);
    return cudaGetLastError();
}

cudaError_t siluMul(const float* g, const float* u, float* out, int count, cudaStream_t s) {
    if (count <= 0) return cudaSuccess;
    silu_mul<<<blocksFor(count), kBlockSize, 0, s>>>(g, u, out, count);
    return cudaGetLastError();
}

} // namespace gpu::llm
