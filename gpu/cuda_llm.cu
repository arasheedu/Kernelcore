// The LLM decode kernels that Booth compiles for pcuda, compiled here by nvcc, unchanged.

#include "cuda_llm.hpp"

#include "cuda_kernels.hpp"  // kBlockSize

#include <cuda_runtime.h>

#include "kernels/llm/matvec_q8.cu"
#include "kernels/llm/matvec_q4.cu"
#include "kernels/llm/matvec_q4x8.cu"
#include "kernels/llm/quantize_q8x4.cu"
#include "kernels/llm/matvec_q4x8_q8.cu"
#include "kernels/llm/matvec_q4x8_q8x2.cu"
#include "kernels/llm/add_rms_norm.cu"
#include "kernels/llm/silu_mul_q8x4.cu"
#include "kernels/llm/add_sumsq.cu"
#include "kernels/llm/norm_q8x4.cu"
#include "kernels/llm/rms_norm.cu"
#include "kernels/llm/rope.cu"
#include "kernels/llm/attn_scores.cu"
#include "kernels/llm/softmax.cu"
#include "kernels/llm/kv_append.cu"
#include "kernels/llm/attn_values.cu"
#include "kernels/llm/silu_mul.cu"
// K-quant kernels of the ggml backend (prompt processing and decode)
#include "kernels/llm/quantize_cols.cu"
#include "kernels/llm/matvec_q4k_q8.cu"
#include "kernels/llm/matvec_q6k_q8.cu"
#include "kernels/llm/matmul_q4k_q8x4.cu"
#include "kernels/llm/matmul_q6k_q8x4.cu"
#include "kernels/llm/matmul_q4k_q8x8.cu"
#include "kernels/llm/matmul_q6k_q8x8.cu"
#include "kernels/llm/quantize_cols_k.cu"
#include "kernels/llm/matmul_q4k_q8kx4.cu"
#include "kernels/llm/matmul_q6k_q8kx4.cu"
#include "kernels/llm/matmul_q4k_q8kx8.cu"
#include "kernels/llm/matmul_q6k_q8kx8.cu"

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

cudaError_t quantizeQ8x4(const float* x, int* xq, float* xd, int count, cudaStream_t s) {
    if (count <= 0) return cudaSuccess;
    quantize_q8x4<<<blocksFor(count), kBlockSize, 0, s>>>(x, xq, xd, count);
    return cudaGetLastError();
}

cudaError_t matvecQ4x8Q8x2(const int* qw, const int* xq, const float* xd, float* y, const int* dims, int rows,
                           cudaStream_t s) {
    if (rows <= 0) return cudaSuccess;
    matvec_q4x8_q8x2<<<blocksFor(rows / 2), kBlockSize, 0, s>>>(qw, xq, xd, y, dims);
    return cudaGetLastError();
}

cudaError_t matvecQ4x8Q8(const int* qw, const int* xq, const float* xd, float* y, const int* dims, int rows,
                         cudaStream_t s) {
    if (rows <= 0) return cudaSuccess;
    matvec_q4x8_q8<<<blocksFor(rows), kBlockSize, 0, s>>>(qw, xq, xd, y, dims);
    return cudaGetLastError();
}

cudaError_t addRmsNorm(float* x, const float* t, const float* w, float* y, int n, float eps, cudaStream_t s) {
    add_rms_norm<<<1, 1, 0, s>>>(x, t, w, y, n, eps);
    return cudaGetLastError();
}

cudaError_t addSumsq(float* x, const float* t, int* q, int nb, cudaStream_t s) {
    if (nb <= 0) return cudaSuccess;
    add_sumsq<<<blocksFor(nb), kBlockSize, 0, s>>>(x, t, q, nb, 0);
    return cudaGetLastError();
}

cudaError_t normQ8x4(const float* x, const float* w, int* q, int nb, float eps, cudaStream_t s) {
    if (nb <= 0) return cudaSuccess;
    norm_q8x4<<<blocksFor(nb), kBlockSize, 0, s>>>(x, w, q, nb, 0, eps);
    return cudaGetLastError();
}

cudaError_t siluMulQ8x4(const float* g, const float* u, int* xq, float* xd, int count, cudaStream_t s) {
    if (count <= 0) return cudaSuccess;
    silu_mul_q8x4<<<blocksFor(count), kBlockSize, 0, s>>>(g, u, xq, xd, count);
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

cudaError_t softmax(float* x, int n, int stride, int rows, cudaStream_t s) {
    if (rows <= 0) return cudaSuccess;
    ::softmax<<<blocksFor(rows), kBlockSize, 0, s>>>(x, n, stride, rows);
    return cudaGetLastError();
}

cudaError_t kvAppend(const float* k, const float* v, float* kt, float* vc, const int* dims, int n, cudaStream_t s) {
    if (n <= 0) return cudaSuccess;
    kv_append<<<blocksFor(n), kBlockSize, 0, s>>>(k, v, kt, vc, dims);
    return cudaGetLastError();
}

cudaError_t attnValues(const float* p, const float* v, float* out, const int* dims, int total, cudaStream_t s) {
    if (total <= 0) return cudaSuccess;
    attn_values<<<blocksFor(total), kBlockSize, 0, s>>>(p, v, out, dims, 0);
    return cudaGetLastError();
}

cudaError_t quantizeCols(const float* x, int* q, const int* dims, int nb, int ncols, cudaStream_t s) {
    if (nb <= 0 || ncols <= 0) return cudaSuccess;
    quantize_cols<<<blocksFor(nb * ncols), kBlockSize, 0, s>>>(x, q, dims, 0);
    return cudaGetLastError();
}

cudaError_t matvecKQ8(bool q6, const int* qw, const int* xq, const float* xd, float* y, const int* dims, int rows,
                      cudaStream_t s) {
    if (rows <= 0) return cudaSuccess;
    if (q6) matvec_q6k_q8<<<blocksFor(rows), kBlockSize, 0, s>>>(qw, xq, xd, y, dims);
    else matvec_q4k_q8<<<blocksFor(rows), kBlockSize, 0, s>>>(qw, xq, xd, y, dims);
    return cudaGetLastError();
}

cudaError_t quantizeColsK(const float* x, int* q, const int* dims, int nsb, int ncols, cudaStream_t s) {
    if (nsb <= 0 || ncols <= 0) return cudaSuccess;
    quantize_cols_k<<<blocksFor(nsb * ncols), kBlockSize, 0, s>>>(x, q, dims, 0);
    return cudaGetLastError();
}

cudaError_t matmulKQ8K(bool q6, int tile, const int* qw, const int* xq, float* y, const int* dims, int rows, int ncols,
                       cudaStream_t s) {
    if (rows <= 0 || ncols <= 0) return cudaSuccess;
    if (tile != 4 && tile != 8) return cudaErrorInvalidValue;
    const unsigned b = blocksFor(rows * ((ncols + tile - 1) / tile));
    if (tile == 8) {
        if (q6) matmul_q6k_q8kx8<<<b, kBlockSize, 0, s>>>(qw, xq, y, dims, 0);
        else matmul_q4k_q8kx8<<<b, kBlockSize, 0, s>>>(qw, xq, y, dims, 0);
    } else {
        if (q6) matmul_q6k_q8kx4<<<b, kBlockSize, 0, s>>>(qw, xq, y, dims, 0);
        else matmul_q4k_q8kx4<<<b, kBlockSize, 0, s>>>(qw, xq, y, dims, 0);
    }
    return cudaGetLastError();
}

cudaError_t matmulKQ8(bool q6, int tile, const int* qw, const int* xq, float* y, const int* dims, int rows, int ncols,
                      cudaStream_t s) {
    if (rows <= 0 || ncols <= 0) return cudaSuccess;
    if (tile != 4 && tile != 8) return cudaErrorInvalidValue;
    const unsigned b = blocksFor(rows * ((ncols + tile - 1) / tile));
    if (tile == 8) {
        if (q6) matmul_q6k_q8x8<<<b, kBlockSize, 0, s>>>(qw, xq, y, dims, 0);
        else matmul_q4k_q8x8<<<b, kBlockSize, 0, s>>>(qw, xq, y, dims, 0);
    } else {
        if (q6) matmul_q6k_q8x4<<<b, kBlockSize, 0, s>>>(qw, xq, y, dims, 0);
        else matmul_q4k_q8x4<<<b, kBlockSize, 0, s>>>(qw, xq, y, dims, 0);
    }
    return cudaGetLastError();
}

cudaError_t siluMul(const float* g, const float* u, float* out, int count, cudaStream_t s) {
    if (count <= 0) return cudaSuccess;
    silu_mul<<<blocksFor(count), kBlockSize, 0, s>>>(g, u, out, count);
    return cudaGetLastError();
}

} // namespace gpu::llm
