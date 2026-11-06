#pragma once

// Host-callable launchers for the nvcc-compiled copies of kernels/llm/*.cu (256-thread blocks).
// Plain C++ interface: all pointers are CUDA device pointers; each call only enqueues.

#include <cuda_runtime_api.h>

#include <cstdint>

namespace gpu::llm {

cudaError_t matvecQ8(const int8_t* q, const float* scales, const float* x, float* y, const int* dims, int rows,
                     cudaStream_t s);
cudaError_t matvecQ4(const uint8_t* qs, const float* scales, const float* x, float* y, const int* dims, int rows,
                     cudaStream_t s);
cudaError_t matvecQ4x8(const uint8_t* qs, const float* scales, const float* x, float* y, const int* dims, int rows,
                       cudaStream_t s);
cudaError_t quantizeQ8x4(const float* x, int* xq, float* xd, int count, cudaStream_t s);
cudaError_t matvecQ4x8Q8(const int* qw, const int* xq, const float* xd, float* y, const int* dims, int rows,
                         cudaStream_t s);
cudaError_t rmsNorm(const float* x, const float* w, float* y, int n, int rows, float eps, cudaStream_t s);
cudaError_t addRmsNorm(float* x, const float* t, const float* w, float* y, int n, float eps, cudaStream_t s);
cudaError_t siluMulQ8x4(const float* g, const float* u, int* xq, float* xd, int count, cudaStream_t s);
cudaError_t rope(float* x, int pairs_per_head, int pos, int count, float log2_base, cudaStream_t s);
cudaError_t attnScores(const float* q, const float* k, float* scores, const int* dims, int total, float scale,
                       cudaStream_t s);
cudaError_t softmax(float* x, int n, int stride, int rows, cudaStream_t s);
cudaError_t kvAppend(const float* k, const float* v, float* kt, float* vc, const int* dims, int n, cudaStream_t s);
cudaError_t attnValues(const float* p, const float* v, float* out, const int* dims, int total, cudaStream_t s);
cudaError_t siluMul(const float* g, const float* u, float* out, int count, cudaStream_t s);

} // namespace gpu::llm
