#pragma once

#include "ggml.h"
#include "ggml-backend.h"

// Kernelcore backend: matrix multiplications with GGUF K-quant weights (Q4_K, Q6_K) run as CUDA C kernels
// on the CPU cores, through the Kernelcore runtime (https://github.com/arasheedu/Kernelcore):
// Booth-compiled kernels, translated to AVX2 / AVX-VNNI. Everything else stays on the CPU backend.
// Windows x64 only.

#ifdef  __cplusplus
extern "C" {
#endif

GGML_BACKEND_API ggml_backend_t     ggml_backend_kernelcore_init(void);
GGML_BACKEND_API bool               ggml_backend_is_kc(ggml_backend_t backend);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_kernelcore_reg(void);

#ifdef  __cplusplus
}
#endif
