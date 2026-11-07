#pragma once

#include "ggml.h"
#include "ggml-backend.h"

// pcuda backend: matrix multiplications with GGUF K-quant weights (Q4_K, Q6_K) run as CUDA C kernels
// on the CPU cores, through the pcuda runtime (https://github.com/arasheedu/CudaOnCPUCores):
// Booth-compiled kernels, translated to AVX2 / AVX-VNNI. Everything else stays on the CPU backend.
// Windows x64 only.

#ifdef  __cplusplus
extern "C" {
#endif

GGML_BACKEND_API ggml_backend_t     ggml_backend_pcuda_init(void);
GGML_BACKEND_API bool               ggml_backend_is_pcuda(ggml_backend_t backend);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_pcuda_reg(void);

#ifdef  __cplusplus
}
#endif
