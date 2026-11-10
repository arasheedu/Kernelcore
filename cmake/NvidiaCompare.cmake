# Opt-in (KC_ENABLE_NVIDIA=ON): compile the same kernel sources with nvcc and build
# kc_vs_cuda, which runs them on Kernelcore (CPU) and on the NVIDIA GPU side by side.
# The CPU-only build never reaches this file, so it keeps no CUDA dependency.

if(NOT KC_BUILD_TESTS)
  message(FATAL_ERROR "KC_ENABLE_NVIDIA needs KC_BUILD_TESTS=ON (it reuses the test kernels)")
endif()
if(KC_ENABLE_ASAN)
  message(FATAL_ERROR "KC_ENABLE_NVIDIA is not supported together with KC_ENABLE_ASAN")
endif()

# Find nvcc before enable_language(CUDA) so its version can decide the host-compiler flags.
if(NOT CMAKE_CUDA_COMPILER)
  find_program(KC_NVCC nvcc HINTS "$ENV{CUDA_PATH}/bin")
  if(NOT KC_NVCC)
    message(FATAL_ERROR "KC_ENABLE_NVIDIA: nvcc not found. Install the CUDA Toolkit or set CUDA_PATH.")
  endif()
  set(CMAKE_CUDA_COMPILER "${KC_NVCC}")
endif()
execute_process(COMMAND "${CMAKE_CUDA_COMPILER}" --version OUTPUT_VARIABLE _nvcc_out ERROR_QUIET)
if(NOT _nvcc_out MATCHES "release ([0-9]+)\\.([0-9]+)")
  message(FATAL_ERROR "KC_ENABLE_NVIDIA: could not read the nvcc version from '${CMAKE_CUDA_COMPILER}'")
endif()
set(KC_NVCC_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")

# CUDA < 12.4 rejects MSVC 19.40+ (VS 2022 17.10 and later) twice: nvcc's own host check and
# the STL's static_assert. Override both, only in that combination. Upgrading CUDA removes this.
if(KC_NVCC_VERSION VERSION_LESS 12.4 AND MSVC_VERSION GREATER_EQUAL 1940)
  message(WARNING "CUDA ${KC_NVCC_VERSION} does not officially support MSVC ${MSVC_VERSION}; "
                  "building with -allow-unsupported-compiler. CUDA 12.4+ avoids this.")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)  # the GPU(s) in this machine
endif()
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
message(STATUS "kc_vs_cuda: nvcc ${KC_NVCC_VERSION}, architectures ${CMAKE_CUDA_ARCHITECTURES}")

# Device code: default nvcc settings (-O3, precise sinf/cosf, FMA contraction on), as a
# typical CUDA build would use. The kernel sources are #included from kernels/.
add_library(kc_gpu_kernels STATIC gpu/cuda_kernels.cu gpu/cuda_llm.cu)
target_include_directories(kc_gpu_kernels PUBLIC gpu PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(kc_gpu_kernels PUBLIC CUDA::cudart_static)
set_property(TARGET kc_gpu_kernels PROPERTY CUDA_RUNTIME_LIBRARY Static)
# The .cu files under kernels/ are #included, never compiled on their own here.
set_property(SOURCE gpu/cuda_kernels.cu APPEND PROPERTY OBJECT_DEPENDS
  "${PROJECT_SOURCE_DIR}/kernels/vector_add.cu"
  "${PROJECT_SOURCE_DIR}/kernels/test/scale_add.cu"
  "${PROJECT_SOURCE_DIR}/kernels/test/trig_mix.cu")
foreach(k matvec_q8 matvec_q4 matvec_q4x8 quantize_q8x4 matvec_q4x8_q8 add_rms_norm silu_mul_q8x4 rms_norm rope attn_scores softmax attn_values silu_mul
        quantize_cols matvec_q4k_q8 matvec_q6k_q8 matmul_q4k_q8x4 matmul_q6k_q8x4 matmul_q4k_q8x8 matmul_q6k_q8x8
        quantize_cols_k matmul_q4k_q8kx4 matmul_q6k_q8kx4 matmul_q4k_q8kx8 matmul_q6k_q8kx8)
  set_property(SOURCE gpu/cuda_llm.cu APPEND PROPERTY OBJECT_DEPENDS "${PROJECT_SOURCE_DIR}/kernels/llm/${k}.cu")
endforeach()

add_executable(kc_vs_cuda app/compare.cpp)
target_link_libraries(kc_vs_cuda PRIVATE kernelcore kc_test_kernels kc_gpu_kernels)
set_property(TARGET kc_vs_cuda PROPERTY LINKER_LANGUAGE CXX)

# Correctness-only run on small sizes. Exit code 77 (no usable GPU) marks the test skipped.
add_test(NAME kc_vs_cuda_check COMMAND kc_vs_cuda --check)
set_tests_properties(kc_vs_cuda_check PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77)

# One LLM decode step (Llama 3.2 shapes) on Kernelcore, on the GPU and in plain C++.
add_executable(kc_llm_bench app/llm_bench.cpp)
target_link_libraries(kc_llm_bench PRIVATE kernelcore kc_llm kc_gpu_kernels)
set_property(TARGET kc_llm_bench PROPERTY LINKER_LANGUAGE CXX)
add_test(NAME kc_llm_bench_check COMMAND kc_llm_bench --check)
set_tests_properties(kc_llm_bench_check PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77)

# The ggml backend's K-quant kernels (quantize_cols, matvec, 4/8-column matmuls) on Kernelcore and on
# the GPU: results compared, one Llama 3.2 3B layer's matmuls timed per batch size.
add_executable(kc_kquant_vs_cuda app/kquant_vs_cuda.cpp)
target_link_libraries(kc_kquant_vs_cuda PRIVATE kernelcore kc_llm kc_gpu_kernels)
set_property(TARGET kc_kquant_vs_cuda PROPERTY LINKER_LANGUAGE CXX)
add_test(NAME kc_kquant_vs_cuda_check COMMAND kc_kquant_vs_cuda --check)
set_tests_properties(kc_kquant_vs_cuda_check PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77)
