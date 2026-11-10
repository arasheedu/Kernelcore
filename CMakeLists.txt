cmake_minimum_required(VERSION 3.26)
project(kernelcore VERSION 0.2.0 LANGUAGES C CXX ASM_MASM)

# ---- platform gate -----------------------------------------------------------
if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "Kernelcore targets Windows x64 only (Booth --cpu emits x86-64 System V code)")
endif()
if(NOT MSVC)
  message(FATAL_ERROR "Kernelcore requires MSVC or clang-cl (MSVC ABI, ml64 for the ABI thunk)")
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|arm64")
  message(FATAL_ERROR "ARM64 Windows is not supported")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

option(KC_BUILD_TESTS   "Build tests"                                   ON)
option(KC_ENABLE_ASAN   "Build with AddressSanitizer (/fsanitize=address)" OFF)
option(KC_ENABLE_NVIDIA "Build kc_vs_cuda: the same kernels on Kernelcore and on an NVIDIA GPU (needs CUDA Toolkit)" OFF)

# ---- compiler settings -------------------------------------------------------
add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00 UNICODE _UNICODE)
add_compile_options(
  "$<$<COMPILE_LANGUAGE:CXX>:/W4;/WX;/permissive-;/EHsc;/utf-8;/Zc:__cplusplus;/Zi>"
  "$<$<COMPILE_LANGUAGE:ASM_MASM>:/W3;/WX;/Zi>")
# /CETCOMPAT:NO: the fault guard resumes execution by rewriting RSP/RIP, which
# hardware shadow stacks would reject.
add_link_options(/DEBUG /CETCOMPAT:NO)
if(KC_ENABLE_ASAN)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/fsanitize=address>")
  add_compile_definitions(KC_ASAN=1)
  string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
else()
  # Control Flow Guard: safe because kernel code is only reached via the MASM thunk.
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/guard:cf>")
  add_link_options(/guard:cf)
endif()

# ---- build-time object checker (same parser as the runtime) -------------------
add_executable(kc_objcheck tools/objcheck.cpp src/cpu/booth_elf.cpp)
# Build-time translator for the native code paths (Booth --metal/--ir -> C++).
add_executable(kc_mslgen tools/mslgen.cpp)

include(cmake/BoothKernel.cmake)

# ---- runtime library ---------------------------------------------------------
add_library(kernelcore STATIC
  src/runtime.cpp
  src/allocation_registry.cpp
  src/stream.cpp
  src/cpu/booth_elf.cpp
  src/cpu/booth_image.cpp
  src/cpu/cpu_backend.cpp
  src/cpu/fault_guard.cpp
  src/cpu/guarded_alloc.cpp
  src/cpu/sysv_shims_crt.cpp
  src/cpu/sysv_invoke.asm
  src/cpu/sysv_shims.asm
  adapters/vector_add_adapter.cpp
  adapters/builtin_kernels.cpp)
target_include_directories(kernelcore PUBLIC include PRIVATE src)
target_link_libraries(kernelcore PUBLIC bcrypt synchronization)
booth_add_kernel(TARGET kernelcore NAME vector_add SOURCE kernels/vector_add.cu)

# ---- LLM decode kernels: synthetic Llama decode step (tests, kc_llm_bench) --
add_library(kc_llm STATIC llm/llm_kernels.cpp llm/llm_model.cpp llm/llm_reference.cpp llm/gguf.cpp llm/gguf_model.cpp)
target_include_directories(kc_llm PUBLIC llm)
target_link_libraries(kc_llm PUBLIC kernelcore)
foreach(k matvec_q8 matvec_q4 matvec_q4x8 quantize_q8x4 quantize_cols quantize_cols_k matmul_q4k_q8kx4 matmul_q6k_q8kx4 matmul_q4k_q8kx8 matmul_q6k_q8kx8 matmul_q4k_q8x4 matmul_q6k_q8x4 matmul_q4k_q8x8 matmul_q6k_q8x8 matvec_q4x8_q8 matvec_q4x8_q8x2 matvec_q4k_q8 matvec_q6k_q8 rope_ff mul_rows rope_ggml add_sumsq_f sumsq_part norm_mul_q8_f silu_mul_q8x4f norm_mul_q8k_f silu_mul_q8kf add_rms_norm silu_mul_q8x4 add_sumsq norm_q8x4 rms_norm rope kv_append attn_scores softmax attn_values silu_mul matmul_f16 set_rows_f16 fa_scores fa_max fa_exp fa_values)
  booth_add_kernel(TARGET kc_llm NAME ${k} SOURCE kernels/llm/${k}.cu)
endforeach()

# ---- applications ------------------------------------------------------------
add_executable(kc_llm_real app/llm_real.cpp)
target_link_libraries(kc_llm_real PRIVATE kernelcore kc_llm)

add_executable(kc_poc app/main.cpp)
target_link_libraries(kc_poc PRIVATE kernelcore)

add_executable(kc_bench app/bench.cpp)
target_link_libraries(kc_bench PRIVATE kernelcore)
target_include_directories(kc_bench PRIVATE src)

# ---- tests -------------------------------------------------------------------
if(KC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---- optional: side-by-side comparison with real CUDA ------------------------
if(KC_ENABLE_NVIDIA)
  include(cmake/NvidiaCompare.cmake)
endif()

kc_add_refresh_prebuilt_target()
