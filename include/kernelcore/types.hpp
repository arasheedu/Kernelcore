#pragma once

#include <cstddef>
#include <cstdint>

namespace kc {

enum class CopyKind : int { HostToDevice = 0, DeviceToHost = 1, DeviceToDevice = 2, HostToHost = 3 };

// How kernels execute on the CPU (kcSetKernelCodegen, env KC_CODEGEN=auto|booth|native|simd).
//   Booth   Booth's own x86 object (--cpu), loaded and run through the System V thunk.
//   Native  Booth's IR via its --metal output, compiled by MSVC: one CUDA thread at a time.
//   Simd    the same, 8 CUDA threads per AVX2 register (falls back to Native per kernel).
//   Auto    the fastest available: Simd, then Native, then Booth.
// All paths are bit-identical for race-free kernels.
enum class Codegen : int { Auto = 0, Booth = 1, Native = 2, Simd = 3 };

struct DeviceProperties {
    char     name[128] = {};
    unsigned worker_count = 0;     // threads in the chunked-launch pool (the launching thread also runs chunks)
    // Logical processors on the fastest cores, capped at worker_count + 1: the `workers` to pass for
    // launches bound by memory bandwidth (slow cores add no bandwidth, only late chunks).
    unsigned performance_workers = 0;
    size_t   total_memory = 0;     // physical host memory in bytes
    bool     is_hardware_gpu = false;
    bool     guard_pages = false;  // allocations end against a no-access page
    bool     large_pages = false;  // allocations of >= 2 MB use large pages (KC_LARGE_PAGES=1 + privilege)
    size_t   kernel_stack_bytes = 0;
};

} // namespace kc

// Opaque handles (CUDA-style naming, project-prefixed).
struct kcStream_st;
struct kcEvent_st;
struct kcGraph_st;
using kcStream_t = kcStream_st*;   // nullptr = default stream (synchronous launches)
using kcEvent_t  = kcEvent_st*;
using kcGraph_t  = kcGraph_st*;
