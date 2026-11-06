#pragma once

#include <cstddef>
#include <cstdint>

namespace pcuda {

enum class CopyKind : int { HostToDevice = 0, DeviceToHost = 1, DeviceToDevice = 2, HostToHost = 3 };

struct DeviceProperties {
    char     name[128] = {};
    unsigned worker_count = 0;     // logical processors available to the runtime
    size_t   total_memory = 0;     // physical host memory in bytes
    bool     is_hardware_gpu = false;
    bool     guard_pages = false;  // allocations end against a no-access page
    size_t   kernel_stack_bytes = 0;
};

} // namespace pcuda

// Opaque handles (CUDA-style naming, project-prefixed).
struct pcudaStream_st;
struct pcudaEvent_st;
using pcudaStream_t = pcudaStream_st*;   // nullptr = default stream (synchronous launches)
using pcudaEvent_t  = pcudaEvent_st*;
