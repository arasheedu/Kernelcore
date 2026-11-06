#pragma once

// Host memory blocks backing "device" allocations.
// Guarded mode: VirtualAlloc with a trailing PAGE_NOACCESS page; the user block ends
// flush against it (64-byte aligned), so an overrun of more than the alignment slack
// faults at the first bad byte. Unguarded mode: _aligned_malloc(64).

#include <pcuda/error.hpp>

#include <cstddef>

namespace pcuda::cpu {

constexpr size_t kAllocAlignment = 64;

struct HostBlock {
    std::byte* user = nullptr;
    size_t     user_bytes = 0;
    void*      region = nullptr;
    size_t     region_bytes = 0;
    bool       guarded = false;
};

Error allocateBlock(size_t bytes, bool guarded, HostBlock& out) noexcept;
void  releaseBlock(const HostBlock& block) noexcept;

} // namespace pcuda::cpu
