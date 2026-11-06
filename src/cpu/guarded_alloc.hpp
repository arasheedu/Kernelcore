#pragma once

// Host memory blocks backing "device" allocations.
// Guarded mode: VirtualAlloc with a trailing PAGE_NOACCESS page; the user block ends
// flush against it (64-byte aligned), so an overrun of more than the alignment slack
// faults at the first bad byte. Unguarded mode: _aligned_malloc(64).
// Large pages (opt-in, needs the "Lock pages in memory" right): blocks of at least one large
// page (2 MB on x64) are backed by large pages, which cuts TLB misses when kernels stream
// through big buffers. The block still ends flush against a reserved no-access range in
// guarded mode. If large pages can't be had (fragmented memory), normal pages are used.

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
    bool       large = false;    // region is large-page memory (VirtualAlloc MEM_LARGE_PAGES)
    void*      fence = nullptr;  // large + guarded: reserved no-access range after the region
};

// large_pages: try large pages for blocks of at least largePageSize() bytes.
Error allocateBlock(size_t bytes, bool guarded, bool large_pages, HostBlock& out) noexcept;
void  releaseBlock(const HostBlock& block) noexcept;

// Enables SeLockMemoryPrivilege for this process; true when large pages can be used.
bool   enableLargePages() noexcept;
size_t largePageSize() noexcept;  // 0 when unsupported

} // namespace pcuda::cpu
