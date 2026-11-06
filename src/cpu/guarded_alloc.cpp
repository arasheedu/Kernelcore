#include "guarded_alloc.hpp"

#include <windows.h>

#include <malloc.h>
#include <cstring>
#include <limits>

namespace pcuda::cpu {
namespace {

bool roundUp(size_t v, size_t a, size_t& r) noexcept {
    if (v > std::numeric_limits<size_t>::max() - (a - 1)) return false;
    r = (v + a - 1) / a * a;
    return true;
}

size_t pageSize() noexcept {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
}

} // namespace

Error allocateBlock(size_t bytes, bool guarded, HostBlock& out) noexcept {
    out = HostBlock{};
    if (bytes == 0) return Error::InvalidValue;
    size_t padded;
    if (!roundUp(bytes, kAllocAlignment, padded)) return Error::OutOfMemory;

    if (!guarded) {
        void* p = _aligned_malloc(padded, kAllocAlignment);
        if (!p) return Error::OutOfMemory;
        std::memset(p, 0, padded);
        out = {static_cast<std::byte*>(p), bytes, p, padded, false};
        return Error::Success;
    }

    const size_t page = pageSize();
    size_t data;
    if (!roundUp(padded, page, data) || data > std::numeric_limits<size_t>::max() - page) return Error::OutOfMemory;
    const size_t region_bytes = data + page;
    void* region = VirtualAlloc(nullptr, region_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!region) return Error::OutOfMemory;
    DWORD old = 0;
    if (!VirtualProtect(static_cast<std::byte*>(region) + data, page, PAGE_NOACCESS, &old)) {
        VirtualFree(region, 0, MEM_RELEASE);
        return Error::OutOfMemory;
    }
    // VirtualAlloc memory is already zeroed.
    out = {static_cast<std::byte*>(region) + (data - padded), bytes, region, region_bytes, true};
    return Error::Success;
}

void releaseBlock(const HostBlock& b) noexcept {
    if (!b.region) return;
    if (b.guarded) VirtualFree(b.region, 0, MEM_RELEASE);
    else _aligned_free(b.region);
}

} // namespace pcuda::cpu
