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

constexpr size_t kFenceBytes = 64 * 1024;  // allocation granularity

// Large-page region of `data` bytes (a multiple of the large page size), plus, when guarded, a
// reserved no-access range right after it. VirtualAlloc can't place both in one call, so find
// free address space first, release it, and claim the two parts at that address (another
// thread may take it in between: retry a few times).
bool allocateLarge(size_t data, bool guarded, void** region, void** fence) noexcept {
    const size_t lp = largePageSize();
    for (int attempt = 0; attempt < 4; ++attempt) {
        void* probe = VirtualAlloc(nullptr, data + lp + kFenceBytes, MEM_RESERVE, PAGE_NOACCESS);
        if (!probe) return false;
        const auto base = (reinterpret_cast<uintptr_t>(probe) + lp - 1) / lp * lp;
        VirtualFree(probe, 0, MEM_RELEASE);
        void* r = VirtualAlloc(reinterpret_cast<void*>(base), data, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                               PAGE_READWRITE);
        if (!r) {
            if (GetLastError() != ERROR_INVALID_ADDRESS) return false;  // no large pages left
            continue;
        }
        if (!guarded) {
            *region = r, *fence = nullptr;
            return true;
        }
        void* f = VirtualAlloc(static_cast<std::byte*>(r) + data, kFenceBytes, MEM_RESERVE, PAGE_NOACCESS);
        if (f) {
            *region = r, *fence = f;
            return true;
        }
        VirtualFree(r, 0, MEM_RELEASE);
    }
    return false;
}

} // namespace

size_t largePageSize() noexcept { return GetLargePageMinimum(); }

bool enableLargePages() noexcept {
    if (largePageSize() == 0) return false;
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token)) return false;
    TOKEN_PRIVILEGES tp{};
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    bool ok = LookupPrivilegeValueW(nullptr, SE_LOCK_MEMORY_NAME, &tp.Privileges[0].Luid) &&
              AdjustTokenPrivileges(token, FALSE, &tp, 0, nullptr, nullptr) &&
              GetLastError() == ERROR_SUCCESS;  // ERROR_NOT_ALL_ASSIGNED: the account lacks the right
    CloseHandle(token);
    return ok;
}

Error allocateBlock(size_t bytes, bool guarded, bool large_pages, HostBlock& out) noexcept {
    out = HostBlock{};
    if (bytes == 0) return Error::InvalidValue;
    size_t padded;
    if (!roundUp(bytes, kAllocAlignment, padded)) return Error::OutOfMemory;

    if (large_pages && largePageSize() && padded >= largePageSize()) {
        size_t data;
        void *region = nullptr, *fence = nullptr;
        if (roundUp(padded, largePageSize(), data) && allocateLarge(data, guarded, &region, &fence)) {
            // Large-page memory is zeroed like any VirtualAlloc memory.
            out = {static_cast<std::byte*>(region) + (data - padded), bytes, region, data, guarded, true, fence};
            return Error::Success;
        }
        // Not enough contiguous physical memory for large pages: use normal pages.
    }

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
    if (b.large) {
        VirtualFree(b.region, 0, MEM_RELEASE);
        if (b.fence) VirtualFree(b.fence, 0, MEM_RELEASE);
    } else if (b.guarded) {
        VirtualFree(b.region, 0, MEM_RELEASE);
    } else {
        _aligned_free(b.region);
    }
}

} // namespace pcuda::cpu
