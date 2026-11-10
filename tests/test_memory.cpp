// Allocation, registry and copy semantics.

#include "harness.hpp"

#include <windows.h>
#include <psapi.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

using kc::CopyKind;
using kc::Error;

namespace {
bool writeFaults(volatile unsigned char* p) {
    __try {
        *p = 1;
        return false;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER
                                                                   : EXCEPTION_CONTINUE_SEARCH) {
        return true;
    }
}
} // namespace

TEST_CASE(malloc_argument_validation) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    void* p = reinterpret_cast<void*>(1);
    CHECK_ERR(kcMalloc(&p, 0), Error::InvalidValue);
    CHECK(p == nullptr);
    CHECK_ERR(kcMalloc(nullptr, 16), Error::InvalidValue);
    CHECK_ERR(kcMalloc(&p, std::numeric_limits<size_t>::max()), Error::OutOfMemory);
    CHECK_ERR(kcMalloc(&p, std::numeric_limits<size_t>::max() / 2), Error::OutOfMemory);
}

TEST_CASE(malloc_free_and_alignment) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    for (size_t bytes : {size_t(1), size_t(7), size_t(64), size_t(4095), size_t(4096), size_t(1) << 20}) {
        void* p = nullptr;
        CHECK_OK(kcMalloc(&p, bytes));
        REQUIRE(p != nullptr);
        CHECK(reinterpret_cast<uintptr_t>(p) % 64 == 0);
        CHECK_OK(kcFree(p));
    }
}

TEST_CASE(free_rejects_bad_pointers) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    CHECK_OK(kcFree(nullptr));
    void* p = nullptr;
    REQUIRE_OK(kcMalloc(&p, 256));
    CHECK_ERR(kcFree(static_cast<char*>(p) + 8), Error::InvalidPointer);  // interior
    CHECK(std::strstr(kcGetLastErrorDetail(), "interior") != nullptr);
    int stack_var = 0;
    CHECK_ERR(kcFree(&stack_var), Error::InvalidPointer);  // never allocated
    CHECK_OK(kcFree(p));
    CHECK_ERR(kcFree(p), Error::InvalidPointer);  // double free
}

TEST_CASE(memcpy_roundtrip_and_interior_ranges) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    std::vector<int> host(1000), back(1000, -1);
    std::iota(host.begin(), host.end(), 0);
    void* d = nullptr;
    REQUIRE_OK(kcMalloc(&d, host.size() * sizeof(int)));
    CHECK_OK(kcMemcpy(d, host.data(), host.size() * sizeof(int), CopyKind::HostToDevice));
    CHECK_OK(kcMemcpy(back.data(), d, back.size() * sizeof(int), CopyKind::DeviceToHost));
    CHECK(back == host);
    // interior pointer, exact fit to the end
    int tail[10] = {};
    CHECK_OK(kcMemcpy(tail, static_cast<int*>(d) + 990, sizeof tail, CopyKind::DeviceToHost));
    CHECK(tail[0] == 990 && tail[9] == 999);
    // one byte too many
    CHECK_ERR(kcMemcpy(tail, static_cast<int*>(d) + 990, sizeof tail + 1, CopyKind::DeviceToHost),
              Error::InvalidPointer);
    CHECK_ERR(kcMemcpy(d, host.data(), host.size() * sizeof(int) + 4, CopyKind::HostToDevice),
              Error::InvalidPointer);
    CHECK_OK(kcFree(d));
}

TEST_CASE(memcpy_direction_is_enforced) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    std::vector<char> h1(64, 'a'), h2(64, 'b');
    void* d = nullptr;
    REQUIRE_OK(kcMalloc(&d, 64));
    CHECK_ERR(kcMemcpy(h1.data(), h2.data(), 64, CopyKind::HostToDevice), Error::InvalidPointer);
    CHECK_ERR(kcMemcpy(h1.data(), h2.data(), 64, CopyKind::DeviceToHost), Error::InvalidPointer);
    CHECK_ERR(kcMemcpy(d, h2.data(), 64, CopyKind::DeviceToDevice), Error::InvalidPointer);
    CHECK_OK(kcMemcpy(h1.data(), h2.data(), 64, CopyKind::HostToHost));
    CHECK(h1 == h2);
    CHECK_ERR(kcMemcpy(d, h1.data(), 64, static_cast<CopyKind>(42)), Error::InvalidValue);
    CHECK_OK(kcFree(d));
}

TEST_CASE(memcpy_zero_bytes_and_nulls) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    CHECK_OK(kcMemcpy(nullptr, nullptr, 0, CopyKind::HostToDevice));
    char buf[4] = {};
    CHECK_ERR(kcMemcpy(nullptr, buf, 4, CopyKind::HostToHost), Error::InvalidValue);
    CHECK_ERR(kcMemcpy(buf, nullptr, 4, CopyKind::HostToHost), Error::InvalidValue);
}

TEST_CASE(device_to_device_overlap_has_memmove_semantics) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    std::vector<int> host(100);
    std::iota(host.begin(), host.end(), 0);
    int* d = nullptr;
    REQUIRE_OK(kcMalloc(reinterpret_cast<void**>(&d), 100 * sizeof(int)));
    CHECK_OK(kcMemcpy(d, host.data(), 100 * sizeof(int), CopyKind::HostToDevice));
    CHECK_OK(kcMemcpy(d + 10, d, 50 * sizeof(int), CopyKind::DeviceToDevice));
    std::vector<int> back(100);
    CHECK_OK(kcMemcpy(back.data(), d, 100 * sizeof(int), CopyKind::DeviceToHost));
    for (int i = 0; i < 50; ++i) CHECK(back[10 + i] == i);
    CHECK_OK(kcFree(d));
}

TEST_CASE(memcpy_from_unreadable_host_pointer_is_an_error) {
#if defined(KC_ASAN)
    std::printf("  skipped under ASan (ASan reports the wild read before the SEH guard)\n");
#else
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    void* d = nullptr;
    REQUIRE_OK(kcMalloc(&d, 64));
    CHECK_ERR(kcMemcpy(d, reinterpret_cast<const void*>(uintptr_t{0x10}), 64, CopyKind::HostToDevice),
              Error::InvalidValue);
    CHECK_OK(kcFree(d));
#endif
}

TEST_CASE(memset_validates_and_fills) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    unsigned char* d = nullptr;
    REQUIRE_OK(kcMalloc(reinterpret_cast<void**>(&d), 128));
    CHECK_OK(kcMemset(d, 0xAB, 128));
    unsigned char back[128];
    CHECK_OK(kcMemcpy(back, d, 128, CopyKind::DeviceToHost));
    bool all = true;
    for (unsigned char c : back) all = all && c == 0xAB;
    CHECK(all);
    CHECK_ERR(kcMemset(d, 0, 129), Error::InvalidPointer);
    CHECK_OK(kcFree(d));
}

TEST_CASE(allocations_end_against_a_guard_page) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    kc::DeviceProperties p{};
    REQUIRE_OK(kcGetDeviceProperties(&p, 0));
    if (!p.guard_pages) {
        std::printf("  skipped: KC_GUARD_PAGES=0\n");
        return;
    }
    unsigned char* d = nullptr;
    REQUIRE_OK(kcMalloc(reinterpret_cast<void**>(&d), 4096));  // multiple of 64: no slack
    CHECK(!writeFaults(d + 4095));
    CHECK(writeFaults(d + 4096));
    CHECK_OK(kcFree(d));
}

TEST_CASE(shutdown_frees_leaked_allocations) {
    REQUIRE_OK(kcInit());
    void* p = nullptr;
    REQUIRE_OK(kcMalloc(&p, 1024));
    REQUIRE_OK(kcMalloc(&p, 1024));
    CHECK_OK(kcShutdown());
    CHECK(std::strstr(kcGetLastErrorDetail(), "2 allocation") != nullptr);
}

// Opt-in large pages: without the "Lock pages in memory" right, allocations fall back to normal
// pages; with it, big blocks are large-page memory that still ends against a faulting range.
TEST_CASE(large_pages_back_big_allocations_or_fall_back) {
    _putenv_s("KC_LARGE_PAGES", "1");
    struct Unset {
        ~Unset() { _putenv_s("KC_LARGE_PAGES", ""); }
    } unset;
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    kc::DeviceProperties p{};
    REQUIRE_OK(kcGetDeviceProperties(&p, 0));
    const size_t bytes = 3u << 20;  // 3 MB: two large pages, user block flush with the end
    unsigned char* d = nullptr;
    REQUIRE_OK(kcMalloc(reinterpret_cast<void**>(&d), bytes));
    std::memset(d, 7, bytes);
    CHECK(d[0] == 7 && d[bytes - 1] == 7);
    PSAPI_WORKING_SET_EX_INFORMATION info{};
    info.VirtualAddress = d;
    REQUIRE(QueryWorkingSetEx(GetCurrentProcess(), &info, sizeof info));
    const bool large = info.VirtualAttributes.Valid && info.VirtualAttributes.LargePage;
    std::printf("  large pages %s; this block %s large-page memory\n", p.large_pages ? "enabled" : "unavailable",
                large ? "is" : "is not");
    if (!p.large_pages) {
        CHECK(!large);
    } else {
        CHECK(large);  // memory may be too fragmented on a busy machine, but not right after start
        if (p.guard_pages) CHECK(writeFaults(d + bytes));
    }
    CHECK_OK(kcFree(d));
}
