// Win64 -> System V thunk: argument passing, register preservation, fault recovery.

#include "harness.hpp"

#include "cpu/fault_guard.hpp"

#include <windows.h>

#include <bit>
#include <cstdint>
#include <thread>

using pcuda::ArgPacker;
using pcuda::Error;
using pcuda::LaunchArgs;
using pcuda::cpu::KernelScope;

extern "C" {
uint64_t pcuda_test_check_preserved(const void* entry, const uint64_t* gpr6, const uint64_t* xmm8, void** anchor);
void sysv_echo_clobber();
void sysv_fault_read();
void sysv_div0();
void sysv_recurse();
}

namespace {

struct GuardRef {
    GuardRef() { pcuda::cpu::acquireFaultGuard(); }
    ~GuardRef() { pcuda::cpu::releaseFaultGuard(); }
};

constexpr const char* kRegNames[] = {"rbx", "rbp", "rdi", "rsi", "r12", "r13", "r14", "r15",
                                     "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12",
                                     "xmm13", "xmm14", "xmm15", "DF", "MXCSR"};

void reportMask(uint64_t mask) {
    for (int i = 0; i < 20; ++i)
        if (mask & (1ull << i)) th::fail(__FILE__, __LINE__, std::string("not preserved: ") + kRegNames[i]);
}

} // namespace

TEST_CASE(arguments_and_alignment_reach_sysv_code) {
    GuardRef g;
    alignas(16) uint64_t out[16] = {};
    const uint64_t gpr[6] = {reinterpret_cast<uint64_t>(out), 0x1111111111111111ull, 0x2222222222222222ull,
                             0x3333333333333333ull, 0x4444444444444444ull, 0x5555555555555555ull};
    uint64_t xmm[8];
    for (int i = 0; i < 8; ++i) xmm[i] = 0xA0A0A0A0A0A0A000ull + static_cast<uint64_t>(i);
    uint64_t r;
    {
        KernelScope scope("echo", nullptr, 0);
        r = pcuda_test_check_preserved(reinterpret_cast<const void*>(&sysv_echo_clobber), gpr, xmm, scope.anchorSlot());
        CHECK(!scope.faulted());
    }
    CHECK((r & 0xFF) == 0);  // normal return
    reportMask(r >> 8);
    for (int i = 0; i < 6; ++i) CHECK(out[i] == gpr[i]);
    for (int i = 0; i < 8; ++i) CHECK(out[6 + i] == xmm[i]);
    CHECK(out[14] == 0);  // System V stack alignment at entry
}

TEST_CASE(recovery_path_restores_registers_after_access_violation) {
    GuardRef g;
    const uint64_t gpr[6] = {};
    const uint64_t xmm[8] = {};
    for (int round = 0; round < 100; ++round) {
        uint64_t r;
        pcuda::FaultInfo f{};
        {
            KernelScope scope("fault_read", nullptr, 0);
            r = pcuda_test_check_preserved(reinterpret_cast<const void*>(&sysv_fault_read), gpr, xmm,
                                           scope.anchorSlot());
            REQUIRE(scope.faulted());
            f = scope.fault();
        }
        CHECK((r & 0xFF) == 1);  // resumed at pcuda_sysv_recover
        reportMask(r >> 8);
        CHECK(f.code == EXCEPTION_ACCESS_VIOLATION);
        CHECK(f.fault_address == 0);
        CHECK(f.access_type == 0);  // read
    }
}

TEST_CASE(recovery_from_integer_divide_by_zero) {
    GuardRef g;
    const uint64_t gpr[6] = {};
    const uint64_t xmm[8] = {};
    uint64_t r;
    pcuda::FaultInfo f{};
    {
        KernelScope scope("div0", nullptr, 0);
        r = pcuda_test_check_preserved(reinterpret_cast<const void*>(&sysv_div0), gpr, xmm, scope.anchorSlot());
        f = scope.fault();
    }
    CHECK((r & 0xFF) == 1);
    reportMask(r >> 8);
    CHECK(f.code == EXCEPTION_INT_DIVIDE_BY_ZERO);
}

TEST_CASE(stack_overflow_is_recovered_repeatedly_on_one_thread) {
    GuardRef g;
    bool ok = true;
    int recovered = 0;
    // Default std::thread stack (1 MB) keeps this quick; three rounds prove the guard page is re-armed.
    std::thread t([&] {
        LaunchArgs a{};
        for (int i = 0; i < 3; ++i) {
            const auto res = pcuda::cpu::invokeSysV(reinterpret_cast<const void*>(&sysv_recurse), a, "recurse",
                                                    nullptr, 0);
            if (!res.faulted || res.fault.code != EXCEPTION_STACK_OVERFLOW) ok = false;
            else ++recovered;
        }
        alignas(16) uint64_t out[16] = {};
        a.gpr[0] = reinterpret_cast<uint64_t>(out);
        const auto res = pcuda::cpu::invokeSysV(reinterpret_cast<const void*>(&sysv_echo_clobber), a, "echo",
                                                nullptr, 0);
        if (res.faulted || out[0] != a.gpr[0]) ok = false;
    });
    t.join();
    CHECK(ok);
    CHECK(recovered == 3);
}

static bool sehCatchesNullWrite() {
    __try {
        volatile int* p = nullptr;
        *p = 1;
        return false;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return true;  // the guard passed it on (EXCEPTION_CONTINUE_SEARCH)
    }
}

TEST_CASE(faults_outside_kernel_scope_are_not_swallowed) {
    GuardRef g;
    CHECK(sehCatchesNullWrite());
}

TEST_CASE(arg_packer_sign_extends_and_limits_arguments) {
    LaunchArgs a;
    ArgPacker pk(a);
    pk.i32(-5).ptr(reinterpret_cast<void*>(0x1234)).f32(1.5f);
    CHECK_OK(pk.finish(77));
    CHECK(a.n_gpr == 3);
    CHECK(a.gpr[0] == 0xFFFFFFFFFFFFFFFBull);
    CHECK(a.gpr[1] == 0x1234);
    CHECK(a.gpr[2] == 77);  // hidden nthreads, full 64 bits
    CHECK(a.nthreads == 77);
    CHECK(a.n_xmm == 1);
    CHECK(a.xmm[0] == std::bit_cast<uint32_t>(1.5f));

    LaunchArgs b;
    ArgPacker too_many(b);
    too_many.i32(1).i32(2).i32(3).i32(4).i32(5).i32(6);  // 6 declared + nthreads > 6 registers
    CHECK_ERR(too_many.finish(1), Error::UnsupportedOperation);

    LaunchArgs c;
    ArgPacker neg(c);
    CHECK_ERR(neg.finish(-1), Error::InvalidValue);
}
