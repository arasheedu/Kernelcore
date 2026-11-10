// Fault containment: a misbehaving kernel returns KernelFault, poisons the context,
// and kcDeviceReset recovers. The process must survive every case.

#include "harness.hpp"
#include "test_kernels.hpp"

#include <windows.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using kc::CopyKind;
using kc::Error;

TEST_CASE(out_of_bounds_write_is_contained_1000_times) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    kc::Codegen used{};
    REQUIRE_OK(kcGetKernelCodegenFor("oob_write", &used));
    kc::DeviceProperties props{};
    REQUIRE_OK(kcGetDeviceProperties(&props, 0));
    for (int round = 0; round < 1000; ++round) {
        float* out = nullptr;
        REQUIRE_OK(kcMalloc(reinterpret_cast<void**>(&out), 1024 * sizeof(float)));
        const testkernels::OobWriteArgs args{out, 1024};
        const Error e = kcLaunchKernel("oob_write", &args);
        if (e != Error::KernelFault) {
            th::fail(__FILE__, __LINE__, "round " + std::to_string(round) + ": " + th::describe(e));
            return;
        }
        if (round == 0) {
            const std::string detail = kcGetLastErrorDetail();
            std::printf("  fault detail: %s\n", detail.c_str());
            CHECK(detail.find("oob_write") != std::string::npos);
            CHECK(detail.find("0xC0000005") != std::string::npos);
        }
        kc::FaultInfo f{};
        REQUIRE_OK(kcGetLastFault(&f));
        if (round == 0) {
            CHECK(f.code == EXCEPTION_ACCESS_VIOLATION);
            CHECK(f.access_type == 1);  // write
            CHECK(f.in_kernel_image == (used == kc::Codegen::Booth));  // native paths run outside the image
            CHECK(std::strcmp(f.kernel, "oob_write") == 0);
            if (props.guard_pages)  // first byte past the allocation faults
                CHECK(f.fault_address == reinterpret_cast<uint64_t>(out + 1024));
        }
        void* p = nullptr;
        CHECK_ERR(kcMalloc(&p, 16), Error::ContextPoisoned);
        CHECK_ERR(kcLaunchKernel("oob_write", &args), Error::ContextPoisoned);
        CHECK_ERR(kcDeviceSynchronize(), Error::ContextPoisoned);
        REQUIRE_OK(kcDeviceReset());  // frees `out` too
        REQUIRE_OK(kcMalloc(&p, 16));
        REQUIRE_OK(kcFree(p));
    }
}

TEST_CASE(free_is_still_allowed_after_a_fault) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    float* out = nullptr;
    REQUIRE_OK(kcMalloc(reinterpret_cast<void**>(&out), 1024 * sizeof(float)));
    const testkernels::OobWriteArgs args{out, 1024};
    CHECK_ERR(kcLaunchKernel("oob_write", &args), Error::KernelFault);
    CHECK_OK(kcFree(out));
    CHECK_OK(kcDeviceReset());
}

TEST_CASE(integer_divide_by_zero_is_contained) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    int *a = nullptr, *out = nullptr;
    REQUIRE_OK(kcMalloc(reinterpret_cast<void**>(&a), 64 * sizeof(int)));
    REQUIRE_OK(kcMalloc(reinterpret_cast<void**>(&out), 64 * sizeof(int)));
    const testkernels::IntDivArgs args{a, out, 0, 64};
    CHECK_ERR(kcLaunchKernel("int_div", &args), Error::KernelFault);
    kc::FaultInfo f{};
    CHECK_OK(kcGetLastFault(&f));
    CHECK(f.code == EXCEPTION_INT_DIVIDE_BY_ZERO);
    kc::Codegen used{};
    CHECK_OK(kcGetKernelCodegenFor("int_div", &used));
    CHECK(f.in_kernel_image == (used == kc::Codegen::Booth));
    CHECK_OK(kcDeviceReset());
}

TEST_CASE(stack_overflow_in_kernel_is_contained_and_rearmed) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    for (int round = 0; round < 3; ++round) {
        int* out = nullptr;
        REQUIRE_OK(kcMalloc(reinterpret_cast<void**>(&out), sizeof(int)));
        const testkernels::DeepRecursionArgs args{out, 100000000, 1};
        CHECK_ERR(kcLaunchKernel("deep_recursion", &args), Error::KernelFault);
        kc::FaultInfo f{};
        CHECK_OK(kcGetLastFault(&f));
        CHECK(f.code == EXCEPTION_STACK_OVERFLOW);
        REQUIRE_OK(kcDeviceReset());
    }
    int* out = nullptr;
    REQUIRE_OK(kcMalloc(reinterpret_cast<void**>(&out), sizeof(int)));
    const testkernels::DeepRecursionArgs ok_args{out, 500, 1};
    CHECK_OK(kcLaunchKernel("deep_recursion", &ok_args));
    int v = 0;
    CHECK_OK(kcMemcpy(&v, out, sizeof v, CopyKind::DeviceToHost));
    CHECK(v == 500);
}

TEST_CASE(async_fault_surfaces_at_stream_synchronize) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    kcStream_t s = nullptr;
    REQUIRE_OK(kcStreamCreate(&s));
    float* out = nullptr;
    REQUIRE_OK(kcMalloc(reinterpret_cast<void**>(&out), 1024 * sizeof(float)));
    const testkernels::OobWriteArgs bad{out, 1024};
    CHECK_OK(kcLaunchKernel("oob_write", &bad, s));   // queued: success at submission
    CHECK_OK(kcLaunchKernel("oob_write", &bad, s));   // skipped after the first fault
    CHECK_ERR(kcStreamSynchronize(s), Error::KernelFault);
    CHECK(std::strstr(kcGetLastErrorDetail(), "oob_write") != nullptr);
    CHECK_ERR(kcStreamSynchronize(s), Error::ContextPoisoned);
    CHECK_OK(kcStreamDestroy(s));
    CHECK_OK(kcDeviceReset());
}

TEST_CASE(chunked_fault_is_contained) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    int *a = nullptr, *out = nullptr;
    REQUIRE_OK(kcMalloc(reinterpret_cast<void**>(&a), 64 * sizeof(int)));
    REQUIRE_OK(kcMalloc(reinterpret_cast<void**>(&out), 64 * sizeof(int)));
    // int_div is not chunkable: explicit rejection, nothing runs
    const testkernels::IntDivArgs args{a, out, 0, 64};
    CHECK_ERR(kcLaunchKernelChunked("int_div", &args, 4), Error::UnsupportedOperation);
    void* p = nullptr;
    CHECK_OK(kcMalloc(&p, 4));
}
