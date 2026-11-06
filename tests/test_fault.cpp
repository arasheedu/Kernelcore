// Fault containment: a misbehaving kernel returns KernelFault, poisons the context,
// and pcudaDeviceReset recovers. The process must survive every case.

#include "harness.hpp"
#include "test_kernels.hpp"

#include <windows.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using pcuda::CopyKind;
using pcuda::Error;

TEST_CASE(out_of_bounds_write_is_contained_1000_times) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    pcuda::Codegen used{};
    REQUIRE_OK(pcudaGetKernelCodegenFor("oob_write", &used));
    pcuda::DeviceProperties props{};
    REQUIRE_OK(pcudaGetDeviceProperties(&props, 0));
    for (int round = 0; round < 1000; ++round) {
        float* out = nullptr;
        REQUIRE_OK(pcudaMalloc(reinterpret_cast<void**>(&out), 1024 * sizeof(float)));
        const testkernels::OobWriteArgs args{out, 1024};
        const Error e = pcudaLaunchKernel("oob_write", &args);
        if (e != Error::KernelFault) {
            th::fail(__FILE__, __LINE__, "round " + std::to_string(round) + ": " + th::describe(e));
            return;
        }
        if (round == 0) {
            const std::string detail = pcudaGetLastErrorDetail();
            std::printf("  fault detail: %s\n", detail.c_str());
            CHECK(detail.find("oob_write") != std::string::npos);
            CHECK(detail.find("0xC0000005") != std::string::npos);
        }
        pcuda::FaultInfo f{};
        REQUIRE_OK(pcudaGetLastFault(&f));
        if (round == 0) {
            CHECK(f.code == EXCEPTION_ACCESS_VIOLATION);
            CHECK(f.access_type == 1);  // write
            CHECK(f.in_kernel_image == (used == pcuda::Codegen::Booth));  // native paths run outside the image
            CHECK(std::strcmp(f.kernel, "oob_write") == 0);
            if (props.guard_pages)  // first byte past the allocation faults
                CHECK(f.fault_address == reinterpret_cast<uint64_t>(out + 1024));
        }
        void* p = nullptr;
        CHECK_ERR(pcudaMalloc(&p, 16), Error::ContextPoisoned);
        CHECK_ERR(pcudaLaunchKernel("oob_write", &args), Error::ContextPoisoned);
        CHECK_ERR(pcudaDeviceSynchronize(), Error::ContextPoisoned);
        REQUIRE_OK(pcudaDeviceReset());  // frees `out` too
        REQUIRE_OK(pcudaMalloc(&p, 16));
        REQUIRE_OK(pcudaFree(p));
    }
}

TEST_CASE(free_is_still_allowed_after_a_fault) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    float* out = nullptr;
    REQUIRE_OK(pcudaMalloc(reinterpret_cast<void**>(&out), 1024 * sizeof(float)));
    const testkernels::OobWriteArgs args{out, 1024};
    CHECK_ERR(pcudaLaunchKernel("oob_write", &args), Error::KernelFault);
    CHECK_OK(pcudaFree(out));
    CHECK_OK(pcudaDeviceReset());
}

TEST_CASE(integer_divide_by_zero_is_contained) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    int *a = nullptr, *out = nullptr;
    REQUIRE_OK(pcudaMalloc(reinterpret_cast<void**>(&a), 64 * sizeof(int)));
    REQUIRE_OK(pcudaMalloc(reinterpret_cast<void**>(&out), 64 * sizeof(int)));
    const testkernels::IntDivArgs args{a, out, 0, 64};
    CHECK_ERR(pcudaLaunchKernel("int_div", &args), Error::KernelFault);
    pcuda::FaultInfo f{};
    CHECK_OK(pcudaGetLastFault(&f));
    CHECK(f.code == EXCEPTION_INT_DIVIDE_BY_ZERO);
    pcuda::Codegen used{};
    CHECK_OK(pcudaGetKernelCodegenFor("int_div", &used));
    CHECK(f.in_kernel_image == (used == pcuda::Codegen::Booth));
    CHECK_OK(pcudaDeviceReset());
}

TEST_CASE(stack_overflow_in_kernel_is_contained_and_rearmed) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    for (int round = 0; round < 3; ++round) {
        int* out = nullptr;
        REQUIRE_OK(pcudaMalloc(reinterpret_cast<void**>(&out), sizeof(int)));
        const testkernels::DeepRecursionArgs args{out, 100000000, 1};
        CHECK_ERR(pcudaLaunchKernel("deep_recursion", &args), Error::KernelFault);
        pcuda::FaultInfo f{};
        CHECK_OK(pcudaGetLastFault(&f));
        CHECK(f.code == EXCEPTION_STACK_OVERFLOW);
        REQUIRE_OK(pcudaDeviceReset());
    }
    int* out = nullptr;
    REQUIRE_OK(pcudaMalloc(reinterpret_cast<void**>(&out), sizeof(int)));
    const testkernels::DeepRecursionArgs ok_args{out, 500, 1};
    CHECK_OK(pcudaLaunchKernel("deep_recursion", &ok_args));
    int v = 0;
    CHECK_OK(pcudaMemcpy(&v, out, sizeof v, CopyKind::DeviceToHost));
    CHECK(v == 500);
}

TEST_CASE(async_fault_surfaces_at_stream_synchronize) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    pcudaStream_t s = nullptr;
    REQUIRE_OK(pcudaStreamCreate(&s));
    float* out = nullptr;
    REQUIRE_OK(pcudaMalloc(reinterpret_cast<void**>(&out), 1024 * sizeof(float)));
    const testkernels::OobWriteArgs bad{out, 1024};
    CHECK_OK(pcudaLaunchKernel("oob_write", &bad, s));   // queued: success at submission
    CHECK_OK(pcudaLaunchKernel("oob_write", &bad, s));   // skipped after the first fault
    CHECK_ERR(pcudaStreamSynchronize(s), Error::KernelFault);
    CHECK(std::strstr(pcudaGetLastErrorDetail(), "oob_write") != nullptr);
    CHECK_ERR(pcudaStreamSynchronize(s), Error::ContextPoisoned);
    CHECK_OK(pcudaStreamDestroy(s));
    CHECK_OK(pcudaDeviceReset());
}

TEST_CASE(chunked_fault_is_contained) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    int *a = nullptr, *out = nullptr;
    REQUIRE_OK(pcudaMalloc(reinterpret_cast<void**>(&a), 64 * sizeof(int)));
    REQUIRE_OK(pcudaMalloc(reinterpret_cast<void**>(&out), 64 * sizeof(int)));
    // int_div is not chunkable: explicit rejection, nothing runs
    const testkernels::IntDivArgs args{a, out, 0, 64};
    CHECK_ERR(pcudaLaunchKernelChunked("int_div", &args, 4), Error::UnsupportedOperation);
    void* p = nullptr;
    CHECK_OK(pcudaMalloc(&p, 4));
}
