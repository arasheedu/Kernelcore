// Lifecycle, device queries and diagnostics.

#include "harness.hpp"

#include <cstring>
#include <string>

using pcuda::Error;

TEST_CASE(calls_before_init_return_not_initialized) {
    void* p = nullptr;
    CHECK_ERR(pcudaMalloc(&p, 16), Error::NotInitialized);
    int n = 0;
    CHECK_ERR(pcudaGetDeviceCount(&n), Error::NotInitialized);
    CHECK_ERR(pcudaDeviceSynchronize(), Error::NotInitialized);
    CHECK_ERR(pcudaDeviceReset(), Error::NotInitialized);
    CHECK(std::strstr(pcudaGetLastErrorDetail(), "pcudaInit") != nullptr);
    CHECK_OK(pcudaShutdown());  // idempotent even when never initialized
}

TEST_CASE(init_and_shutdown_are_idempotent) {
    CHECK_OK(pcudaInit());
    CHECK_OK(pcudaInit());
    CHECK_OK(pcudaShutdown());
    CHECK_OK(pcudaShutdown());
    for (int i = 0; i < 20; ++i) {  // repeated full cycles: threads and handlers must not leak
        CHECK_OK(pcudaInit());
        CHECK_OK(pcudaShutdown());
    }
}

TEST_CASE(device_properties_identify_cpu_backend) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    int n = -1;
    CHECK_OK(pcudaGetDeviceCount(&n));
    CHECK(n == 1);
    pcuda::DeviceProperties p{};
    CHECK_OK(pcudaGetDeviceProperties(&p, 0));
    CHECK(!p.is_hardware_gpu);
    CHECK(std::strstr(p.name, "CPU") != nullptr);
    CHECK(p.worker_count >= 1);
    CHECK(p.total_memory > 0);
    CHECK(p.kernel_stack_bytes >= 1024 * 1024);
    CHECK_ERR(pcudaGetDeviceProperties(&p, 1), Error::InvalidDevice);
    CHECK_ERR(pcudaGetDeviceProperties(nullptr, 0), Error::InvalidValue);
    CHECK_OK(pcudaSetDevice(0));
    CHECK_ERR(pcudaSetDevice(1), Error::InvalidDevice);
    CHECK_ERR(pcudaSetDevice(-1), Error::InvalidDevice);
    int d = -1;
    CHECK_OK(pcudaGetDevice(&d));
    CHECK(d == 0);
}

TEST_CASE(error_strings_cover_every_code) {
    for (int i = 0; i <= static_cast<int>(Error::InternalError); ++i) {
        const char* s = pcudaGetErrorString(static_cast<Error>(i));
        CHECK(s && std::strcmp(s, "UnknownError") != 0);
    }
    CHECK(std::strcmp(pcudaGetErrorString(static_cast<Error>(999)), "UnknownError") == 0);
}

TEST_CASE(unknown_kernel_and_bad_registration) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    int dummy = 0;
    CHECK_ERR(pcudaLaunchKernel("no_such_kernel", &dummy), Error::KernelNotFound);
    CHECK_ERR(pcudaLaunchKernel(nullptr, &dummy), Error::InvalidValue);
    CHECK_ERR(pcudaRegisterKernel(nullptr), Error::InvalidValue);
    pcuda::KernelEntry incomplete{};
    incomplete.name = "x";
    CHECK_ERR(pcudaRegisterKernel(&incomplete), Error::InvalidValue);
}

TEST_CASE(unsupported_operations_are_explicit) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    CHECK_ERR(pcudaStreamWaitEvent(nullptr, nullptr), Error::UnsupportedOperation);
    CHECK(std::strlen(pcudaGetLastErrorDetail()) > 0);
}

TEST_CASE(no_fault_recorded_initially) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    pcuda::FaultInfo f{};
    CHECK_ERR(pcudaGetLastFault(&f), Error::NotReady);
    CHECK_ERR(pcudaGetLastFault(nullptr), Error::InvalidValue);
}
