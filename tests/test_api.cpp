// Lifecycle, device queries and diagnostics.

#include "harness.hpp"

#include <cstring>
#include <string>

using kc::Error;

TEST_CASE(calls_before_init_return_not_initialized) {
    void* p = nullptr;
    CHECK_ERR(kcMalloc(&p, 16), Error::NotInitialized);
    int n = 0;
    CHECK_ERR(kcGetDeviceCount(&n), Error::NotInitialized);
    CHECK_ERR(kcDeviceSynchronize(), Error::NotInitialized);
    CHECK_ERR(kcDeviceReset(), Error::NotInitialized);
    CHECK(std::strstr(kcGetLastErrorDetail(), "kcInit") != nullptr);
    CHECK_OK(kcShutdown());  // idempotent even when never initialized
}

TEST_CASE(init_and_shutdown_are_idempotent) {
    CHECK_OK(kcInit());
    CHECK_OK(kcInit());
    CHECK_OK(kcShutdown());
    CHECK_OK(kcShutdown());
    for (int i = 0; i < 20; ++i) {  // repeated full cycles: threads and handlers must not leak
        CHECK_OK(kcInit());
        CHECK_OK(kcShutdown());
    }
}

TEST_CASE(device_properties_identify_cpu_backend) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    int n = -1;
    CHECK_OK(kcGetDeviceCount(&n));
    CHECK(n == 1);
    kc::DeviceProperties p{};
    CHECK_OK(kcGetDeviceProperties(&p, 0));
    CHECK(!p.is_hardware_gpu);
    CHECK(std::strstr(p.name, "CPU") != nullptr);
    CHECK(p.worker_count >= 1);
    CHECK(p.performance_workers >= 1 && p.performance_workers <= p.worker_count + 1);
    CHECK(p.total_memory > 0);
    CHECK(p.kernel_stack_bytes >= 1024 * 1024);
    CHECK_ERR(kcGetDeviceProperties(&p, 1), Error::InvalidDevice);
    CHECK_ERR(kcGetDeviceProperties(nullptr, 0), Error::InvalidValue);
    CHECK_OK(kcSetDevice(0));
    CHECK_ERR(kcSetDevice(1), Error::InvalidDevice);
    CHECK_ERR(kcSetDevice(-1), Error::InvalidDevice);
    int d = -1;
    CHECK_OK(kcGetDevice(&d));
    CHECK(d == 0);
}

TEST_CASE(error_strings_cover_every_code) {
    for (int i = 0; i <= static_cast<int>(Error::InternalError); ++i) {
        const char* s = kcGetErrorString(static_cast<Error>(i));
        CHECK(s && std::strcmp(s, "UnknownError") != 0);
    }
    CHECK(std::strcmp(kcGetErrorString(static_cast<Error>(999)), "UnknownError") == 0);
}

TEST_CASE(unknown_kernel_and_bad_registration) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    int dummy = 0;
    CHECK_ERR(kcLaunchKernel("no_such_kernel", &dummy), Error::KernelNotFound);
    CHECK_ERR(kcLaunchKernel(nullptr, &dummy), Error::InvalidValue);
    CHECK_ERR(kcRegisterKernel(nullptr), Error::InvalidValue);
    kc::KernelEntry incomplete{};
    incomplete.name = "x";
    CHECK_ERR(kcRegisterKernel(&incomplete), Error::InvalidValue);
}

TEST_CASE(unsupported_operations_are_explicit) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    CHECK_ERR(kcStreamWaitEvent(nullptr, nullptr), Error::UnsupportedOperation);
    CHECK(std::strlen(kcGetLastErrorDetail()) > 0);
}

TEST_CASE(no_fault_recorded_initially) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    kc::FaultInfo f{};
    CHECK_ERR(kcGetLastFault(&f), Error::NotReady);
    CHECK_ERR(kcGetLastFault(nullptr), Error::InvalidValue);
}
