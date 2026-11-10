// Graphs: captured launches run later, in order, with the same results as direct launches;
// capture rules; failures inside a graph.

#include "harness.hpp"
#include "test_kernels.hpp"

#include <kernelcore/kernels.hpp>

#include <string>
#include <vector>

using kc::CopyKind;
using kc::Error;

namespace {

template <class T>
T* devFill(const std::vector<T>& v) {
    void* p = nullptr;
    if (kcMalloc(&p, v.size() * sizeof(T)) != Error::Success) return nullptr;
    if (kcMemcpy(p, v.data(), v.size() * sizeof(T), CopyKind::HostToDevice) != Error::Success) return nullptr;
    return static_cast<T*>(p);
}
template <class T>
std::vector<T> host(const T* d, size_t n) {
    std::vector<T> v(n);
    kcMemcpy(v.data(), d, n * sizeof(T), CopyKind::DeviceToHost);
    return v;
}

constexpr int kN = 100003;  // not a multiple of the chunk alignment

// out = a + b (chunked), then out2 = out * 2 + 1 (single launch), then out = out2 + b (chunked):
// each step reads the previous step's output, so a missing barrier changes the result.
void enqueueChain(float* a, float* b, float* out, float* out2, kcStream_t s) {
    const kc::kernels::VectorAddArgs add1{a, b, out, kN};
    const testkernels::ScaleAddArgs sc{out, out2, 2.0f, 1.0f, kN};
    const kc::kernels::VectorAddArgs add2{out2, b, out, kN};
    CHECK_OK(kcLaunchKernelChunked("vector_add", &add1, 8, 1024, s));
    CHECK_OK(kcLaunchKernel("scale_add", &sc, s));
    CHECK_OK(kcLaunchKernelChunked("vector_add", &add2, 8, 1024, s));
}

} // namespace

TEST_CASE(graph_runs_captured_launches_in_order_with_direct_results) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    std::vector<float> a(kN), b(kN);
    for (int i = 0; i < kN; ++i) a[i] = float(i % 97) * 0.5f, b[i] = float(i % 13) - 6.0f;
    float *da = devFill(a), *db = devFill(b), *dout = devFill(std::vector<float>(kN, 0.0f)),
          *dout2 = devFill(std::vector<float>(kN, 0.0f));
    REQUIRE(da && db && dout && dout2);

    enqueueChain(da, db, dout, dout2, nullptr);  // direct, synchronous
    const auto want = host(dout, kN);

    REQUIRE_OK(kcMemset(dout, 0, kN * sizeof(float)));
    REQUIRE_OK(kcMemset(dout2, 0, kN * sizeof(float)));
    REQUIRE_OK(kcStreamBeginCapture(nullptr));
    enqueueChain(da, db, dout, dout2, nullptr);
    kcGraph_t g = nullptr;
    REQUIRE_OK(kcStreamEndCapture(nullptr, &g));
    REQUIRE(g != nullptr);
    CHECK(host(dout, kN) == std::vector<float>(kN, 0.0f));  // nothing ran during capture

    CHECK_OK(kcGraphLaunch(g));
    CHECK(host(dout, kN) == want);
    CHECK_OK(kcGraphLaunch(g));  // a graph can be launched again
    CHECK(host(dout, kN) == want);

    // Asynchronous launch on a created stream, destroyed before it has necessarily run.
    REQUIRE_OK(kcMemset(dout, 0, kN * sizeof(float)));
    kcStream_t s = nullptr;
    REQUIRE_OK(kcStreamCreate(&s));
    CHECK_OK(kcGraphLaunch(g, s));
    CHECK_OK(kcGraphDestroy(g));
    CHECK_OK(kcStreamSynchronize(s));
    CHECK(host(dout, kN) == want);
    CHECK_OK(kcStreamDestroy(s));
}

TEST_CASE(capture_on_a_created_stream_leaves_other_streams_running) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    std::vector<float> a(kN, 1.0f), b(kN, 2.0f);
    float *da = devFill(a), *db = devFill(b), *dout = devFill(std::vector<float>(kN, 0.0f)),
          *dout2 = devFill(std::vector<float>(kN, 0.0f));
    REQUIRE(da && db && dout && dout2);
    kcStream_t s = nullptr;
    REQUIRE_OK(kcStreamCreate(&s));
    REQUIRE_OK(kcStreamBeginCapture(s));
    enqueueChain(da, db, dout, dout2, s);  // recorded
    const kc::kernels::VectorAddArgs direct{da, db, dout2, kN};
    CHECK_OK(kcLaunchKernel("vector_add", &direct));  // the default stream still runs launches
    kcGraph_t g = nullptr;
    REQUIRE_OK(kcStreamEndCapture(s, &g));
    CHECK(host(dout2, kN) == std::vector<float>(kN, 3.0f));
    CHECK(host(dout, kN) == std::vector<float>(kN, 0.0f));
    CHECK_OK(kcGraphLaunch(g, s));
    CHECK_OK(kcStreamSynchronize(s));
    CHECK(host(dout, kN) == std::vector<float>(kN, 9.0f));  // (1 + 2) * 2 + 1 + 2
    CHECK_OK(kcGraphDestroy(g));
    CHECK_OK(kcStreamDestroy(s));
}

TEST_CASE(capture_rules) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    float* d = devFill(std::vector<float>(64, 1.0f));
    REQUIRE(d);
    kcGraph_t g = nullptr;
    CHECK_ERR(kcStreamEndCapture(nullptr, &g), Error::InvalidValue);  // not capturing
    CHECK_ERR(kcStreamEndCapture(nullptr, nullptr), Error::InvalidValue);
    REQUIRE_OK(kcStreamBeginCapture(nullptr));
    CHECK_ERR(kcStreamBeginCapture(nullptr), Error::InvalidValue);  // already capturing
    float h[64];
    CHECK_ERR(kcMemcpy(h, d, sizeof h, CopyKind::DeviceToHost), Error::UnsupportedOperation);
    CHECK_ERR(kcMemset(d, 0, 64), Error::UnsupportedOperation);
    CHECK_ERR(kcDeviceSynchronize(), Error::UnsupportedOperation);
    CHECK_ERR(kcStreamSynchronize(nullptr), Error::UnsupportedOperation);
    CHECK_ERR(kcFree(d), Error::UnsupportedOperation);
    const kc::kernels::VectorAddArgs bad{d, d, d, 1 << 20};  // validated at capture
    CHECK_ERR(kcLaunchKernel("vector_add", &bad), Error::InvalidPointer);
    REQUIRE_OK(kcStreamEndCapture(nullptr, &g));
    CHECK_OK(kcGraphLaunch(g));  // empty graph
    CHECK_OK(kcGraphDestroy(g));
    CHECK_ERR(kcGraphDestroy(g), Error::InvalidResourceHandle);
    CHECK_ERR(kcGraphLaunch(g), Error::InvalidResourceHandle);
    CHECK_OK(kcMemcpy(h, d, sizeof h, CopyKind::DeviceToHost));  // allowed again
    CHECK_OK(kcFree(d));
}

TEST_CASE(fault_in_a_graph_stops_it_and_names_the_node) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    const int n = 1024;
    float *out = devFill(std::vector<float>(n, 0.0f)), *a = devFill(std::vector<float>(n, 1.0f)),
          *after = devFill(std::vector<float>(n, 0.0f));
    REQUIRE(out && a && after);
    const kc::kernels::VectorAddArgs before{a, a, out, n};
    const testkernels::OobWriteArgs oob{out, n};
    const kc::kernels::VectorAddArgs later{a, a, after, n};
    REQUIRE_OK(kcStreamBeginCapture(nullptr));
    CHECK_OK(kcLaunchKernelChunked("vector_add", &before, 4, 64));
    CHECK_OK(kcLaunchKernel("oob_write", &oob));
    CHECK_OK(kcLaunchKernelChunked("vector_add", &later, 4, 64));
    kcGraph_t g = nullptr;
    REQUIRE_OK(kcStreamEndCapture(nullptr, &g));
    CHECK_ERR(kcGraphLaunch(g), Error::KernelFault);
    const std::string why = kcGetLastErrorDetail();
    std::printf("  graph fault detail: %s\n", why.c_str());
    CHECK(why.find("graph node 1 ('oob_write')") != std::string::npos);
    CHECK(after[0] == 0.0f);  // the node after the fault was skipped (device memory is host memory)
    CHECK_ERR(kcGraphLaunch(g), Error::ContextPoisoned);
    REQUIRE_OK(kcDeviceReset());  // frees the buffers and the graph
    CHECK_ERR(kcGraphLaunch(g), Error::InvalidResourceHandle);
}

TEST_CASE(concurrent_group_runs_independent_launches_in_one_step) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    const int n = 50000;
    std::vector<float> a(n), b(n);
    for (int i = 0; i < n; ++i) a[i] = float(i % 31), b[i] = float(i % 7) - 3.0f;
    float *da = devFill(a), *db = devFill(b), *o1 = devFill(std::vector<float>(n, 0.0f)),
          *o2 = devFill(std::vector<float>(n, 0.0f)), *o3 = devFill(std::vector<float>(n, 0.0f));
    REQUIRE(da && db && o1 && o2 && o3);
    CHECK_ERR(kcStreamConcurrentBegin(nullptr), Error::InvalidValue);  // not capturing
    REQUIRE_OK(kcStreamBeginCapture(nullptr));
    REQUIRE_OK(kcStreamConcurrentBegin(nullptr));
    CHECK_ERR(kcStreamConcurrentBegin(nullptr), Error::InvalidValue);  // already open
    const kc::kernels::VectorAddArgs s1{da, db, o1, n}, s2{db, db, o2, n};
    CHECK_OK(kcLaunchKernelChunked("vector_add", &s1, 4, 1024));  // independent: one step
    CHECK_OK(kcLaunchKernelChunked("vector_add", &s2, 4, 1024));
    REQUIRE_OK(kcStreamConcurrentEnd(nullptr));
    CHECK_ERR(kcStreamConcurrentEnd(nullptr), Error::InvalidValue);
    const kc::kernels::VectorAddArgs s3{o1, o2, o3, n};  // reads both: the next step
    CHECK_OK(kcLaunchKernelChunked("vector_add", &s3, 4, 1024));
    kcGraph_t g = nullptr;
    REQUIRE_OK(kcStreamEndCapture(nullptr, &g));
    CHECK_OK(kcGraphLaunch(g));
    std::vector<float> want(n);
    for (int i = 0; i < n; ++i) want[i] = (a[i] + b[i]) + (b[i] + b[i]);
    CHECK(host(o3, n) == want);
    CHECK_OK(kcGraphDestroy(g));
}
