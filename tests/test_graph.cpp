// Graphs: captured launches run later, in order, with the same results as direct launches;
// capture rules; failures inside a graph.

#include "harness.hpp"
#include "test_kernels.hpp"

#include <pcuda/kernels.hpp>

#include <string>
#include <vector>

using pcuda::CopyKind;
using pcuda::Error;

namespace {

template <class T>
T* devFill(const std::vector<T>& v) {
    void* p = nullptr;
    if (pcudaMalloc(&p, v.size() * sizeof(T)) != Error::Success) return nullptr;
    if (pcudaMemcpy(p, v.data(), v.size() * sizeof(T), CopyKind::HostToDevice) != Error::Success) return nullptr;
    return static_cast<T*>(p);
}
template <class T>
std::vector<T> host(const T* d, size_t n) {
    std::vector<T> v(n);
    pcudaMemcpy(v.data(), d, n * sizeof(T), CopyKind::DeviceToHost);
    return v;
}

constexpr int kN = 100003;  // not a multiple of the chunk alignment

// out = a + b (chunked), then out2 = out * 2 + 1 (single launch), then out = out2 + b (chunked):
// each step reads the previous step's output, so a missing barrier changes the result.
void enqueueChain(float* a, float* b, float* out, float* out2, pcudaStream_t s) {
    const pcuda::kernels::VectorAddArgs add1{a, b, out, kN};
    const testkernels::ScaleAddArgs sc{out, out2, 2.0f, 1.0f, kN};
    const pcuda::kernels::VectorAddArgs add2{out2, b, out, kN};
    CHECK_OK(pcudaLaunchKernelChunked("vector_add", &add1, 8, 1024, s));
    CHECK_OK(pcudaLaunchKernel("scale_add", &sc, s));
    CHECK_OK(pcudaLaunchKernelChunked("vector_add", &add2, 8, 1024, s));
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

    REQUIRE_OK(pcudaMemset(dout, 0, kN * sizeof(float)));
    REQUIRE_OK(pcudaMemset(dout2, 0, kN * sizeof(float)));
    REQUIRE_OK(pcudaStreamBeginCapture(nullptr));
    enqueueChain(da, db, dout, dout2, nullptr);
    pcudaGraph_t g = nullptr;
    REQUIRE_OK(pcudaStreamEndCapture(nullptr, &g));
    REQUIRE(g != nullptr);
    CHECK(host(dout, kN) == std::vector<float>(kN, 0.0f));  // nothing ran during capture

    CHECK_OK(pcudaGraphLaunch(g));
    CHECK(host(dout, kN) == want);
    CHECK_OK(pcudaGraphLaunch(g));  // a graph can be launched again
    CHECK(host(dout, kN) == want);

    // Asynchronous launch on a created stream, destroyed before it has necessarily run.
    REQUIRE_OK(pcudaMemset(dout, 0, kN * sizeof(float)));
    pcudaStream_t s = nullptr;
    REQUIRE_OK(pcudaStreamCreate(&s));
    CHECK_OK(pcudaGraphLaunch(g, s));
    CHECK_OK(pcudaGraphDestroy(g));
    CHECK_OK(pcudaStreamSynchronize(s));
    CHECK(host(dout, kN) == want);
    CHECK_OK(pcudaStreamDestroy(s));
}

TEST_CASE(capture_on_a_created_stream_leaves_other_streams_running) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    std::vector<float> a(kN, 1.0f), b(kN, 2.0f);
    float *da = devFill(a), *db = devFill(b), *dout = devFill(std::vector<float>(kN, 0.0f)),
          *dout2 = devFill(std::vector<float>(kN, 0.0f));
    REQUIRE(da && db && dout && dout2);
    pcudaStream_t s = nullptr;
    REQUIRE_OK(pcudaStreamCreate(&s));
    REQUIRE_OK(pcudaStreamBeginCapture(s));
    enqueueChain(da, db, dout, dout2, s);  // recorded
    const pcuda::kernels::VectorAddArgs direct{da, db, dout2, kN};
    CHECK_OK(pcudaLaunchKernel("vector_add", &direct));  // the default stream still runs launches
    pcudaGraph_t g = nullptr;
    REQUIRE_OK(pcudaStreamEndCapture(s, &g));
    CHECK(host(dout2, kN) == std::vector<float>(kN, 3.0f));
    CHECK(host(dout, kN) == std::vector<float>(kN, 0.0f));
    CHECK_OK(pcudaGraphLaunch(g, s));
    CHECK_OK(pcudaStreamSynchronize(s));
    CHECK(host(dout, kN) == std::vector<float>(kN, 9.0f));  // (1 + 2) * 2 + 1 + 2
    CHECK_OK(pcudaGraphDestroy(g));
    CHECK_OK(pcudaStreamDestroy(s));
}

TEST_CASE(capture_rules) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    float* d = devFill(std::vector<float>(64, 1.0f));
    REQUIRE(d);
    pcudaGraph_t g = nullptr;
    CHECK_ERR(pcudaStreamEndCapture(nullptr, &g), Error::InvalidValue);  // not capturing
    CHECK_ERR(pcudaStreamEndCapture(nullptr, nullptr), Error::InvalidValue);
    REQUIRE_OK(pcudaStreamBeginCapture(nullptr));
    CHECK_ERR(pcudaStreamBeginCapture(nullptr), Error::InvalidValue);  // already capturing
    float h[64];
    CHECK_ERR(pcudaMemcpy(h, d, sizeof h, CopyKind::DeviceToHost), Error::UnsupportedOperation);
    CHECK_ERR(pcudaMemset(d, 0, 64), Error::UnsupportedOperation);
    CHECK_ERR(pcudaDeviceSynchronize(), Error::UnsupportedOperation);
    CHECK_ERR(pcudaStreamSynchronize(nullptr), Error::UnsupportedOperation);
    CHECK_ERR(pcudaFree(d), Error::UnsupportedOperation);
    const pcuda::kernels::VectorAddArgs bad{d, d, d, 1 << 20};  // validated at capture
    CHECK_ERR(pcudaLaunchKernel("vector_add", &bad), Error::InvalidPointer);
    REQUIRE_OK(pcudaStreamEndCapture(nullptr, &g));
    CHECK_OK(pcudaGraphLaunch(g));  // empty graph
    CHECK_OK(pcudaGraphDestroy(g));
    CHECK_ERR(pcudaGraphDestroy(g), Error::InvalidResourceHandle);
    CHECK_ERR(pcudaGraphLaunch(g), Error::InvalidResourceHandle);
    CHECK_OK(pcudaMemcpy(h, d, sizeof h, CopyKind::DeviceToHost));  // allowed again
    CHECK_OK(pcudaFree(d));
}

TEST_CASE(fault_in_a_graph_stops_it_and_names_the_node) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    const int n = 1024;
    float *out = devFill(std::vector<float>(n, 0.0f)), *a = devFill(std::vector<float>(n, 1.0f)),
          *after = devFill(std::vector<float>(n, 0.0f));
    REQUIRE(out && a && after);
    const pcuda::kernels::VectorAddArgs before{a, a, out, n};
    const testkernels::OobWriteArgs oob{out, n};
    const pcuda::kernels::VectorAddArgs later{a, a, after, n};
    REQUIRE_OK(pcudaStreamBeginCapture(nullptr));
    CHECK_OK(pcudaLaunchKernelChunked("vector_add", &before, 4, 64));
    CHECK_OK(pcudaLaunchKernel("oob_write", &oob));
    CHECK_OK(pcudaLaunchKernelChunked("vector_add", &later, 4, 64));
    pcudaGraph_t g = nullptr;
    REQUIRE_OK(pcudaStreamEndCapture(nullptr, &g));
    CHECK_ERR(pcudaGraphLaunch(g), Error::KernelFault);
    const std::string why = pcudaGetLastErrorDetail();
    std::printf("  graph fault detail: %s\n", why.c_str());
    CHECK(why.find("graph node 1 ('oob_write')") != std::string::npos);
    CHECK(after[0] == 0.0f);  // the node after the fault was skipped (device memory is host memory)
    CHECK_ERR(pcudaGraphLaunch(g), Error::ContextPoisoned);
    REQUIRE_OK(pcudaDeviceReset());  // frees the buffers and the graph
    CHECK_ERR(pcudaGraphLaunch(g), Error::InvalidResourceHandle);
}
