// Streams and events: ordering, independence, error surfacing, shutdown with pending work.

#include "harness.hpp"

#include <kernelcore/kernels.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

using namespace std::chrono_literals;
using kc::CopyKind;
using kc::Error;

namespace {

struct OrderLog {
    std::mutex       m;
    std::vector<int> seen;
};
struct OrderItem {
    OrderLog* log;
    int       index;
};
void appendIndex(void* p) {
    auto* it = static_cast<OrderItem*>(p);
    std::lock_guard lk(it->log->m);
    it->log->seen.push_back(it->index);
}

// One-shot gate with a timeout so a broken implementation fails instead of hanging.
struct Gate {
    std::mutex              m;
    std::condition_variable cv;
    bool                    open = false;
    bool                    timed_out = false;
};
void waitGate(void* p) {
    auto* g = static_cast<Gate*>(p);
    std::unique_lock lk(g->m);
    if (!g->cv.wait_for(lk, 10s, [&] { return g->open; })) g->timed_out = true;
}
void openGate(void* p) {
    auto* g = static_cast<Gate*>(p);
    {
        std::lock_guard lk(g->m);
        g->open = true;
    }
    g->cv.notify_all();
}
void sleep50(void*) { std::this_thread::sleep_for(50ms); }
void bump(void* p) { static_cast<std::atomic<int>*>(p)->fetch_add(1); }

struct ReentryProbe {
    Error result = Error::Success;
};
void callApiFromHostFunc(void* p) {
    void* q = nullptr;
    static_cast<ReentryProbe*>(p)->result = kcMalloc(&q, 16);
}

} // namespace

TEST_CASE(stream_preserves_submission_order) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    kcStream_t s = nullptr;
    REQUIRE_OK(kcStreamCreate(&s));
    OrderLog log;
    std::vector<OrderItem> items(2000);
    for (int i = 0; i < 2000; ++i) {
        items[i] = {&log, i};
        CHECK_OK(kcLaunchHostFunc(s, &appendIndex, &items[i]));
    }
    CHECK_OK(kcStreamSynchronize(s));
    REQUIRE(log.seen.size() == 2000);
    bool ordered = true;
    for (int i = 0; i < 2000; ++i) ordered = ordered && log.seen[i] == i;
    CHECK(ordered);
    CHECK_OK(kcStreamDestroy(s));
}

TEST_CASE(two_streams_run_independently) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    kcStream_t a = nullptr, b = nullptr;
    REQUIRE_OK(kcStreamCreate(&a));
    REQUIRE_OK(kcStreamCreate(&b));
    Gate g;
    CHECK_OK(kcLaunchHostFunc(a, &waitGate, &g));  // blocks stream a ...
    CHECK_OK(kcLaunchHostFunc(b, &openGate, &g));  // ... until stream b runs
    CHECK_OK(kcStreamSynchronize(a));
    CHECK_OK(kcStreamSynchronize(b));
    CHECK(!g.timed_out);
    CHECK_OK(kcStreamDestroy(a));
    CHECK_OK(kcStreamDestroy(b));
}

TEST_CASE(events_mark_completion_points) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    kcStream_t s = nullptr;
    kcEvent_t ev = nullptr;
    REQUIRE_OK(kcStreamCreate(&s));
    REQUIRE_OK(kcEventCreate(&ev));
    CHECK_OK(kcEventQuery(ev));        // never recorded: complete
    CHECK_OK(kcEventSynchronize(ev));  // never recorded: returns immediately
    Gate g;
    CHECK_OK(kcLaunchHostFunc(s, &waitGate, &g));
    CHECK_OK(kcEventRecord(ev, s));
    CHECK_ERR(kcEventQuery(ev), Error::NotReady);
    CHECK_ERR(kcStreamQuery(s), Error::NotReady);
    openGate(&g);
    CHECK_OK(kcEventSynchronize(ev));
    CHECK_OK(kcEventQuery(ev));
    CHECK_OK(kcStreamSynchronize(s));
    CHECK_OK(kcStreamQuery(s));
    CHECK_OK(kcEventDestroy(ev));
    CHECK_OK(kcStreamDestroy(s));
}

TEST_CASE(invalid_handles_are_rejected) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    auto bogus_stream = reinterpret_cast<kcStream_t>(uintptr_t{0x1234});
    auto bogus_event = reinterpret_cast<kcEvent_t>(uintptr_t{0x5678});
    CHECK_ERR(kcStreamSynchronize(bogus_stream), Error::InvalidResourceHandle);
    CHECK_ERR(kcStreamDestroy(bogus_stream), Error::InvalidResourceHandle);
    CHECK_ERR(kcEventRecord(bogus_event), Error::InvalidResourceHandle);
    CHECK_ERR(kcEventSynchronize(bogus_event), Error::InvalidResourceHandle);
    kcStream_t s = nullptr;
    REQUIRE_OK(kcStreamCreate(&s));
    CHECK_OK(kcStreamDestroy(s));
    CHECK_ERR(kcStreamDestroy(s), Error::InvalidResourceHandle);  // already destroyed
    CHECK_ERR(kcLaunchHostFunc(s, &sleep50, nullptr), Error::InvalidResourceHandle);
}

TEST_CASE(host_funcs_cannot_reenter_the_api) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    ReentryProbe probe;
    CHECK_OK(kcLaunchHostFunc(nullptr, &callApiFromHostFunc, &probe));
    CHECK(probe.result == Error::UnsupportedOperation);
}

TEST_CASE(shutdown_drains_pending_work) {
    REQUIRE_OK(kcInit());
    kcStream_t s = nullptr;
    REQUIRE_OK(kcStreamCreate(&s));
    std::atomic<int> count{0};
    CHECK_OK(kcLaunchHostFunc(s, &sleep50, nullptr));
    for (int i = 0; i < 500; ++i) CHECK_OK(kcLaunchHostFunc(s, &bump, &count));
    CHECK_OK(kcShutdown());  // no explicit synchronize
    CHECK(count.load() == 500);
}

TEST_CASE(kernels_on_two_streams_produce_correct_results) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    const int n = 1 << 18;
    std::vector<float> a(n), b(n), r1(n), r2(n);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = 0.5f * static_cast<float>(i);
    }
    float *da, *db, *o1, *o2;
    REQUIRE_OK(kcMalloc(reinterpret_cast<void**>(&da), n * 4));
    REQUIRE_OK(kcMalloc(reinterpret_cast<void**>(&db), n * 4));
    REQUIRE_OK(kcMalloc(reinterpret_cast<void**>(&o1), n * 4));
    REQUIRE_OK(kcMalloc(reinterpret_cast<void**>(&o2), n * 4));
    CHECK_OK(kcMemcpy(da, a.data(), n * 4, CopyKind::HostToDevice));
    CHECK_OK(kcMemcpy(db, b.data(), n * 4, CopyKind::HostToDevice));
    kcStream_t s1, s2;
    REQUIRE_OK(kcStreamCreate(&s1));
    REQUIRE_OK(kcStreamCreate(&s2));
    for (int rep = 0; rep < 10; ++rep) {
        CHECK_OK(kc::kernels::launchVectorAdd(da, db, o1, n, s1));
        CHECK_OK(kc::kernels::launchVectorAdd(db, da, o2, n, s2));
    }
    CHECK_OK(kcDeviceSynchronize());
    CHECK_OK(kcMemcpy(r1.data(), o1, n * 4, CopyKind::DeviceToHost));
    CHECK_OK(kcMemcpy(r2.data(), o2, n * 4, CopyKind::DeviceToHost));
    bool ok = true;
    for (int i = 0; i < n; ++i) ok = ok && r1[i] == a[i] + b[i] && r2[i] == b[i] + a[i];
    CHECK(ok);
}

TEST_CASE(memcpy_waits_for_queued_kernels) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    const int n = 1 << 20;
    std::vector<float> a(n, 1.0f), out(n, 0.0f);
    float *da, *dout;
    REQUIRE_OK(kcMalloc(reinterpret_cast<void**>(&da), n * 4));
    REQUIRE_OK(kcMalloc(reinterpret_cast<void**>(&dout), n * 4));
    CHECK_OK(kcMemcpy(da, a.data(), n * 4, CopyKind::HostToDevice));
    kcStream_t s;
    REQUIRE_OK(kcStreamCreate(&s));
    CHECK_OK(kcLaunchHostFunc(s, &sleep50, nullptr));
    CHECK_OK(kc::kernels::launchVectorAdd(da, da, dout, n, s));
    CHECK_OK(kcMemcpy(out.data(), dout, n * 4, CopyKind::DeviceToHost));  // no explicit sync
    CHECK(out[0] == 2.0f && out[n - 1] == 2.0f);
}
