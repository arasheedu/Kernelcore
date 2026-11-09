// N-core chunking experiment: planner properties and results equal to the serial reference.

#include "harness.hpp"

#include "stream.hpp"
#include "test_kernels.hpp"

#include <pcuda/kernels.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

using pcuda::CopyKind;
using pcuda::Error;

TEST_CASE(planner_covers_exactly_and_respects_limits) {
    for (int64_t total : {1ll, 63ll, 64ll, 65ll, 4095ll, 4096ll, 4097ll, 100003ll, 1ll << 24}) {
        for (unsigned workers : {1u, 2u, 3u, 4u, 7u, 16u, 64u}) {
            for (int64_t min_chunk : {1ll, 64ll, 4096ll}) {
                const auto chunks = pcuda::planChunks(total, workers, min_chunk);
                CHECK(pcuda::chunksCoverExactly(chunks, total));
                CHECK(chunks.size() <= workers);
                for (size_t i = 0; i + 1 < chunks.size(); ++i) {
                    CHECK(chunks[i].length % 64 == 0);  // interior boundaries 64-element aligned
                    CHECK(chunks[i].length >= min_chunk);
                }
            }
        }
    }
    CHECK(pcuda::planChunks(0, 4, 64).empty());
    // Guided: exact cover, aligned interior boundaries, sizes never grow, the first is about a
    // 2 * threads share.
    for (int64_t total : {1ll, 63ll, 4097ll, 100003ll, 1ll << 24}) {
        for (unsigned threads : {1u, 3u, 12u, 64u}) {
            for (int64_t min_chunk : {1ll, 16ll, 4096ll}) {
                const auto chunks = pcuda::planChunksGuided(total, threads, min_chunk, 16);
                CHECK(pcuda::chunksCoverExactly(chunks, total));
                for (size_t i = 0; i + 1 < chunks.size(); ++i) {
                    CHECK(chunks[i].length % 16 == 0);
                    CHECK(chunks[i].length >= min_chunk);
                    CHECK(chunks[i + 1].length <= chunks[i].length);
                }
                CHECK(chunks.front().length <= std::max<int64_t>((total / (2 * threads) + 15) / 16 * 16, std::max<int64_t>(min_chunk + 15, 16)) ||
                      chunks.size() == 1);
            }
        }
    }
    CHECK(pcuda::planChunksGuided(0, 4, 16).empty());
    CHECK(!pcuda::chunksCoverExactly({{0, 10}, {9, 5}}, 14));   // overlap
    CHECK(!pcuda::chunksCoverExactly({{0, 10}, {11, 5}}, 16));  // gap
}

TEST_CASE(pool_steps_respect_their_thread_caps) {
    pcuda::WorkerPool pool(7, 1 << 20);
    constexpr int kTasks = 64;
    struct Log {
        std::mutex                   m;
        std::vector<std::thread::id> ids;
        int                          runs = 0;
    };
    // Tasks that last long enough for every pool thread to join; each records its thread.
    auto make = [](Log& log) {
        std::vector<pcuda::Stream::Task> t;
        for (int i = 0; i < kTasks; ++i)
            t.push_back([&log]() -> Error {
                std::this_thread::sleep_for(std::chrono::microseconds(300));
                std::lock_guard lk(log.m);
                ++log.runs;
                if (std::find(log.ids.begin(), log.ids.end(), std::this_thread::get_id()) == log.ids.end())
                    log.ids.push_back(std::this_thread::get_id());
                return Error::Success;
            });
        return t;
    };
    for (unsigned cap : {1u, 2u, 5u}) {
        Log log;
        CHECK_OK(pool.run(make(log), cap));
        CHECK(log.runs == kTasks);
        CHECK(log.ids.size() <= cap);
    }
    // A job of steps with different caps (a graph): each step keeps its own cap.
    Log a, b, c;
    const auto ta = make(a), tb = make(b), tc = make(c);
    const pcuda::WorkerPool::Step steps[] = {{ta.data(), ta.size(), 1}, {tb.data(), tb.size(), 0}, {tc.data(), tc.size(), 3}};
    CHECK_OK(pool.runSteps(steps, nullptr));
    CHECK(a.runs == kTasks && b.runs == kTasks && c.runs == kTasks);
    CHECK(a.ids.size() == 1);
    CHECK(a.ids[0] == std::this_thread::get_id());  // a cap of 1 leaves the step to the caller
    CHECK(b.ids.size() > 1);                        // uncapped: the pool joins in
    CHECK(c.ids.size() <= 3);
}

TEST_CASE(chunked_vector_add_matches_serial_reference) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    for (int n : {1, 1000, 4097, (1 << 20) + 3}) {
        std::vector<float> a(n), b(n), serial(n), chunked(n);
        for (int i = 0; i < n; ++i) {
            a[i] = static_cast<float>(i % 977) * 1.25f;
            b[i] = static_cast<float>(i % 389) * -0.5f;
        }
        float *da, *db, *ds, *dc;
        REQUIRE_OK(pcudaMalloc(reinterpret_cast<void**>(&da), n * 4u));
        REQUIRE_OK(pcudaMalloc(reinterpret_cast<void**>(&db), n * 4u));
        REQUIRE_OK(pcudaMalloc(reinterpret_cast<void**>(&ds), n * 4u));
        REQUIRE_OK(pcudaMalloc(reinterpret_cast<void**>(&dc), n * 4u));
        CHECK_OK(pcudaMemcpy(da, a.data(), n * 4u, CopyKind::HostToDevice));
        CHECK_OK(pcudaMemcpy(db, b.data(), n * 4u, CopyKind::HostToDevice));
        CHECK_OK(pcuda::kernels::launchVectorAdd(da, db, ds, n));
        CHECK_OK(pcudaMemcpy(serial.data(), ds, n * 4u, CopyKind::DeviceToHost));
        for (unsigned w : {1u, 2u, 4u, hw}) {
            for (int64_t min_chunk : {int64_t{64}, int64_t{4096}}) {
                CHECK_OK(pcudaMemset(dc, 0xFF, n * 4u));
                CHECK_OK(pcuda::kernels::launchVectorAddChunked(da, db, dc, n, w, min_chunk));
                CHECK_OK(pcudaMemcpy(chunked.data(), dc, n * 4u, CopyKind::DeviceToHost));
                bool same = true;
                for (int i = 0; i < n; ++i)
                    same = same && std::bit_cast<uint32_t>(chunked[i]) == std::bit_cast<uint32_t>(serial[i]);
                if (!same)
                    th::fail(__FILE__, __LINE__, "mismatch n=" + std::to_string(n) + " workers=" + std::to_string(w));
            }
        }
        CHECK_OK(pcudaFree(da));
        CHECK_OK(pcudaFree(db));
        CHECK_OK(pcudaFree(ds));
        CHECK_OK(pcudaFree(dc));
    }
}

// scale_add (float scalars in xmm) and trig_mix (libm externals) opt into chunking too.
TEST_CASE(chunked_scale_add_and_trig_mix_match_serial) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    for (int n : {1, 4097, (1 << 18) + 5}) {
        std::vector<float> x(n), serial(n), chunked(n);
        for (int i = 0; i < n; ++i) x[i] = static_cast<float>(i % 2001) / 1000.0f - 1.0f;  // [-1, 1]
        float *dx, *ds, *dc;
        REQUIRE_OK(pcudaMalloc(reinterpret_cast<void**>(&dx), n * 4u));
        REQUIRE_OK(pcudaMalloc(reinterpret_cast<void**>(&ds), n * 4u));
        REQUIRE_OK(pcudaMalloc(reinterpret_cast<void**>(&dc), n * 4u));
        CHECK_OK(pcudaMemcpy(dx, x.data(), n * 4u, CopyKind::HostToDevice));
        const testkernels::ScaleAddArgs sa_s{dx, ds, 2.5f, 1.0f, n}, sa_c{dx, dc, 2.5f, 1.0f, n};
        const testkernels::TrigMixArgs tm_s{dx, ds, n}, tm_c{dx, dc, n};
        struct Case { const char* name; const void* serial; const void* chunked; };
        for (const Case& c : {Case{"scale_add", &sa_s, &sa_c}, Case{"trig_mix", &tm_s, &tm_c}}) {
            CHECK_OK(pcudaLaunchKernel(c.name, c.serial));
            CHECK_OK(pcudaMemcpy(serial.data(), ds, n * 4u, CopyKind::DeviceToHost));
            for (unsigned w : {2u, hw}) {
                CHECK_OK(pcudaMemset(dc, 0xFF, n * 4u));
                CHECK_OK(pcudaLaunchKernelChunked(c.name, c.chunked, w, 64));
                CHECK_OK(pcudaMemcpy(chunked.data(), dc, n * 4u, CopyKind::DeviceToHost));
                bool same = true;
                for (int i = 0; i < n; ++i)
                    same = same && std::bit_cast<uint32_t>(chunked[i]) == std::bit_cast<uint32_t>(serial[i]);
                if (!same)
                    th::fail(__FILE__, __LINE__,
                             std::string(c.name) + " mismatch n=" + std::to_string(n) + " workers=" + std::to_string(w));
            }
        }
        CHECK_OK(pcudaFree(dx));
        CHECK_OK(pcudaFree(ds));
        CHECK_OK(pcudaFree(dc));
    }
}

TEST_CASE(chunked_launch_on_a_stream) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    const int n = 1 << 20;
    std::vector<float> a(n, 3.0f), out(n);
    float *da, *dout;
    REQUIRE_OK(pcudaMalloc(reinterpret_cast<void**>(&da), n * 4u));
    REQUIRE_OK(pcudaMalloc(reinterpret_cast<void**>(&dout), n * 4u));
    CHECK_OK(pcudaMemcpy(da, a.data(), n * 4u, CopyKind::HostToDevice));
    pcudaStream_t s;
    REQUIRE_OK(pcudaStreamCreate(&s));
    CHECK_OK(pcuda::kernels::launchVectorAddChunked(da, da, dout, n, 4, 4096, s));
    CHECK_OK(pcudaStreamSynchronize(s));
    CHECK_OK(pcudaMemcpy(out.data(), dout, n * 4u, CopyKind::DeviceToHost));
    CHECK(out[0] == 6.0f && out[n / 2] == 6.0f && out[n - 1] == 6.0f);
}

TEST_CASE(chunked_argument_validation) {
    RuntimeFixture fx;
    REQUIRE(fx.ok);
    float* d;
    REQUIRE_OK(pcudaMalloc(reinterpret_cast<void**>(&d), 256 * 4));
    CHECK_ERR(pcuda::kernels::launchVectorAddChunked(d, d, d, 256, 0), Error::InvalidValue);
    CHECK_ERR(pcuda::kernels::launchVectorAddChunked(d, d, d, 256, 5000), Error::InvalidValue);
    CHECK_ERR(pcuda::kernels::launchVectorAddChunked(d, d, d, 257, 4), Error::InvalidPointer);
    CHECK_OK(pcuda::kernels::launchVectorAddChunked(d, d, d, 0, 4));
}
