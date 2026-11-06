// N-core chunking experiment: planner properties and results equal to the serial reference.

#include "harness.hpp"

#include "stream.hpp"

#include <pcuda/kernels.hpp>

#include <bit>
#include <cstdint>
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
    CHECK(!pcuda::chunksCoverExactly({{0, 10}, {9, 5}}, 14));   // overlap
    CHECK(!pcuda::chunksCoverExactly({{0, 10}, {11, 5}}, 16));  // gap
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
