// pcuda_bench: serial vs chunked vector_add. Reports median of 21 runs after 5 warm-ups.
// Scheduler overhead is measured with a zero-work-like tiny launch.

#include <pcuda/api.hpp>
#include <pcuda/kernels.hpp>

#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

template <class F>
double medianMicros(F&& f) {
    for (int i = 0; i < 5; ++i) f();
    std::vector<double> t;
    for (int i = 0; i < 21; ++i) {
        const auto s = Clock::now();
        f();
        t.push_back(std::chrono::duration<double, std::micro>(Clock::now() - s).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

const char* powerPlan() {
    SYSTEM_POWER_STATUS s{};
    if (GetSystemPowerStatus(&s)) return s.ACLineStatus == 1 ? "AC power" : "battery/unknown";
    return "unknown";
}

} // namespace

int main() {
    if (pcudaInit() != pcuda::Error::Success) {
        std::fprintf(stderr, "pcudaInit failed: %s\n", pcudaGetLastErrorDetail());
        return 1;
    }
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    std::printf("pcuda_bench: %u logical processors, %s, median of 21 runs (5 warm-up)\n", hw, powerPlan());
    std::printf("%12s %12s", "elements", "serial us");
    std::vector<unsigned> workers{2, 4};
    if (hw > 4) workers.push_back(hw);
    for (unsigned w : workers) std::printf("   %3uw us  speedup", w);
    std::printf("\n");

    for (int n : {1 << 10, 1 << 14, 1 << 18, 1 << 22, 1 << 24}) {
        float *a, *b, *o;
        if (pcudaMalloc(reinterpret_cast<void**>(&a), n * 4u) != pcuda::Error::Success ||
            pcudaMalloc(reinterpret_cast<void**>(&b), n * 4u) != pcuda::Error::Success ||
            pcudaMalloc(reinterpret_cast<void**>(&o), n * 4u) != pcuda::Error::Success) {
            std::fprintf(stderr, "allocation failed\n");
            return 1;
        }
        const double serial = medianMicros([&] { pcuda::kernels::launchVectorAdd(a, b, o, n); });
        std::printf("%12d %12.1f", n, serial);
        for (unsigned w : workers) {
            const double t = medianMicros([&] { pcuda::kernels::launchVectorAddChunked(a, b, o, n, w); });
            std::printf("  %9.1f  %6.2fx", t, serial / t);
        }
        std::printf("\n");
        pcudaFree(a);
        pcudaFree(b);
        pcudaFree(o);
    }

    float* t;
    pcudaMalloc(reinterpret_cast<void**>(&t), 64 * 4);
    const double launch = medianMicros([&] { pcuda::kernels::launchVectorAdd(t, t, t, 1); });
    const double chunked = medianMicros([&] { pcuda::kernels::launchVectorAddChunked(t, t, t, 1, 4); });
    std::printf("launch overhead (1 element): synchronous %.1f us, chunked %.1f us\n", launch, chunked);
    pcudaShutdown();
    return 0;
}
