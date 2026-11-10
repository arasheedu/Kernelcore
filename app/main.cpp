// kc_poc: end-to-end demo using only the Kernelcore runtime API.
// Exit code 0 on success, 1 on any failure (with a diagnostic on stderr).

#include <kernelcore/api.hpp>
#include <kernelcore/kernels.hpp>

#include <cstdio>
#include <vector>

KC_DECLARE_KERNEL_IMAGE(vector_add);

namespace {

bool check(kc::Error e, const char* what) {
    if (e == kc::Error::Success) return true;
    std::fprintf(stderr, "error: %s failed: %s (%s)\n", what, kcGetErrorString(e), kcGetLastErrorDetail());
    return false;
}

int run() {
    using kc::CopyKind;
    if (!check(kcInit(), "kcInit")) return 1;

    kc::DeviceProperties props{};
    if (!check(kcGetDeviceProperties(&props, 0), "kcGetDeviceProperties")) return 1;
    std::printf("device 0      : %s\n", props.name);
    std::printf("workers       : %u (+ caller; %u on the fastest cores)\n", props.worker_count, props.performance_workers);
    std::printf("host memory   : %.1f GiB\n", static_cast<double>(props.total_memory) / (1u << 30));
    std::printf("guard pages   : %s\n", props.guard_pages ? "on" : "off");
    std::printf("kernel stack  : %zu MiB\n", props.kernel_stack_bytes >> 20);
    std::printf("vector_add.o  : %zu bytes, %s, sha256 %.16s...\n", kc_kimg_vector_add_size,
                kc_kimg_vector_add_provenance, kc_kimg_vector_add_sha256);

    const int n = 1 << 20;
    std::vector<float> a(n), b(n), out(n);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = 2.0f * static_cast<float>(i);
    }
    float *da = nullptr, *db = nullptr, *dout = nullptr;
    if (!check(kcMalloc(reinterpret_cast<void**>(&da), n * sizeof(float)), "kcMalloc a") ||
        !check(kcMalloc(reinterpret_cast<void**>(&db), n * sizeof(float)), "kcMalloc b") ||
        !check(kcMalloc(reinterpret_cast<void**>(&dout), n * sizeof(float)), "kcMalloc out"))
        return 1;
    if (!check(kcMemcpy(da, a.data(), n * sizeof(float), CopyKind::HostToDevice), "copy a") ||
        !check(kcMemcpy(db, b.data(), n * sizeof(float), CopyKind::HostToDevice), "copy b"))
        return 1;
    if (!check(kc::kernels::launchVectorAdd(da, db, dout, n), "launch vector_add")) return 1;
    if (!check(kcDeviceSynchronize(), "kcDeviceSynchronize")) return 1;
    if (!check(kcMemcpy(out.data(), dout, n * sizeof(float), CopyKind::DeviceToHost), "copy out")) return 1;

    for (int i = 0; i < n; ++i) {
        if (out[i] != a[i] + b[i]) {
            std::fprintf(stderr, "error: mismatch at %d: %f != %f\n", i, out[i], a[i] + b[i]);
            return 1;
        }
    }
    std::printf("vector_add    : %d elements correct\n", n);

    kcFree(da);
    kcFree(db);
    kcFree(dout);
    if (!check(kcShutdown(), "kcShutdown")) return 1;
    std::printf("PASS\n");
    return 0;
}

} // namespace

int main() { return run(); }
