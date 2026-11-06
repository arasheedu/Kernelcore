// pcuda_poc: end-to-end demo using only the pcuda runtime API.
// Exit code 0 on success, 1 on any failure (with a diagnostic on stderr).

#include <pcuda/api.hpp>
#include <pcuda/kernels.hpp>

#include <cstdio>
#include <vector>

PCUDA_DECLARE_KERNEL_IMAGE(vector_add);

namespace {

bool check(pcuda::Error e, const char* what) {
    if (e == pcuda::Error::Success) return true;
    std::fprintf(stderr, "error: %s failed: %s (%s)\n", what, pcudaGetErrorString(e), pcudaGetLastErrorDetail());
    return false;
}

int run() {
    using pcuda::CopyKind;
    if (!check(pcudaInit(), "pcudaInit")) return 1;

    pcuda::DeviceProperties props{};
    if (!check(pcudaGetDeviceProperties(&props, 0), "pcudaGetDeviceProperties")) return 1;
    std::printf("device 0      : %s\n", props.name);
    std::printf("workers       : %u\n", props.worker_count);
    std::printf("host memory   : %.1f GiB\n", static_cast<double>(props.total_memory) / (1u << 30));
    std::printf("guard pages   : %s\n", props.guard_pages ? "on" : "off");
    std::printf("kernel stack  : %zu MiB\n", props.kernel_stack_bytes >> 20);
    std::printf("vector_add.o  : %zu bytes, %s, sha256 %.16s...\n", pcuda_kimg_vector_add_size,
                pcuda_kimg_vector_add_provenance, pcuda_kimg_vector_add_sha256);

    const int n = 1 << 20;
    std::vector<float> a(n), b(n), out(n);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = 2.0f * static_cast<float>(i);
    }
    float *da = nullptr, *db = nullptr, *dout = nullptr;
    if (!check(pcudaMalloc(reinterpret_cast<void**>(&da), n * sizeof(float)), "pcudaMalloc a") ||
        !check(pcudaMalloc(reinterpret_cast<void**>(&db), n * sizeof(float)), "pcudaMalloc b") ||
        !check(pcudaMalloc(reinterpret_cast<void**>(&dout), n * sizeof(float)), "pcudaMalloc out"))
        return 1;
    if (!check(pcudaMemcpy(da, a.data(), n * sizeof(float), CopyKind::HostToDevice), "copy a") ||
        !check(pcudaMemcpy(db, b.data(), n * sizeof(float), CopyKind::HostToDevice), "copy b"))
        return 1;
    if (!check(pcuda::kernels::launchVectorAdd(da, db, dout, n), "launch vector_add")) return 1;
    if (!check(pcudaDeviceSynchronize(), "pcudaDeviceSynchronize")) return 1;
    if (!check(pcudaMemcpy(out.data(), dout, n * sizeof(float), CopyKind::DeviceToHost), "copy out")) return 1;

    for (int i = 0; i < n; ++i) {
        if (out[i] != a[i] + b[i]) {
            std::fprintf(stderr, "error: mismatch at %d: %f != %f\n", i, out[i], a[i] + b[i]);
            return 1;
        }
    }
    std::printf("vector_add    : %d elements correct\n", n);

    pcudaFree(da);
    pcudaFree(db);
    pcudaFree(dout);
    if (!check(pcudaShutdown(), "pcudaShutdown")) return 1;
    std::printf("PASS\n");
    return 0;
}

} // namespace

int main() { return run(); }
