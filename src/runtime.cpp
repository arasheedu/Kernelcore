// pcuda runtime core: lifecycle, device, memory, launch, streams, events, diagnostics.

#include "allocation_registry.hpp"
#include "builtin_kernels.hpp"
#include "cpu/cpu_backend.hpp"
#include "detail.hpp"
#include "stream.hpp"

#include <pcuda/api.hpp>

#include <windows.h>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#if !defined(_M_X64) || defined(_M_ARM64EC)
#error "pcuda requires a native x64 build: Booth --cpu emits x86-64 System V code"
#endif

using pcuda::Error;

// ---------------------------------------------------------------- handles

struct pcudaStream_st {
    std::unique_ptr<pcuda::Stream> stream;
};

namespace {
struct EventState {
    std::mutex              mutex;
    std::condition_variable cv;
    uint64_t                recorded = 0;   // generation of the latest record
    uint64_t                completed = 0;  // highest completed generation
};
} // namespace

namespace {
// Logical processors on the fastest cores (P-cores on hybrid Intel CPUs; all of them otherwise).
// Busy efficiency cores make memory-bound launches slower overall on such CPUs (they share the
// ring and memory path but finish chunks late), so the pool leaves them alone by default.
unsigned performanceLogicalProcessors() {
    const unsigned all = std::max(1u, std::thread::hardware_concurrency());
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return all;
    std::vector<uint8_t> buf(len);
    auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len)) return all;
    unsigned best_class = 0, count = 0;
    for (DWORD off = 0; off < len;) {
        auto* e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
        const unsigned cls = e->Processor.EfficiencyClass;
        unsigned lps = 0;
        for (WORD g = 0; g < e->Processor.GroupCount; ++g)
            lps += static_cast<unsigned>(__popcnt64(e->Processor.GroupMask[g].Mask));
        if (cls > best_class) best_class = cls, count = 0;
        if (cls == best_class) count += lps;
        off += e->Size;
    }
    return count ? count : all;
}
} // namespace

struct pcudaEvent_st {
    std::shared_ptr<EventState> state = std::make_shared<EventState>();
};

namespace {
// One captured kernel launch: its argument sets (one per chunk), validated at capture.
struct GraphNode {
    const pcuda::KernelEntry*      entry;
    std::string                    name;
    std::vector<pcuda::LaunchArgs> parts;
};
// Immutable once captured; a running launch keeps it alive after pcudaGraphDestroy.
struct GraphData {
    std::vector<GraphNode>                        nodes;
    std::vector<std::vector<pcuda::Stream::Task>> tasks;  // per node: one task per part
    std::vector<pcuda::WorkerPool::Step>          steps;
};
} // namespace

struct pcudaGraph_st {
    std::shared_ptr<const GraphData> data;
};

// ---------------------------------------------------------------- diagnostics

namespace pcuda::detail {
namespace {
thread_local std::string t_detail;
thread_local bool        t_runtime_thread = false;
} // namespace

void setDetail(std::string text) noexcept {
    try {
        t_detail = std::move(text);
    } catch (...) {
    }
}
void clearDetail() noexcept { t_detail.clear(); }
const std::string& lastDetail() noexcept { return t_detail; }
void markRuntimeThread() noexcept { t_runtime_thread = true; }
bool isRuntimeThread() noexcept { return t_runtime_thread; }
} // namespace pcuda::detail

namespace {

using namespace pcuda;
using detail::setDetail;

constexpr size_t kMiB = 1024 * 1024;

struct Runtime {
    std::shared_mutex lifecycle;  // shared by API calls, exclusive for init/shutdown/reset
    bool              initialized = false;
    std::atomic<bool> poisoned{false};

    bool     guard_pages = true;
    bool     large_pages = false;  // PCUDA_LARGE_PAGES=1 and the privilege could be enabled
    size_t   stack_bytes = 8 * kMiB;
    unsigned hw_threads = 1;
    unsigned chunks_per_worker = 4;  // chunked launches: chunks = workers * this (load balance)
    unsigned pool_spin_us = 50;      // pool threads spin this long before sleeping
    int      current_device = 0;

    AllocationRegistry               registry;
    std::unique_ptr<cpu::CpuBackend> backend;
    std::unique_ptr<Stream>          default_stream;

    std::mutex                                                       objects_mutex;
    std::map<pcudaStream_t, std::unique_ptr<pcudaStream_st>>         streams;
    std::map<pcudaEvent_t, std::unique_ptr<pcudaEvent_st>>           events;
    std::map<pcudaGraph_t, std::unique_ptr<pcudaGraph_st>>           graphs;

    // Streams being captured: launches on them are recorded into a graph instead of run.
    std::mutex                                                 capture_mutex;
    std::map<Stream*, std::shared_ptr<std::vector<GraphNode>>> captures;
    std::atomic<int>                                           capture_count{0};

    std::mutex                                kernels_mutex;
    std::map<std::string, const KernelEntry*> kernels;
    bool                                      builtins_registered = false;

    std::mutex                  pool_mutex;
    std::unique_ptr<WorkerPool> pool;

    std::mutex fault_mutex;
    FaultInfo  last_fault{};
    bool       has_fault = false;

    // Kernel code generation path; kept across device resets. Set from PCUDA_CODEGEN at first
    // init unless pcudaSetKernelCodegen ran earlier.
    std::atomic<Codegen> codegen{Codegen::Auto};
    std::atomic<bool>    codegen_set{false};
};

// Intentionally leaked: no thread joins or frees run from static destructors.
Runtime& R() {
    static Runtime* r = new Runtime();
    return *r;
}

std::string envString(const char* name) {
    char buf[256];
    const DWORD n = GetEnvironmentVariableA(name, buf, sizeof buf);
    return (n == 0 || n >= sizeof buf) ? std::string() : std::string(buf, n);
}

long long envInt(const char* name, long long fallback, long long lo, long long hi) {
    const std::string s = envString(name);
    if (s.empty()) return fallback;
    char* end = nullptr;
    const long long v = std::strtoll(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0' || v < lo || v > hi) return fallback;
    return v;
}

// Wraps every API entry: no exception escapes, detail text is reset per call.
template <class F>
Error api(const char* fn, F&& body) noexcept {
    detail::clearDetail();
    if (detail::isRuntimeThread()) {
        setDetail(std::format("{}: pcuda API called from a runtime worker thread (host funcs must not call pcuda)", fn));
        return Error::UnsupportedOperation;
    }
    try {
        return body();
    } catch (const std::bad_alloc&) {
        setDetail(std::format("{}: out of host memory", fn));
        return Error::OutOfMemory;
    } catch (const std::exception& e) {
        setDetail(std::format("{}: internal error: {}", fn, e.what()));
        return Error::InternalError;
    } catch (...) {
        setDetail(std::format("{}: unknown internal error", fn));
        return Error::InternalError;
    }
}

Error requireInit(const Runtime& r) {
    if (!r.initialized) {
        setDetail("runtime not initialized: call pcudaInit first");
        return Error::NotInitialized;
    }
    return Error::Success;
}

Error requireUsable(const Runtime& r) {
    if (Error e = requireInit(r); e != Error::Success) return e;
    if (r.poisoned.load()) {
        setDetail("context poisoned by an earlier kernel fault: call pcudaDeviceReset");
        return Error::ContextPoisoned;
    }
    return Error::Success;
}

void recordFault(Runtime& r, const FaultInfo& f) {
    std::lock_guard lk(r.fault_mutex);
    r.last_fault = f;
    r.has_fault = true;
    r.poisoned.store(true);
}

// All streams idle; recorded errors are left in place.
void waitAllIdle(Runtime& r) {
    if (r.default_stream) r.default_stream->waitIdle();
    std::vector<Stream*> snap;
    {
        std::lock_guard lk(r.objects_mutex);
        for (auto& [h, s] : r.streams) snap.push_back(s->stream.get());
    }
    for (Stream* s : snap) s->waitIdle();
}

Error findStream(Runtime& r, pcudaStream_t h, Stream** out) {
    if (!h) {
        *out = r.default_stream.get();
        return Error::Success;
    }
    std::lock_guard lk(r.objects_mutex);
    auto it = r.streams.find(h);
    if (it == r.streams.end()) {
        setDetail("unknown stream handle");
        return Error::InvalidResourceHandle;
    }
    *out = it->second->stream.get();
    return Error::Success;
}

Error findEvent(Runtime& r, pcudaEvent_t h, std::shared_ptr<EventState>* out) {
    std::lock_guard lk(r.objects_mutex);
    auto it = r.events.find(h);
    if (!h || it == r.events.end()) {
        setDetail("unknown event handle");
        return Error::InvalidResourceHandle;
    }
    *out = it->second->state;
    return Error::Success;
}

Error findKernel(Runtime& r, const char* name, const KernelEntry** out) {
    if (!name) {
        setDetail("kernel name is null");
        return Error::InvalidValue;
    }
    std::lock_guard lk(r.kernels_mutex);
    auto it = r.kernels.find(name);
    if (it == r.kernels.end()) {
        setDetail(std::format("no kernel registered as '{}'", name));
        return Error::KernelNotFound;
    }
    *out = it->second;
    return Error::Success;
}

Error registerEntry(Runtime& r, const KernelEntry* e) {
    if (!e || !e->name || !*e->name || !e->image || e->image_size == 0 || !e->marshal) {
        setDetail("kernel entry is incomplete (name, image and marshal are required)");
        return Error::InvalidValue;
    }
    if ((e->extent == nullptr) != (e->marshal_chunk == nullptr)) {
        setDetail("chunked kernels must provide both extent and marshal_chunk");
        return Error::InvalidValue;
    }
    std::lock_guard lk(r.kernels_mutex);
    auto [it, inserted] = r.kernels.try_emplace(e->name, e);
    if (!inserted && it->second != e) {
        setDetail(std::format("a different kernel is already registered as '{}'", e->name));
        return Error::InvalidValue;
    }
    return Error::Success;
}

class RegistryContext final : public LaunchContext {
public:
    explicit RegistryContext(Runtime& r) : r_(r) {}
    Error checkDeviceRange(const void* ptr, size_t bytes, const char* what) const noexcept override {
        try {
            std::string why;
            const Error e = r_.registry.checkRange(ptr, bytes, why);
            if (e != Error::Success) setDetail(std::format("argument '{}': {}", what ? what : "?", why));
            return e;
        } catch (...) {
            return Error::InternalError;
        }
    }

private:
    Runtime& r_;
};

// Creates backend, default stream and kernel table. Caller holds lifecycle exclusively.
Error startDevice(Runtime& r) {
    r.guard_pages = envInt("PCUDA_GUARD_PAGES", 1, 0, 1) != 0;
    r.stack_bytes = static_cast<size_t>(envInt("PCUDA_KERNEL_STACK_MB", 8, 1, 1024)) * kMiB;
    // Threads in the chunked-launch pool. A launch's `workers` argument sets the chunk count.
    // Default: the performance cores' logical processors, minus one for the launching (stream)
    // thread, which runs chunks too. More threads than processors made waiting workers take
    // processors from the threads with work (measured: 16 threads ~13% slower than 12).
    r.hw_threads = static_cast<unsigned>(
        envInt("PCUDA_POOL_THREADS", std::max(1u, performanceLogicalProcessors() - 1), 1, 1024));
    r.chunks_per_worker = static_cast<unsigned>(envInt("PCUDA_CHUNKS_PER_WORKER", 4, 1, 64));
    r.pool_spin_us = static_cast<unsigned>(envInt("PCUDA_POOL_SPIN_US", 50, 0, 10000));
    if (!r.codegen_set.load()) {
        const std::string cg = envString("PCUDA_CODEGEN");
        if (cg == "booth") r.codegen.store(Codegen::Booth);
        else if (cg == "native") r.codegen.store(Codegen::Native);
        else if (cg == "simd") r.codegen.store(Codegen::Simd);
        else r.codegen.store(Codegen::Auto);
    }
    // Large pages for big allocations (opt-in): needs the "Lock pages in memory" right
    // (scriptsgrant_lock_pages.ps1). Without it, allocations silently use normal pages.
    r.large_pages = envInt("PCUDA_LARGE_PAGES", 0, 0, 1) != 0 && cpu::enableLargePages();
    r.backend = std::make_unique<cpu::CpuBackend>(r.registry, r.guard_pages, r.large_pages);
    r.backend->setCodegen(r.codegen.load());
    r.default_stream = std::make_unique<Stream>(r.stack_bytes, L"pcuda default stream", r.pool_spin_us);
    r.poisoned.store(false);
    {
        std::lock_guard lk(r.fault_mutex);
        r.last_fault = FaultInfo{};
        r.has_fault = false;
    }
    if (!r.builtins_registered) {
        std::vector<const KernelEntry*> builtins;
        registerBuiltinKernels(builtins);
        for (const KernelEntry* e : builtins)
            if (Error err = registerEntry(r, e); err != Error::Success) return err;
        r.builtins_registered = true;
    }
    return Error::Success;
}

// Drains and destroys everything device-side. Caller holds lifecycle exclusively.
size_t stopDevice(Runtime& r) {
    {
        std::map<pcudaStream_t, std::unique_ptr<pcudaStream_st>> streams;
        {
            std::lock_guard lk(r.objects_mutex);
            streams.swap(r.streams);
        }
        streams.clear();  // each Stream destructor drains, then joins
    }
    r.default_stream.reset();
    {
        std::lock_guard lk(r.pool_mutex);
        r.pool.reset();
    }
    {
        std::lock_guard lk(r.objects_mutex);
        r.events.clear();
        r.graphs.clear();
    }
    {
        std::lock_guard lk(r.capture_mutex);
        r.captures.clear();
        r.capture_count.store(0);
    }
    const auto blocks = r.registry.drain();
    for (const auto& b : blocks) cpu::releaseBlock(b);
    r.backend.reset();  // unloads kernel images, releases the fault guard
    return blocks.size();
}

Error checkPlatform() {
    USHORT process_machine = 0, native_machine = 0;
    if (IsWow64Process2(GetCurrentProcess(), &process_machine, &native_machine) &&
        native_machine != IMAGE_FILE_MACHINE_AMD64 && envString("PCUDA_ALLOW_EMULATION") != "1") {
        setDetail("host CPU is not x64 (running under emulation); set PCUDA_ALLOW_EMULATION=1 to try anyway");
        return Error::UnsupportedOperation;
    }
    return Error::Success;
}

WorkerPool* ensurePool(Runtime& r) {
    std::lock_guard lk(r.pool_mutex);
    if (!r.pool) r.pool = std::make_unique<WorkerPool>(r.hw_threads, r.stack_bytes, r.pool_spin_us);
    return r.pool.get();
}

// Runs one argument set of a kernel; `a` must outlive the task.
Stream::Task partTask(const KernelEntry* entry, const LaunchArgs* a) {
    return [entry, a]() -> Error {
        Runtime& rt = R();
        const Error e = rt.backend->launch(*entry, *a);
        if (e == Error::KernelFault) recordFault(rt, rt.backend->lastFault());
        return e;
    };
}

Error skipIfPoisoned(const Runtime& r) {
    if (!r.poisoned.load()) return Error::Success;
    setDetail("skipped: context poisoned by an earlier kernel fault");
    return Error::ContextPoisoned;
}

// Copies, memsets and synchronization run at once, so during a capture they would overtake the
// recorded work. Like CUDA, they are refused until the capture ends.
Error rejectDuringCapture(Runtime& r, const char* what) {
    if (r.capture_count.load() == 0) return Error::Success;
    setDetail(std::format("{} is not allowed while a stream is capturing (end the capture first)", what));
    return Error::UnsupportedOperation;
}
bool isCapturing(Runtime& r, Stream* s) {
    if (r.capture_count.load() == 0) return false;
    std::lock_guard lk(r.capture_mutex);
    return r.captures.count(s) != 0;
}

Error launchCommon(const char* name, const void* typed_args, pcudaStream_t stream, bool chunked,
                   unsigned workers, int64_t min_chunk) {
    Runtime& r = R();
    std::shared_lock life(r.lifecycle);
    if (Error e = requireUsable(r); e != Error::Success) return e;
    const KernelEntry* entry = nullptr;
    if (Error e = findKernel(r, name, &entry); e != Error::Success) return e;
    Stream* target = nullptr;
    if (Error e = findStream(r, stream, &target); e != Error::Success) return e;
    if (!typed_args) {
        setDetail("typed_args is null");
        return Error::InvalidValue;
    }

    RegistryContext ctx(r);
    LaunchArgs whole;
    if (Error e = entry->marshal(typed_args, ctx, &whole); e != Error::Success) {
        if (detail::lastDetail().empty()) setDetail(std::format("kernel '{}': argument validation failed", name));
        return e;
    }
    if (whole.nthreads == 0) return Error::Success;

    // Load the image now so image errors are reported synchronously, even for async launches.
    const cpu::BoothImage* img = nullptr;
    if (Error e = r.backend->image(*entry, &img); e != Error::Success) return e;

    // The launch as argument sets: the whole launch, or one per chunk.
    std::vector<LaunchArgs> parts;
    if (!chunked) {
        parts.push_back(whole);
    } else {
        if (!entry->extent || !entry->marshal_chunk) {
            setDetail(std::format("kernel '{}' is not registered as chunkable", name));
            return Error::UnsupportedOperation;
        }
        if (workers == 0 || workers > 1024) {
            setDetail("workers must be in [1, 1024]");
            return Error::InvalidValue;
        }
        int64_t total = 0;
        if (Error e = entry->extent(typed_args, &total); e != Error::Success) return e;
        // Several chunks per worker: fast cores pick up the work slow (efficiency) cores would
        // otherwise finish last. Results don't depend on the split (chunks are disjoint).
        // Boundaries at multiples of 16 threads: whole 8-lane simd groups, and 64-byte lines
        // for one float per thread.
        const std::vector<ChunkSpec> chunks = planChunks(total, workers * r.chunks_per_worker, min_chunk, 16);
        if (!chunksCoverExactly(chunks, total)) {
            setDetail("internal: chunk plan does not cover the launch exactly");
            return Error::InternalError;
        }
        parts.resize(chunks.size());
        for (size_t i = 0; i < chunks.size(); ++i)
            if (Error e = entry->marshal_chunk(typed_args, chunks[i].begin, chunks[i].length, &parts[i]);
                e != Error::Success)
                return e;
    }

    if (r.capture_count.load() != 0) {  // recorded, not run
        std::lock_guard lk(r.capture_mutex);
        if (auto it = r.captures.find(target); it != r.captures.end()) {
            it->second->push_back(GraphNode{entry, name, std::move(parts)});
            return Error::Success;
        }
    }

    Stream::Task task;
    if (!chunked) {
        task = [entry, whole]() -> Error {
            Runtime& rt = R();
            if (Error e = skipIfPoisoned(rt); e != Error::Success) return e;
            const Error e = rt.backend->launch(*entry, whole);
            if (e == Error::KernelFault) recordFault(rt, rt.backend->lastFault());
            return e;
        };
    } else {
        WorkerPool* pool = ensurePool(r);
        task = [entry, per_chunk = std::move(parts), pool]() -> Error {
            if (Error e = skipIfPoisoned(R()); e != Error::Success) return e;
            std::vector<Stream::Task> jobs;
            jobs.reserve(per_chunk.size());
            for (const LaunchArgs& a : per_chunk) jobs.push_back(partTask(entry, &a));
            return pool->run(jobs);
        };
    }

    if (!stream) return target->runSync(std::move(task));
    target->enqueue(std::move(task));
    return Error::Success;
}

} // namespace

// ================================================================ lifecycle

Error pcudaInit() noexcept {
    return api("pcudaInit", [] {
        Runtime& r = R();
        std::unique_lock life(r.lifecycle);
        if (r.initialized) return Error::Success;
        if (Error e = checkPlatform(); e != Error::Success) return e;
        try {
            if (Error e = startDevice(r); e != Error::Success) {
                stopDevice(r);
                return e;
            }
        } catch (...) {
            stopDevice(r);
            throw;
        }
        r.initialized = true;
        return Error::Success;
    });
}

Error pcudaShutdown() noexcept {
    return api("pcudaShutdown", [] {
        Runtime& r = R();
        std::unique_lock life(r.lifecycle);
        if (!r.initialized) return Error::Success;
        const size_t leaked = stopDevice(r);
        {
            std::lock_guard lk(r.kernels_mutex);
            r.kernels.clear();
            r.builtins_registered = false;
        }
        r.initialized = false;
        if (leaked) setDetail(std::format("pcudaShutdown freed {} allocation(s) still live", leaked));
        return Error::Success;
    });
}

Error pcudaDeviceReset() noexcept {
    return api("pcudaDeviceReset", [] {
        Runtime& r = R();
        std::unique_lock life(r.lifecycle);
        if (Error e = requireInit(r); e != Error::Success) return e;
        stopDevice(r);
        if (Error e = startDevice(r); e != Error::Success) {
            stopDevice(r);
            r.initialized = false;
            return e;
        }
        return Error::Success;
    });
}

// ================================================================ device

Error pcudaGetDeviceCount(int* count) noexcept {
    return api("pcudaGetDeviceCount", [&] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireInit(r); e != Error::Success) return e;
        if (!count) return Error::InvalidValue;
        *count = 1;
        return Error::Success;
    });
}

Error pcudaGetDeviceProperties(DeviceProperties* props, int device) noexcept {
    return api("pcudaGetDeviceProperties", [&] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireInit(r); e != Error::Success) return e;
        if (!props) return Error::InvalidValue;
        if (device != 0) return Error::InvalidDevice;
        DeviceProperties p{};
        std::snprintf(p.name, sizeof p.name, "pcuda CPU device (Booth x86-64 backend, %s)", r.backend->name());
        p.worker_count = r.hw_threads;
        MEMORYSTATUSEX ms{};
        ms.dwLength = sizeof ms;
        if (GlobalMemoryStatusEx(&ms)) p.total_memory = static_cast<size_t>(ms.ullTotalPhys);
        p.is_hardware_gpu = false;
        p.guard_pages = r.guard_pages;
        p.large_pages = r.large_pages;
        p.kernel_stack_bytes = r.stack_bytes;
        *props = p;
        return Error::Success;
    });
}

Error pcudaSetDevice(int device) noexcept {
    return api("pcudaSetDevice", [&] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireInit(r); e != Error::Success) return e;
        if (device != 0) {
            setDetail(std::format("device {} does not exist (only device 0)", device));
            return Error::InvalidDevice;
        }
        r.current_device = 0;
        return Error::Success;
    });
}

Error pcudaGetDevice(int* device) noexcept {
    return api("pcudaGetDevice", [&] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireInit(r); e != Error::Success) return e;
        if (!device) return Error::InvalidValue;
        *device = r.current_device;
        return Error::Success;
    });
}

Error pcudaDeviceSynchronize() noexcept {
    return api("pcudaDeviceSynchronize", [] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireInit(r); e != Error::Success) return e;
        if (Error e = rejectDuringCapture(r, "pcudaDeviceSynchronize"); e != Error::Success) return e;
        Error first = r.default_stream->synchronize();
        std::string first_detail = detail::lastDetail();
        std::vector<Stream*> snap;
        {
            std::lock_guard lk(r.objects_mutex);
            for (auto& [h, s] : r.streams) snap.push_back(s->stream.get());
        }
        for (Stream* s : snap) {
            const Error e = s->synchronize();
            if (first == Error::Success && e != Error::Success) {
                first = e;
                first_detail = detail::lastDetail();
            }
        }
        if (first != Error::Success) {
            setDetail(first_detail);
            return first;
        }
        return requireUsable(r);
    });
}

// ================================================================ memory

Error pcudaMalloc(void** ptr, size_t bytes) noexcept {
    return api("pcudaMalloc", [&] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireUsable(r); e != Error::Success) return e;
        if (!ptr) {
            setDetail("ptr is null");
            return Error::InvalidValue;
        }
        *ptr = nullptr;
        if (bytes == 0) {
            setDetail("zero-byte allocations are rejected");
            return Error::InvalidValue;
        }
        return r.backend->allocate(ptr, bytes);
    });
}

Error pcudaFree(void* ptr) noexcept {
    return api("pcudaFree", [&] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireInit(r); e != Error::Success) return e;
        if (Error e = rejectDuringCapture(r, "pcudaFree"); e != Error::Success) return e;
        if (!ptr) return Error::Success;
        waitAllIdle(r);  // a queued kernel may still use this memory
        return r.backend->release(ptr);
    });
}

Error pcudaMemcpy(void* dst, const void* src, size_t bytes, CopyKind kind) noexcept {
    return api("pcudaMemcpy", [&] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireUsable(r); e != Error::Success) return e;
        if (Error e = rejectDuringCapture(r, "pcudaMemcpy"); e != Error::Success) return e;
        const int k = static_cast<int>(kind);
        if (k < 0 || k > 3) {
            setDetail("invalid CopyKind");
            return Error::InvalidValue;
        }
        if (bytes == 0) return Error::Success;
        if (!dst || !src) {
            setDetail("null pointer with non-zero byte count");
            return Error::InvalidValue;
        }
        waitAllIdle(r);
        return r.backend->copy(dst, src, bytes, kind);
    });
}

Error pcudaMemset(void* dst, int value, size_t bytes) noexcept {
    return api("pcudaMemset", [&] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireUsable(r); e != Error::Success) return e;
        if (Error e = rejectDuringCapture(r, "pcudaMemset"); e != Error::Success) return e;
        if (bytes == 0) return Error::Success;
        std::string why;
        if (Error e = r.registry.checkRange(dst, bytes, why); e != Error::Success) {
            setDetail(why);
            return e;
        }
        waitAllIdle(r);
        if (!cpu::protectedFill(dst, value, bytes)) return Error::InternalError;
        return Error::Success;
    });
}

// ================================================================ kernels

Error pcudaRegisterKernel(const KernelEntry* entry) noexcept {
    return api("pcudaRegisterKernel", [&] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireInit(r); e != Error::Success) return e;
        return registerEntry(r, entry);
    });
}

Error pcudaLaunchKernel(const char* name, const void* typed_args, pcudaStream_t stream) noexcept {
    return api("pcudaLaunchKernel", [&] { return launchCommon(name, typed_args, stream, false, 1, 1); });
}

Error pcudaLaunchKernelChunked(const char* name, const void* typed_args, unsigned workers, int64_t min_chunk,
                               pcudaStream_t stream) noexcept {
    return api("pcudaLaunchKernelChunked",
               [&] { return launchCommon(name, typed_args, stream, true, workers, min_chunk); });
}

// ================================================================ streams

Error pcudaStreamCreate(pcudaStream_t* stream) noexcept {
    return api("pcudaStreamCreate", [&] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireUsable(r); e != Error::Success) return e;
        if (!stream) return Error::InvalidValue;
        auto s = std::make_unique<pcudaStream_st>();
        s->stream = std::make_unique<Stream>(r.stack_bytes, L"pcuda stream", r.pool_spin_us);
        pcudaStream_t h = s.get();
        std::lock_guard lk(r.objects_mutex);
        r.streams.emplace(h, std::move(s));
        *stream = h;
        return Error::Success;
    });
}

Error pcudaStreamDestroy(pcudaStream_t stream) noexcept {
    return api("pcudaStreamDestroy", [&] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireInit(r); e != Error::Success) return e;
        std::unique_ptr<pcudaStream_st> owned;
        {
            std::lock_guard lk(r.objects_mutex);
            auto it = r.streams.find(stream);
            if (!stream || it == r.streams.end()) {
                setDetail("unknown stream handle");
                return Error::InvalidResourceHandle;
            }
            owned = std::move(it->second);
            r.streams.erase(it);
        }
        owned.reset();  // drains, then joins
        return Error::Success;
    });
}

Error pcudaStreamSynchronize(pcudaStream_t stream) noexcept {
    return api("pcudaStreamSynchronize", [&] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireInit(r); e != Error::Success) return e;
        Stream* s = nullptr;
        if (Error e = findStream(r, stream, &s); e != Error::Success) return e;
        if (isCapturing(r, s)) return rejectDuringCapture(r, "pcudaStreamSynchronize on a capturing stream");
        if (Error e = s->synchronize(); e != Error::Success) return e;
        return requireUsable(r);
    });
}

Error pcudaStreamQuery(pcudaStream_t stream) noexcept {
    return api("pcudaStreamQuery", [&] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireInit(r); e != Error::Success) return e;
        Stream* s = nullptr;
        if (Error e = findStream(r, stream, &s); e != Error::Success) return e;
        return s->idle() ? Error::Success : Error::NotReady;
    });
}

Error pcudaLaunchHostFunc(pcudaStream_t stream, void (*fn)(void*), void* user) noexcept {
    return api("pcudaLaunchHostFunc", [&] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireUsable(r); e != Error::Success) return e;
        if (!fn) return Error::InvalidValue;
        Stream* s = nullptr;
        if (Error e = findStream(r, stream, &s); e != Error::Success) return e;
        if (isCapturing(r, s)) return rejectDuringCapture(r, "pcudaLaunchHostFunc on a capturing stream");
        Stream::Task t = [fn, user]() -> Error {
            fn(user);
            return Error::Success;
        };
        if (!stream) return s->runSync(std::move(t));
        s->enqueue(std::move(t));
        return Error::Success;
    });
}

// ================================================================ graphs

Error pcudaStreamBeginCapture(pcudaStream_t stream) noexcept {
    return api("pcudaStreamBeginCapture", [&] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireUsable(r); e != Error::Success) return e;
        Stream* s = nullptr;
        if (Error e = findStream(r, stream, &s); e != Error::Success) return e;
        std::lock_guard lk(r.capture_mutex);
        if (r.captures.count(s)) {
            setDetail("the stream is already capturing");
            return Error::InvalidValue;
        }
        r.captures.emplace(s, std::make_shared<std::vector<GraphNode>>());
        r.capture_count.fetch_add(1);
        return Error::Success;
    });
}

Error pcudaStreamEndCapture(pcudaStream_t stream, pcudaGraph_t* graph) noexcept {
    return api("pcudaStreamEndCapture", [&] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireInit(r); e != Error::Success) return e;
        if (!graph) return Error::InvalidValue;
        *graph = nullptr;
        Stream* s = nullptr;
        if (Error e = findStream(r, stream, &s); e != Error::Success) return e;
        std::shared_ptr<std::vector<GraphNode>> nodes;
        {
            std::lock_guard lk(r.capture_mutex);
            auto it = r.captures.find(s);
            if (it == r.captures.end()) {
                setDetail("the stream is not capturing");
                return Error::InvalidValue;
            }
            nodes = std::move(it->second);
            r.captures.erase(it);
            r.capture_count.fetch_sub(1);
        }
        auto data = std::make_shared<GraphData>();
        data->nodes = std::move(*nodes);
        data->tasks.resize(data->nodes.size());
        data->steps.reserve(data->nodes.size());
        for (size_t k = 0; k < data->nodes.size(); ++k) {  // nodes and parts no longer move
            const GraphNode& n = data->nodes[k];
            for (const LaunchArgs& a : n.parts) data->tasks[k].push_back(partTask(n.entry, &a));
            data->steps.push_back({data->tasks[k].data(), data->tasks[k].size()});
        }
        auto g = std::make_unique<pcudaGraph_st>();
        g->data = std::move(data);
        pcudaGraph_t h = g.get();
        std::lock_guard lk(r.objects_mutex);
        r.graphs.emplace(h, std::move(g));
        *graph = h;
        return Error::Success;
    });
}

Error pcudaGraphLaunch(pcudaGraph_t graph, pcudaStream_t stream) noexcept {
    return api("pcudaGraphLaunch", [&] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireUsable(r); e != Error::Success) return e;
        std::shared_ptr<const GraphData> data;
        {
            std::lock_guard lk(r.objects_mutex);
            auto it = r.graphs.find(graph);
            if (!graph || it == r.graphs.end()) {
                setDetail("unknown graph handle");
                return Error::InvalidResourceHandle;
            }
            data = it->second->data;
        }
        Stream* s = nullptr;
        if (Error e = findStream(r, stream, &s); e != Error::Success) return e;
        if (isCapturing(r, s)) return rejectDuringCapture(r, "pcudaGraphLaunch on a capturing stream");
        if (data->nodes.empty()) return Error::Success;
        WorkerPool* pool = ensurePool(r);
        Stream::Task task = [data, pool]() -> Error {
            if (Error e = skipIfPoisoned(R()); e != Error::Success) return e;
            size_t failed = 0;
            const Error e = pool->runSteps(data->steps, &failed);
            if (e != Error::Success)
                setDetail(std::format("graph node {} ('{}'), {}", failed, data->nodes[failed].name,
                                      detail::lastDetail()));
            return e;
        };
        if (!stream) return s->runSync(std::move(task));
        s->enqueue(std::move(task));
        return Error::Success;
    });
}

Error pcudaGraphDestroy(pcudaGraph_t graph) noexcept {
    return api("pcudaGraphDestroy", [&] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireInit(r); e != Error::Success) return e;
        std::lock_guard lk(r.objects_mutex);
        auto it = r.graphs.find(graph);
        if (!graph || it == r.graphs.end()) {
            setDetail("unknown graph handle");
            return Error::InvalidResourceHandle;
        }
        r.graphs.erase(it);  // a queued launch keeps the graph's data alive
        return Error::Success;
    });
}

// ================================================================ events

Error pcudaEventCreate(pcudaEvent_t* event) noexcept {
    return api("pcudaEventCreate", [&] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireUsable(r); e != Error::Success) return e;
        if (!event) return Error::InvalidValue;
        auto ev = std::make_unique<pcudaEvent_st>();
        pcudaEvent_t h = ev.get();
        std::lock_guard lk(r.objects_mutex);
        r.events.emplace(h, std::move(ev));
        *event = h;
        return Error::Success;
    });
}

Error pcudaEventDestroy(pcudaEvent_t event) noexcept {
    return api("pcudaEventDestroy", [&] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireInit(r); e != Error::Success) return e;
        std::lock_guard lk(r.objects_mutex);
        auto it = r.events.find(event);
        if (!event || it == r.events.end()) {
            setDetail("unknown event handle");
            return Error::InvalidResourceHandle;
        }
        r.events.erase(it);  // pending record tasks keep the shared state alive
        return Error::Success;
    });
}

Error pcudaEventRecord(pcudaEvent_t event, pcudaStream_t stream) noexcept {
    return api("pcudaEventRecord", [&] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireUsable(r); e != Error::Success) return e;
        std::shared_ptr<EventState> st;
        if (Error e = findEvent(r, event, &st); e != Error::Success) return e;
        Stream* s = nullptr;
        if (Error e = findStream(r, stream, &s); e != Error::Success) return e;
        if (isCapturing(r, s)) return rejectDuringCapture(r, "pcudaEventRecord on a capturing stream");
        uint64_t gen;
        {
            std::lock_guard lk(st->mutex);
            gen = ++st->recorded;
        }
        s->enqueue([st, gen]() -> Error {
            {
                std::lock_guard lk(st->mutex);
                if (gen > st->completed) st->completed = gen;
            }
            st->cv.notify_all();
            return Error::Success;
        });
        return Error::Success;
    });
}

Error pcudaEventSynchronize(pcudaEvent_t event) noexcept {
    return api("pcudaEventSynchronize", [&] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireInit(r); e != Error::Success) return e;
        std::shared_ptr<EventState> st;
        if (Error e = findEvent(r, event, &st); e != Error::Success) return e;
        std::unique_lock lk(st->mutex);
        const uint64_t target = st->recorded;
        st->cv.wait(lk, [&] { return st->completed >= target; });
        return Error::Success;
    });
}

Error pcudaEventQuery(pcudaEvent_t event) noexcept {
    return api("pcudaEventQuery", [&] {
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireInit(r); e != Error::Success) return e;
        std::shared_ptr<EventState> st;
        if (Error e = findEvent(r, event, &st); e != Error::Success) return e;
        std::lock_guard lk(st->mutex);
        return st->completed >= st->recorded ? Error::Success : Error::NotReady;
    });
}

Error pcudaStreamWaitEvent(pcudaStream_t, pcudaEvent_t) noexcept {
    return api("pcudaStreamWaitEvent", [] {
        setDetail("cross-stream waits are not implemented in this POC");
        return Error::UnsupportedOperation;
    });
}

// ================================================================ code generation path

Error pcudaSetKernelCodegen(Codegen codegen) noexcept {
    return api("pcudaSetKernelCodegen", [&] {
        const int c = static_cast<int>(codegen);
        if (c < 0 || c > 3) {
            setDetail("codegen must be Auto, Booth, Native or Simd");
            return Error::InvalidValue;
        }
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        r.codegen.store(codegen);
        r.codegen_set.store(true);
        if (r.backend) r.backend->setCodegen(codegen);
        return Error::Success;
    });
}

Error pcudaGetKernelCodegen(Codegen* codegen) noexcept {
    return api("pcudaGetKernelCodegen", [&] {
        if (!codegen) return Error::InvalidValue;
        *codegen = R().codegen.load();
        return Error::Success;
    });
}

Error pcudaGetKernelCodegenFor(const char* name, Codegen* used) noexcept {
    return api("pcudaGetKernelCodegenFor", [&] {
        if (!used) return Error::InvalidValue;
        Runtime& r = R();
        std::shared_lock life(r.lifecycle);
        if (Error e = requireInit(r); e != Error::Success) return e;
        const KernelEntry* entry = nullptr;
        if (Error e = findKernel(r, name, &entry); e != Error::Success) return e;
        *used = r.backend->resolve(*entry);
        return Error::Success;
    });
}

const char* pcudaCodegenName(Codegen codegen) noexcept {
    switch (codegen) {
    case Codegen::Auto: return "auto";
    case Codegen::Booth: return "booth";
    case Codegen::Native: return "native";
    case Codegen::Simd: return "simd";
    }
    return "?";
}

// ================================================================ diagnostics

Error pcudaGetLastFault(FaultInfo* info) noexcept {
    return api("pcudaGetLastFault", [&] {
        if (!info) return Error::InvalidValue;
        Runtime& r = R();
        std::lock_guard lk(r.fault_mutex);
        if (!r.has_fault) return Error::NotReady;
        *info = r.last_fault;
        return Error::Success;
    });
}

const char* pcudaGetErrorString(Error e) noexcept {
    switch (e) {
    case Error::Success: return "Success";
    case Error::NotInitialized: return "NotInitialized";
    case Error::InvalidValue: return "InvalidValue";
    case Error::InvalidDevice: return "InvalidDevice";
    case Error::OutOfMemory: return "OutOfMemory";
    case Error::InvalidPointer: return "InvalidPointer";
    case Error::InvalidResourceHandle: return "InvalidResourceHandle";
    case Error::UnsupportedOperation: return "UnsupportedOperation";
    case Error::InvalidKernelImage: return "InvalidKernelImage";
    case Error::KernelNotFound: return "KernelNotFound";
    case Error::DynamicCodeBlocked: return "DynamicCodeBlocked";
    case Error::LaunchFailure: return "LaunchFailure";
    case Error::KernelFault: return "KernelFault";
    case Error::ContextPoisoned: return "ContextPoisoned";
    case Error::NotReady: return "NotReady";
    case Error::InternalError: return "InternalError";
    }
    return "UnknownError";
}

const char* pcudaGetLastErrorDetail() noexcept { return detail::lastDetail().c_str(); }
