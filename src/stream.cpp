#include "stream.hpp"

#include "detail.hpp"

#include <windows.h>
#include <process.h>

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <format>
#include <stdexcept>

namespace pcuda {
namespace {

struct ThreadStart {
    std::function<void()> body;
    std::wstring          description;
};

unsigned __stdcall threadMain(void* p) {
    std::unique_ptr<ThreadStart> start(static_cast<ThreadStart*>(p));
    detail::markRuntimeThread();
    ULONG guarantee = 64 * 1024;  // room for the fault guard after a stack overflow
    SetThreadStackGuarantee(&guarantee);
    if (!start->description.empty()) SetThreadDescription(GetCurrentThread(), start->description.c_str());
    start->body();
    return 0;
}

bool cpuHasWaitpkg() {
    static const bool has = [] {
        int r[4] = {};
        __cpuidex(r, 7, 0);
        return (r[2] & (1 << 5)) != 0;  // ECX bit 5: UMONITOR / UMWAIT / TPAUSE
    }();
    return has;
}

// Waits until pred() or `us` microseconds pass; returns pred(). pred() must become true only
// through a write to the cache line at `watch`. With WAITPKG the thread sleeps in UMWAIT (light
// C0.1 state, woken by that write): no wake-up latency and much less power than spinning, which
// lets busy cores keep their turbo clocks. Otherwise it spins with `pause`.
template <class Pred>
bool spinFor(unsigned us, Pred pred, const volatile void* watch = nullptr) {
    if (us == 0) return pred();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(us);
    const bool umwait = watch && cpuHasWaitpkg();
    for (unsigned i = 0;; ++i) {
        if (pred()) return true;
        if (umwait) {
            _umonitor(const_cast<void*>(watch));
            if (pred()) return true;
            _umwait(1, __rdtsc() + 20000);  // C0.1; the OS also caps each wait
            if (std::chrono::steady_clock::now() >= deadline) return pred();
        } else {
            _mm_pause();
            if ((i & 63) == 63 && std::chrono::steady_clock::now() >= deadline) return pred();
        }
    }
}

Error runTask(const Stream::Task& t) noexcept {
    detail::clearDetail();
    try {
        return t();
    } catch (const std::bad_alloc&) {
        detail::setDetail("out of host memory in queued task");
        return Error::OutOfMemory;
    } catch (const std::exception& ex) {
        detail::setDetail(std::string("exception in queued task: ") + ex.what());
        return Error::InternalError;
    } catch (...) {
        detail::setDetail("unknown exception in queued task");
        return Error::InternalError;
    }
}

std::string detailFor(Error e) {
    return e == Error::Success ? std::string() : detail::lastDetail();
}

} // namespace

void* startThread(size_t stack_bytes, const wchar_t* description, std::function<void()> body) {
    if (stack_bytes > 0xFFFFFFFFu) throw std::invalid_argument("stack size too large");
    auto* start = new ThreadStart{std::move(body), description ? description : L""};
    const uintptr_t h = _beginthreadex(nullptr, static_cast<unsigned>(stack_bytes), &threadMain, start,
                                       STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
    if (h == 0) {
        delete start;
        throw std::runtime_error("_beginthreadex failed");
    }
    return reinterpret_cast<void*>(h);
}

void joinThread(void* handle) noexcept {
    if (!handle) return;
    WaitForSingleObject(static_cast<HANDLE>(handle), INFINITE);
    CloseHandle(static_cast<HANDLE>(handle));
}

// ---------------------------------------------------------------- Stream

Stream::Stream(size_t stack_bytes, const wchar_t* description, unsigned spin_us) : spin_us_(spin_us) {
    thread_ = startThread(stack_bytes, description, [this] { loop(); });
}

Stream::~Stream() {
    {
        std::lock_guard lk(mutex_);
        stopping_ = true;
    }
    work_cv_.notify_all();
    joinThread(thread_);
}

void Stream::loop() {
    for (;;) {
        // Spin briefly first: back-to-back launches then skip the thread wake-up.
        spinFor(spin_us_, [&] { return queued_.load(std::memory_order_acquire) != 0 || stopping_.load(); }, &queued_);
        Item item;
        {
            std::unique_lock lk(mutex_);
            work_cv_.wait(lk, [&] { return stopping_.load() || !queue_.empty(); });
            if (queue_.empty()) return;  // stopping and drained
            item = std::move(queue_.front());
            queue_.pop_front();
            queued_.store(queue_.size(), std::memory_order_release);
        }
        const Error e = runTask(item.task);
        std::string why = detailFor(e);
        {
            std::lock_guard lk(mutex_);
            if (item.slot) {
                item.slot->result = e;
                item.slot->detail = std::move(why);
                item.slot->done.store(true, std::memory_order_release);
            } else if (e != Error::Success && first_error_ == Error::Success) {
                first_error_ = e;
                first_error_detail_ = std::move(why);
            }
            ++completed_;
        }
        done_cv_.notify_all();
    }
}

void Stream::enqueue(Task task) {
    {
        std::lock_guard lk(mutex_);
        queue_.push_back(Item{std::move(task), nullptr});
        queued_.store(queue_.size(), std::memory_order_release);
        ++submitted_;
    }
    work_cv_.notify_one();
}

Error Stream::runSync(Task task) {
    auto slot = std::make_shared<SyncSlot>();
    {
        std::lock_guard lk(mutex_);
        queue_.push_back(Item{std::move(task), slot});
        queued_.store(queue_.size(), std::memory_order_release);
        ++submitted_;
    }
    work_cv_.notify_one();
    if (!spinFor(spin_us_, [&] { return slot->done.load(std::memory_order_acquire); }, &slot->done)) {
        std::unique_lock lk(mutex_);
        done_cv_.wait(lk, [&] { return slot->done.load(); });
    }
    std::lock_guard lk(mutex_);  // pairs with the worker's update of the slot
    if (slot->result != Error::Success) detail::setDetail(slot->detail);
    return slot->result;
}

void Stream::waitIdle() {
    std::unique_lock lk(mutex_);
    done_cv_.wait(lk, [&] { return completed_ == submitted_; });
}

Error Stream::synchronize() {
    std::unique_lock lk(mutex_);
    done_cv_.wait(lk, [&] { return completed_ == submitted_; });
    const Error e = first_error_;
    if (e != Error::Success) detail::setDetail(std::move(first_error_detail_));
    first_error_ = Error::Success;
    first_error_detail_.clear();
    return e;
}

bool Stream::idle() {
    std::lock_guard lk(mutex_);
    return completed_ == submitted_;
}

// ---------------------------------------------------------------- WorkerPool

// One job per caller (stream) at a time; several callers may share the pool. A job is a list
// of steps (a launch is one step; a graph is one step per kernel). Chunks of the current step
// are claimed with an atomic counter, so taking work needs no lock; the thread that finishes a
// step's last chunk opens the next step.
struct WorkerPool::Group {
    struct Span {
        const Stream::Task* tasks;
        uint32_t            count;  // > 0
        size_t              step;   // index in the caller's list
        uint32_t            threads;  // threads that may run it (slots below this); 0 = all
    };
    std::vector<Span>                        spans;
    std::unique_ptr<std::atomic<uint32_t>[]> next;       // per span: next unclaimed chunk
    std::unique_ptr<std::atomic<uint32_t>[]> remaining;  // per span: chunks not finished
    std::atomic<uint32_t>                    current{0}; // span being run; spans.size() = done
    std::atomic<bool>                        abort{false};
    uint32_t                                 widest = 0; // the widest step's thread cap (0 = all): higher slots never join
    std::atomic<uint32_t>                    gate{0};    // bumped when a step admitting more threads opens, or at the end

    std::mutex  err_mutex;  // first failure in (step, chunk) order, so results are deterministic
    size_t      err_span = SIZE_MAX, err_chunk = 0;
    Error       err = Error::Success;
    std::string err_detail;

    uint32_t size() const { return static_cast<uint32_t>(spans.size()); }
    bool admits(uint32_t s, uint32_t slot) const { return !spans[s].threads || slot < spans[s].threads; }
    void fail(uint32_t s, uint32_t i, Error e, std::string why) {
        std::lock_guard lk(err_mutex);
        if (s < err_span || (s == err_span && i < err_chunk)) {
            err_span = s, err_chunk = i, err = e;
            err_detail = std::move(why);
        }
        abort.store(true);
    }
};

WorkerPool::WorkerPool(unsigned threads, size_t stack_bytes, unsigned spin_us) : spin_us_(spin_us) {
    threads = std::max(1u, threads);
    try {
        for (unsigned i = 0; i < threads; ++i)
            threads_.push_back(startThread(stack_bytes, L"pcuda pool worker", [this, i] { loop(i + 1); }));
    } catch (...) {
        stop();
        throw;
    }
}

WorkerPool::~WorkerPool() { stop(); }

void WorkerPool::stop() noexcept {
    stopping_.store(true);
    epoch_.fetch_add(1, std::memory_order_acq_rel);
    WakeByAddressAll(&epoch_);
    for (void* t : threads_) joinThread(t);
    threads_.clear();
}

std::shared_ptr<WorkerPool::Group> WorkerPool::grab(uint32_t slot) {
    std::lock_guard lk(mutex_);
    for (const auto& g : active_)
        if (g->current.load(std::memory_order_acquire) < g->size() && (!g->widest || slot < g->widest)) return g;
    return nullptr;
}

// Runs chunks of g until the whole job is done (every thread that joins returns only then).
// The shared_ptr held by a pool worker keeps g alive while it signals, even after the caller
// has returned. Tasks are touched only while their step is unfinished, so they outlive use.
void WorkerPool::work(Group& g, uint32_t slot) const {
    const uint32_t n = g.size();
    for (;;) {
        const uint32_t s = g.current.load(std::memory_order_acquire);
        if (s >= n) return;
        if (!g.admits(s, slot)) {
            // Not one of this step's threads: sleep (no spinning, so the threads running it keep
            // the processors) until a step that admits this thread opens, or the job ends. Waiting
            // on the gate, not on current, avoids a wake-up per step. The caller has slot 0, so
            // every step runs.
            uint32_t t = s;
            while (t < n && !g.admits(t, slot)) ++t;
            for (;;) {
                uint32_t seen = g.gate.load(std::memory_order_acquire);
                if (g.current.load(std::memory_order_acquire) >= t) break;
                WaitOnAddress(&g.gate, &seen, sizeof seen, INFINITE);
            }
            continue;
        }
        const uint32_t i = g.next[s].fetch_add(1, std::memory_order_acq_rel);
        if (i < g.spans[s].count) {
            const Error e = runTask(g.spans[s].tasks[i]);
            if (e != Error::Success) g.fail(s, i, e, detailFor(e));
            if (g.remaining[s].fetch_sub(1, std::memory_order_acq_rel) == 1) {
                // Last chunk of the step: open the next one (or end the job after a failure).
                const uint32_t to = g.abort.load() ? n : s + 1;
                g.current.store(to, std::memory_order_release);
                WakeByAddressAll(&g.current);
                const uint32_t was = g.spans[s].threads, now = to < n ? g.spans[to].threads : 0;
                if (to == n || (was && (!now || now > was))) {  // threads sleeping on the gate may be admitted
                    g.gate.fetch_add(1, std::memory_order_acq_rel);
                    WakeByAddressAll(&g.gate);
                }
            }
            continue;
        }
        // Every chunk of step s is taken: wait until the threads running them finish.
        if (!spinFor(spin_us_, [&] { return g.current.load(std::memory_order_acquire) != s; }, &g.current)) {
            uint32_t seen = s;
            while (g.current.load(std::memory_order_acquire) == s)
                WaitOnAddress(&g.current, &seen, sizeof seen, INFINITE);
        }
    }
}

void WorkerPool::loop(uint32_t slot) {
    for (;;) {
        uint32_t seen = epoch_.load(std::memory_order_acquire);  // before looking for work
        if (auto g = grab(slot)) {
            work(*g, slot);
            continue;
        }
        if (stopping_.load()) return;
        if (!spinFor(spin_us_, [&] { return epoch_.load(std::memory_order_acquire) != seen; }, &epoch_))
            WaitOnAddress(&epoch_, &seen, sizeof seen, INFINITE);  // returns at once if epoch_ moved
    }
}

Error WorkerPool::run(const std::vector<Stream::Task>& tasks, unsigned threads) {
    if (tasks.empty()) return Error::Success;
    if (tasks.size() == 1) {  // nothing to share: run inline, wake nobody
        const Error e = runTask(tasks[0]);
        if (e != Error::Success) detail::setDetail(std::format("chunk 0: {}", detailFor(e)));
        return e;
    }
    const Step one{tasks.data(), tasks.size(), threads};
    return runSteps(std::span(&one, 1), nullptr);
}

Error WorkerPool::runSteps(std::span<const Step> steps, size_t* failed_step) {
    auto gp = std::make_shared<Group>();
    Group& g = *gp;
    for (size_t k = 0; k < steps.size(); ++k)
        if (steps[k].count) g.spans.push_back({steps[k].tasks, static_cast<uint32_t>(steps[k].count), k, steps[k].threads});
    if (g.spans.empty()) return Error::Success;
    for (const auto& sp : g.spans) {
        if (!sp.threads) {
            g.widest = 0;
            break;
        }
        g.widest = std::max(g.widest, sp.threads);
    }
    g.next = std::make_unique<std::atomic<uint32_t>[]>(g.spans.size());
    g.remaining = std::make_unique<std::atomic<uint32_t>[]>(g.spans.size());
    for (size_t k = 0; k < g.spans.size(); ++k) {
        g.next[k].store(0, std::memory_order_relaxed);
        g.remaining[k].store(g.spans[k].count, std::memory_order_relaxed);
    }
    {
        std::lock_guard lk(mutex_);
        active_.push_back(gp);
    }
    epoch_.fetch_add(1, std::memory_order_acq_rel);
    WakeByAddressAll(&epoch_);  // one call wakes every sleeping worker
    work(g, 0);                 // the caller runs chunks too; returns when the job is done
    {
        std::lock_guard lk(mutex_);
        active_.erase(std::find(active_.begin(), active_.end(), gp));
    }
    std::lock_guard lk(g.err_mutex);
    if (g.err == Error::Success) return Error::Success;
    if (failed_step) *failed_step = g.spans[g.err_span].step;
    detail::setDetail(std::format("chunk {}: {}", g.err_chunk, g.err_detail));
    return g.err;
}

// ---------------------------------------------------------------- chunk planning

std::vector<ChunkSpec> planChunks(int64_t total, unsigned workers, int64_t min_chunk, int64_t align) {
    std::vector<ChunkSpec> out;
    if (total <= 0) return out;
    if (workers == 0) workers = 1;
    if (min_chunk < 1) min_chunk = 1;
    if (align < 1) align = 1;
    int64_t per = (total + workers - 1) / workers;
    per = std::max(per, min_chunk);
    per = (per + align - 1) / align * align;
    for (int64_t b = 0; b < total; b += per) out.push_back({b, std::min(per, total - b)});
    return out;
}

bool chunksCoverExactly(const std::vector<ChunkSpec>& chunks, int64_t total) noexcept {
    int64_t next = 0;
    for (const ChunkSpec& c : chunks) {
        if (c.length <= 0 || c.begin != next) return false;
        next = c.begin + c.length;
    }
    return next == total;
}

} // namespace pcuda
