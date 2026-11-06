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

// One launch at a time per caller; several callers (streams) may share the pool.
// Chunks are claimed with an atomic counter, so taking work needs no lock.
struct WorkerPool::Group {
    std::vector<Stream::Task>* tasks = nullptr;
    uint32_t                   count = 0;
    std::atomic<uint32_t>      next{0};       // next unclaimed chunk
    std::atomic<uint32_t>      remaining{0};  // chunks not finished; the caller waits for 0
    std::vector<Error>         results;       // one slot per chunk, written by its runner only
    std::vector<std::string>   details;
};

WorkerPool::WorkerPool(unsigned threads, size_t stack_bytes, unsigned spin_us) : spin_us_(spin_us) {
    threads = std::max(1u, threads);
    try {
        for (unsigned i = 0; i < threads; ++i)
            threads_.push_back(startThread(stack_bytes, L"pcuda pool worker", [this] { loop(); }));
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

std::shared_ptr<WorkerPool::Group> WorkerPool::grab() {
    std::lock_guard lk(mutex_);
    for (const auto& g : active_)
        if (g->next.load(std::memory_order_relaxed) < g->count) return g;
    return nullptr;
}

// Runs chunks of g until none are left to claim. The shared_ptr keeps g alive while this thread
// signals completion, even after the caller has seen remaining == 0 and returned.
void WorkerPool::work(Group& g) {
    for (;;) {
        const uint32_t i = g.next.fetch_add(1, std::memory_order_acq_rel);
        if (i >= g.count) return;
        const Error e = runTask((*g.tasks)[i]);
        if (e != Error::Success) {
            g.results[i] = e;
            g.details[i] = detailFor(e);
        }
        if (g.remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) WakeByAddressAll(&g.remaining);
    }
}

void WorkerPool::loop() {
    for (;;) {
        uint32_t seen = epoch_.load(std::memory_order_acquire);  // before looking for work
        if (auto g = grab()) {
            work(*g);
            continue;
        }
        if (stopping_.load()) return;
        if (!spinFor(spin_us_, [&] { return epoch_.load(std::memory_order_acquire) != seen; }, &epoch_))
            WaitOnAddress(&epoch_, &seen, sizeof seen, INFINITE);  // returns at once if epoch_ moved
    }
}

Error WorkerPool::run(std::vector<Stream::Task>& tasks) {
    if (tasks.empty()) return Error::Success;
    if (tasks.size() == 1) {  // nothing to share: run inline, wake nobody
        const Error e = runTask(tasks[0]);
        if (e != Error::Success) detail::setDetail(std::format("chunk 0: {}", detailFor(e)));
        return e;
    }
    auto gp = std::make_shared<Group>();
    Group& g = *gp;
    g.tasks = &tasks;
    g.count = static_cast<uint32_t>(tasks.size());
    g.remaining.store(g.count);
    g.results.assign(tasks.size(), Error::Success);
    g.details.assign(tasks.size(), std::string());
    {
        std::lock_guard lk(mutex_);
        active_.push_back(gp);
    }
    epoch_.fetch_add(1, std::memory_order_acq_rel);
    WakeByAddressAll(&epoch_);  // one call wakes every sleeping worker
    work(g);                    // the caller runs chunks of its own launch too
    for (;;) {
        uint32_t left = g.remaining.load(std::memory_order_acquire);
        if (left == 0) break;
        if (spinFor(spin_us_, [&] { return g.remaining.load(std::memory_order_acquire) == 0; }, &g.remaining)) break;
        left = g.remaining.load(std::memory_order_acquire);
        if (left != 0) WaitOnAddress(&g.remaining, &left, sizeof left, INFINITE);
    }
    {
        std::lock_guard lk(mutex_);
        active_.erase(std::find(active_.begin(), active_.end(), gp));
    }
    for (size_t i = 0; i < g.results.size(); ++i) {
        if (g.results[i] != Error::Success) {
            detail::setDetail(std::format("chunk {}: {}", i, g.details[i]));
            return g.results[i];
        }
    }
    return Error::Success;
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
