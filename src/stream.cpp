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

Stream::Stream(size_t stack_bytes, const wchar_t* description) {
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
        Item item;
        {
            std::unique_lock lk(mutex_);
            work_cv_.wait(lk, [&] { return stopping_ || !queue_.empty(); });
            if (queue_.empty()) return;  // stopping and drained
            item = std::move(queue_.front());
            queue_.pop_front();
        }
        const Error e = runTask(item.task);
        std::string why = detailFor(e);
        {
            std::lock_guard lk(mutex_);
            if (item.slot) {
                item.slot->result = e;
                item.slot->detail = std::move(why);
                item.slot->done = true;
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
        ++submitted_;
    }
    work_cv_.notify_one();
}

Error Stream::runSync(Task task) {
    auto slot = std::make_shared<SyncSlot>();
    {
        std::lock_guard lk(mutex_);
        queue_.push_back(Item{std::move(task), slot});
        ++submitted_;
    }
    work_cv_.notify_one();
    std::unique_lock lk(mutex_);
    done_cv_.wait(lk, [&] { return slot->done; });
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

struct WorkerPool::Group {
    std::mutex               mutex;
    std::condition_variable  cv;
    std::atomic<size_t>      remaining{0};
    std::vector<Error>       results;
    std::vector<std::string> details;
};

namespace {

// Spins with `pause` until pred() or `us` microseconds pass. Returns pred().
template <class Pred>
bool spinFor(unsigned us, Pred pred) {
    if (us == 0) return pred();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(us);
    for (unsigned i = 0;; ++i) {
        if (pred()) return true;
        _mm_pause();
        if ((i & 63) == 63 && std::chrono::steady_clock::now() >= deadline) return pred();
    }
}

} // namespace

WorkerPool::WorkerPool(unsigned threads, size_t stack_bytes, unsigned spin_us) : spin_us_(spin_us) {
    threads = std::max(1u, threads);
    try {
        for (unsigned i = 0; i < threads; ++i)
            threads_.push_back(startThread(stack_bytes, L"pcuda pool worker", [this] { loop(); }));
    } catch (...) {
        {
            std::lock_guard lk(mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
        for (void* t : threads_) joinThread(t);
        throw;
    }
}

WorkerPool::~WorkerPool() {
    {
        std::lock_guard lk(mutex_);
        stopping_ = true;
    }
    cv_.notify_all();
    for (void* t : threads_) joinThread(t);
}

bool WorkerPool::takeJob(Job& out, const Group* only) {
    for (auto it = jobs_.begin(); it != jobs_.end(); ++it) {
        if (only && it->group.get() != only) continue;
        out = *it;
        jobs_.erase(it);
        pending_.store(jobs_.size(), std::memory_order_release);
        return true;
    }
    return false;
}

void WorkerPool::execute(const Job& job) {
    const Error e = runTask(*job.task);
    std::string why = detailFor(e);
    // The job owns a reference to the group, so it stays alive while this thread signals it,
    // even if the caller has already seen remaining == 0 and returned.
    std::lock_guard lk(job.group->mutex);
    job.group->results[job.index] = e;
    job.group->details[job.index] = std::move(why);
    job.group->remaining.fetch_sub(1, std::memory_order_acq_rel);
    job.group->cv.notify_all();
}

void WorkerPool::loop() {
    for (;;) {
        spinFor(spin_us_, [&] { return pending_.load(std::memory_order_acquire) != 0 || stopping_.load(); });
        Job job;
        {
            std::unique_lock lk(mutex_);
            cv_.wait(lk, [&] { return stopping_.load() || !jobs_.empty(); });
            if (!takeJob(job, nullptr)) return;  // stopping and drained
        }
        execute(job);
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
    g.remaining.store(tasks.size());
    g.results.assign(tasks.size(), Error::Success);
    g.details.assign(tasks.size(), std::string());
    {
        std::lock_guard lk(mutex_);
        for (size_t i = 0; i < tasks.size(); ++i) jobs_.push_back(Job{&tasks[i], gp, i});
        pending_.store(jobs_.size(), std::memory_order_release);
    }
    // The caller takes one chunk itself; wake only as many sleepers as there are other chunks.
    const size_t wake = std::min(tasks.size() - 1, threads_.size());
    for (size_t i = 0; i < wake; ++i) cv_.notify_one();
    // The caller works on its own launch too, then waits for chunks still running elsewhere.
    for (;;) {
        Job job;
        {
            std::lock_guard lk(mutex_);
            if (!takeJob(job, &g)) break;
        }
        execute(job);
    }
    if (!spinFor(spin_us_, [&] { return g.remaining.load(std::memory_order_acquire) == 0; })) {
        std::unique_lock lk(g.mutex);
        g.cv.wait(lk, [&] { return g.remaining.load(std::memory_order_acquire) == 0; });
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
