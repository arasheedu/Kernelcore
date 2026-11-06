#pragma once

// FIFO execution queue with one dedicated worker thread (explicit stack size).
// Tasks run in submission order. Async task errors are kept (first error wins)
// until synchronize() returns them.

#include <pcuda/error.hpp>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace pcuda {

// Creates a thread with an explicit stack reservation (Windows default is only 1 MB).
// The thread sets a stack guarantee so the fault guard can run after a stack overflow.
void* startThread(size_t stack_bytes, const wchar_t* description, std::function<void()> body);
void  joinThread(void* handle) noexcept;

class Stream {
public:
    using Task = std::function<Error()>;

    // spin_us: the worker and a runSync caller spin this long before blocking (0 = never).
    Stream(size_t stack_bytes, const wchar_t* description, unsigned spin_us = 50);
    ~Stream();  // drains remaining work, then joins
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    void  enqueue(Task task);   // async; failures recorded for synchronize()
    Error runSync(Task task);   // ordered after earlier work; returns this task's result
    void  waitIdle();           // waits without consuming recorded errors
    Error synchronize();        // waitIdle + return and clear the first recorded error
    bool  idle();

private:
    struct SyncSlot {
        std::atomic<bool> done{false};
        Error       result = Error::Success;
        std::string detail;
    };
    struct Item {
        Task                      task;
        std::shared_ptr<SyncSlot> slot;  // null for async items
    };

    void loop();

    std::mutex              mutex_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;
    std::deque<Item>        queue_;
    uint64_t                submitted_ = 0;
    uint64_t                completed_ = 0;
    std::atomic<bool>       stopping_{false};
    std::atomic<size_t>     queued_{0};  // queue_.size(), readable without the lock
    unsigned                spin_us_;
    Error                   first_error_ = Error::Success;
    std::string             first_error_detail_;
    void*                   thread_ = nullptr;
};

// Fixed pool of worker threads for chunked launches.
//
// Latency: chunks are claimed with an atomic counter (no lock per chunk); one WakeByAddressAll
// per launch wakes sleeping workers; workers and the waiting caller spin for `spin_us` (with
// pause) before sleeping in WaitOnAddress. The calling thread also runs chunks of its launch.
class WorkerPool {
public:
    WorkerPool(unsigned threads, size_t stack_bytes, unsigned spin_us = 50);
    ~WorkerPool();
    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    unsigned size() const noexcept { return static_cast<unsigned>(threads_.size()); }
    // Runs all tasks (concurrently: pool threads plus the caller) and waits. Returns the error
    // of the lowest-index failing task, so results are deterministic.
    Error run(const std::vector<Stream::Task>& tasks);

    // One job of several steps (a graph: one step per kernel). The tasks of a step run
    // concurrently; a step starts when the previous one has finished (a barrier, no wake-up
    // per step). After a failing step no later step runs. Returns the first failure in
    // (step, task) order and its step index through failed_step.
    struct Step {
        const Stream::Task* tasks;
        size_t              count;
    };
    Error runSteps(std::span<const Step> steps, size_t* failed_step);

private:
    struct Group;
    void                   loop();
    void                   stop() noexcept;
    std::shared_ptr<Group> grab();  // a published launch with chunks left to claim
    void                   work(Group& g) const;

    std::mutex                          mutex_;   // guards active_ only
    std::vector<std::shared_ptr<Group>> active_;  // launches in progress (one per calling stream)
    std::atomic<uint32_t>               epoch_{0};  // bumped per launch; idle workers wait on it
    std::atomic<bool>                   stopping_{false};
    unsigned                            spin_us_;
    std::vector<void*>                  threads_;
};

// Chunk planning for independent 1-D kernels.
struct ChunkSpec {
    int64_t begin;
    int64_t length;
};
// At most `workers` chunks; each at least min_chunk (except a short tail) and a multiple
// of `align` elements except the tail.
std::vector<ChunkSpec> planChunks(int64_t total, unsigned workers, int64_t min_chunk, int64_t align = 64);
// True if chunks are non-empty, sorted, disjoint and exactly cover [0, total).
bool chunksCoverExactly(const std::vector<ChunkSpec>& chunks, int64_t total) noexcept;

} // namespace pcuda
