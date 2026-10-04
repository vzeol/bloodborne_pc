// bbport: host threads that may call guest code (AvPlayer allocator callbacks).
// Each thread gets a guest TCB (GS base, TLS) from the C runtime before running.
#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <stop_token>
#include <system_error>
#include <thread>
#include "common/types.h"

extern "C" void runtime_thread_attach_host(const char* name);

namespace Libraries::Kernel {
class Thread {
public:
    Thread() = default;
    ~Thread() { Stop(); }
    void Run(std::function<void(std::stop_token)>&& func) {
        std::scoped_lock lock{join_mutex};
        thread = std::jthread([func = std::move(func)](std::stop_token stop) {
            runtime_thread_attach_host("bb:hle");
            func(stop);
        });
        owner.store(thread.get_id(), std::memory_order_release);
        running.store(true, std::memory_order_release);
    }
    // A thread may stop its own Thread object (AvPlayer does); it detaches instead of joining.
    // bbport: two threads may also stop the same object at once (AvPlayer: the game's Stop and
    // the demuxer at the end of the stream). The second one waits for the first and finds the
    // thread gone; a join race would throw std::system_error into guest frames, which cannot
    // be unwound (the process ends).
    void Join() {
        // The thread itself never waits for the lock: whoever holds it is joining this thread
        // (AvPlayer: the game's Stop), and waiting would deadlock. Uncontended, it detaches.
        if (owner.load(std::memory_order_acquire) == std::this_thread::get_id()) {
            std::unique_lock lock{join_mutex, std::try_to_lock};
            if (lock.owns_lock() && thread.joinable() && thread.get_id() == std::this_thread::get_id()) {
                thread.detach();
                running.store(false, std::memory_order_release);
            }
            return;
        }
        std::scoped_lock lock{join_mutex};
        if (!thread.joinable()) {
            return;
        }
        try {
            if (thread.get_id() == std::this_thread::get_id()) {
                thread.detach();
            } else {
                thread.join();
            }
        } catch (const std::system_error&) {
        }
        running.store(false, std::memory_order_release);
    }
    /// Lock-free: callers may hold locks the joined thread needs.
    bool Joinable() const {
        return running.load(std::memory_order_acquire);
    }
    void Stop() {
        if (Joinable()) {
            thread.get_stop_source().request_stop();
            Join();
        }
    }

private:
    std::jthread thread;
    std::mutex join_mutex;
    std::atomic<bool> running{false};
    std::atomic<std::thread::id> owner{};
};
} // namespace Libraries::Kernel
