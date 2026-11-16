#pragma once

#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace bgpgen {

/// Returns the worker count parallel_for would use for (n, grain, threads).
inline size_t parallel_workers(size_t n, size_t grain, unsigned threads) {
  if (n == 0) return 0;
  if (threads == 0) threads = std::thread::hardware_concurrency();
  size_t chunks = (n + grain - 1) / grain;
  return std::min<size_t>(std::max(1u, threads), chunks);
}

/// Runs fn(begin, end, worker) over chunks of [0, n) on `threads` threads
/// (0 = all hardware threads); `worker` indexes [0, parallel_workers(...))
/// so callers can keep per-thread state without synchronization. Chunks are
/// claimed from a shared atomic counter, so uneven work self-balances.
/// Blocks until all chunks finish; the first exception thrown by any chunk
/// is rethrown here.
///
/// Threads are spawned per call — we run a handful of large parallel
/// sections per run, so pool persistence would buy nothing.
template <class Fn>
void parallel_for(size_t n, size_t grain, unsigned threads, Fn&& fn) {
  size_t workers = parallel_workers(n, grain, threads);
  if (workers == 0) return;

  if (workers == 1) {
    fn(size_t{0}, n, size_t{0});
    return;
  }

  size_t chunks = (n + grain - 1) / grain;
  std::atomic<size_t> next_chunk{0};
  std::exception_ptr error;
  std::mutex error_mutex;

  auto work = [&](size_t worker) {
    try {
      while (true) {
        size_t chunk = next_chunk.fetch_add(1, std::memory_order_relaxed);
        if (chunk >= chunks) return;
        size_t begin = chunk * grain;
        fn(begin, std::min(begin + grain, n), worker);
      }
    } catch (...) {
      std::lock_guard lock(error_mutex);
      if (!error) error = std::current_exception();
    }
  };

  {
    std::vector<std::jthread> pool;
    pool.reserve(workers);
    for (size_t i = 0; i < workers; ++i) pool.emplace_back(work, i);
  }
  if (error) std::rethrow_exception(error);
}

} // namespace bgpgen
