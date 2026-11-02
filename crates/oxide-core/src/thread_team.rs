//! Persistent Core-Pinned Spin-Barrier Thread Team for zero-overhead CPU execution.
//! Replaces dynamic Rayon fork-join scheduler in latency-critical LLM loops.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};

/// Trait implemented by parallel workloads dispatched to `ThreadTeam`.
pub trait ParallelTask: Sync {
    /// Execute a chunk of the task for the given thread index.
    fn execute_chunk(&self, thread_idx: usize, num_threads: usize);
}

/// Dynamic adapter for closures.
struct FnTask<F>(F)
where
    F: Fn(usize, usize) + Sync;

impl<F> ParallelTask for FnTask<F>
where
    F: Fn(usize, usize) + Sync,
{
    #[inline(always)]
    fn execute_chunk(&self, thread_idx: usize, num_threads: usize) {
        (self.0)(thread_idx, num_threads);
    }
}

unsafe fn trampoline<T: ParallelTask>(data: *const (), thread_idx: usize, num_threads: usize) {
    // SAFETY: caller guarantees data pointer points to a valid instance of T
    // which remains alive until all workers report completion.
    unsafe {
        let task = &*data.cast::<T>();
        task.execute_chunk(thread_idx, num_threads);
    }
}

/// Internal shared control block for the thread team.
#[repr(align(64))]
#[derive(Debug)]
struct TeamShared {
    epoch: AtomicUsize,
    done_count: AtomicUsize,
    shutdown: AtomicBool,
    data_ptr: AtomicUsize,
    work_fn: AtomicUsize,
    asleep: AtomicU32,
}

/// A zero-allocation, core-pinned thread pool for high-performance CPU inference.
#[derive(Debug)]
pub struct ThreadTeam {
    num_threads: usize,
    shared: Arc<TeamShared>,
    workers: Vec<JoinHandle<()>>,
    dispatch_lock: std::sync::Mutex<()>,
}

const SPIN_LIMIT: usize = 200_000;

impl ThreadTeam {
    /// Create a new thread team with `num_threads` workers pinned to physical hardware cores.
    #[must_use]
    pub fn new_pinned(num_threads: usize) -> Self {
        let num_threads = num_threads.max(1);
        let shared = Arc::new(TeamShared {
            epoch: AtomicUsize::new(0),
            done_count: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            data_ptr: AtomicUsize::new(0),
            work_fn: AtomicUsize::new(0),
            asleep: AtomicU32::new(0),
        });

        let core_ids = core_affinity::get_core_ids().unwrap_or_default();
        let mut workers = Vec::with_capacity(num_threads.saturating_sub(1));

        // Pin the calling thread (thread 0 / master) to core 0 if possible
        if !core_ids.is_empty() {
            let _ = core_affinity::set_for_current(core_ids[0]);
        }

        // Spawn worker threads 1..num_threads
        for thread_idx in 1..num_threads {
            let shared_clone = Arc::clone(&shared);
            let core = if core_ids.is_empty() {
                None
            } else {
                Some(core_ids[thread_idx % core_ids.len()])
            };

            let handle = thread::Builder::new()
                .name(format!("oxide-cpu-worker-{thread_idx}"))
                .spawn(move || {
                    if let Some(c) = core {
                        let _ = core_affinity::set_for_current(c);
                    }

                    let mut local_epoch = 0;
                    let worker_mask = 1u32 << (thread_idx - 1);
                    loop {
                        let mut spins = 0;
                        while shared_clone.epoch.load(Ordering::Acquire) == local_epoch {
                            if shared_clone.shutdown.load(Ordering::Relaxed) {
                                return;
                            }
                            core::hint::spin_loop();
                            spins += 1;
                            if spins >= SPIN_LIMIT {
                                shared_clone.asleep.fetch_or(worker_mask, Ordering::SeqCst);
                                if shared_clone.epoch.load(Ordering::Acquire) != local_epoch {
                                    shared_clone.asleep.fetch_and(!worker_mask, Ordering::SeqCst);
                                    break;
                                }
                                thread::park();
                                shared_clone.asleep.fetch_and(!worker_mask, Ordering::SeqCst);
                                spins = 0;
                            }
                        }

                        if shared_clone.shutdown.load(Ordering::Relaxed) {
                            return;
                        }

                        local_epoch = shared_clone.epoch.load(Ordering::Acquire);
                        let fn_addr = shared_clone.work_fn.load(Ordering::Acquire);
                        let data_addr = shared_clone.data_ptr.load(Ordering::Acquire);
                        if fn_addr != 0 {
                            // SAFETY: Address transmuted to function pointer adhering to trampoline signature.
                            let func: unsafe fn(*const (), usize, usize) =
                                unsafe { std::mem::transmute(fn_addr) };
                            // SAFETY: Pointer is valid on stack of master until dispatch returns.
                            unsafe {
                                func(data_addr as *const (), thread_idx, num_threads);
                            }
                        }

                        shared_clone.done_count.fetch_add(1, Ordering::Release);
                    }
                })
                .expect("Failed to spawn oxide-cpu-worker");

            workers.push(handle);
        }

        Self {
            num_threads,
            shared,
            workers,
            dispatch_lock: std::sync::Mutex::new(()),
        }
    }

    /// Number of threads in the team.
    #[must_use]
    #[inline(always)]
    pub const fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Dispatches a `ParallelTask` across all threads in the team.
    /// Thread 0 (caller) executes chunk 0 synchronously while background workers
    /// execute chunks 1..num_threads in parallel.
    #[inline(always)]
    pub fn dispatch<T: ParallelTask>(&self, task: &T) {
        if self.num_threads <= 1 {
            task.execute_chunk(0, 1);
            return;
        }

        let _guard = self
            .dispatch_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let worker_count = self.num_threads - 1;
        self.shared.done_count.store(0, Ordering::Relaxed);

        let data_ptr = std::ptr::from_ref::<T>(task).cast::<()>() as usize;
        let fn_ptr = trampoline::<T> as *const () as usize;

        self.shared.data_ptr.store(data_ptr, Ordering::Relaxed);
        self.shared.work_fn.store(fn_ptr, Ordering::Release);

        // Signal workers
        self.shared.epoch.fetch_add(1, Ordering::Release);

        // Only unpark workers if any were actually parked
        let sleeping = self.shared.asleep.load(Ordering::Acquire);
        if sleeping != 0 {
            for (i, worker) in self.workers.iter().enumerate() {
                if (sleeping & (1u32 << i)) != 0 {
                    worker.thread().unpark();
                }
            }
        }

        // Master thread executes chunk 0
        task.execute_chunk(0, self.num_threads);

        // Wait for all workers to finish their chunk
        while self.shared.done_count.load(Ordering::Acquire) < worker_count {
            core::hint::spin_loop();
        }

        self.shared.work_fn.store(0, Ordering::Relaxed);
        self.shared.data_ptr.store(0, Ordering::Relaxed);
    }

    /// Dispatches a closure across all threads.
    #[inline(always)]
    pub fn dispatch_fn<F>(&self, f: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        self.dispatch(&FnTask(f));
    }

    /// Dispatches a row-parallel task over `total_rows`.
    /// Each thread gets a contiguous chunk of rows.
    #[inline(always)]
    pub fn dispatch_rows<F>(&self, total_rows: usize, f: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        if total_rows == 0 {
            return;
        }
        self.dispatch_fn(|thread_idx, num_threads| {
            let chunk_size = total_rows.div_ceil(num_threads);
            let start = (thread_idx * chunk_size).min(total_rows);
            let end = ((thread_idx + 1) * chunk_size).min(total_rows);
            if start < end {
                f(start, end);
            }
        });
    }
}

impl Drop for ThreadTeam {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.epoch.fetch_add(1, Ordering::Release);
        for worker in &self.workers {
            worker.thread().unpark();
        }
        while let Some(worker) = self.workers.pop() {
            let _ = worker.join();
        }
    }
}

// Global thread team instance management
static GLOBAL_THREAD_TEAM: std::sync::OnceLock<ThreadTeam> = std::sync::OnceLock::new();

/// Initialize or retrieve the global persistent thread team pinned to CPU cores.
#[must_use]
pub fn global_thread_team(threads: usize) -> &'static ThreadTeam {
    GLOBAL_THREAD_TEAM.get_or_init(|| ThreadTeam::new_pinned(threads))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn test_thread_team_dispatch_rows() {
        let team = ThreadTeam::new_pinned(4);
        let n = 1000;
        let values: Vec<u64> = (0..n as u64).collect();
        let sum = AtomicU64::new(0);

        team.dispatch_rows(n, |start, end| {
            let mut local_sum = 0;
            for &v in &values[start..end] {
                local_sum += v;
            }
            sum.fetch_add(local_sum, Ordering::Relaxed);
        });

        let expected: u64 = (0..n as u64).sum();
        assert_eq!(sum.load(Ordering::Relaxed), expected);
    }
}
