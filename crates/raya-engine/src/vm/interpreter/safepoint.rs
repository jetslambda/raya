//! Safepoint Infrastructure for STW Pauses
//!
//! Epoch-based cooperative safepoint coordination for stop-the-world (STW)
//! operations like garbage collection, VM snapshotting, and debugging
//! (task G4).
//!
//! ## Protocol
//!
//! 1. A coordinator thread calls [`SafepointCoordinator::request_stw_pause`]
//!    with a reason. This sets the fast-path flag and blocks until every
//!    registered worker has arrived at a safepoint.
//! 2. Workers call [`SafepointCoordinator::poll`] at allocation sites,
//!    call boundaries, loop back-edges, and task operations. The fast path
//!    is a single atomic load; the slow path arrives at the current epoch
//!    and waits until that epoch is released.
//! 3. When all workers have arrived, `request_stw_pause` returns — the
//!    world is stopped and the caller may inspect or collect.
//! 4. [`SafepointCoordinator::resume_from_pause`] bumps the epoch, clears
//!    the flag, and releases all workers. Workers leave only after this
//!    bump, so the stopped interval is exactly bounded by request/resume.
//!
//! ## Guarantees
//!
//! - All workers reach a safepoint within one loop iteration, one call, or
//!   one allocation of the request.
//! - Workers cannot resume before `resume_from_pause` (the previous
//!   barrier design released them as soon as the count matched, racing
//!   the external operation).
//! - Worker registration is dynamic: registering and deregistering adjust
//!   the arrival target under the same lock, so neither the coordinator
//!   nor workers deadlock on count changes.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Reasons for requesting a safepoint pause
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// Garbage collection
    GarbageCollection,
    /// VM state snapshotting
    Snapshot,
    /// Debugger breakpoint
    Debug,
}

/// Statistics tracking for safepoint operations
#[derive(Debug, Default)]
pub struct SafepointStats {
    /// Total number of safepoints executed
    total_safepoints: AtomicUsize,
    /// Total time spent at safepoints (microseconds)
    total_pause_time_us: AtomicUsize,
    /// Maximum pause time (microseconds)
    max_pause_time_us: AtomicUsize,
}

impl SafepointStats {
    fn reset(&self) {
        self.total_safepoints.store(0, Ordering::Relaxed);
        self.total_pause_time_us.store(0, Ordering::Relaxed);
        self.max_pause_time_us.store(0, Ordering::Relaxed);
    }

    fn total_safepoints(&self) -> usize {
        self.total_safepoints.load(Ordering::Relaxed)
    }

    fn total_pause_time_us(&self) -> usize {
        self.total_pause_time_us.load(Ordering::Relaxed)
    }

    fn max_pause_time_us(&self) -> usize {
        self.max_pause_time_us.load(Ordering::Relaxed)
    }
}

#[derive(Debug)]
struct CoordinatorState {
    worker_count: usize,
    arrived: usize,
    reason: Option<StopReason>,
}

/// Coordinates stop-the-world pauses across all worker threads
pub struct SafepointCoordinator {
    /// Fast-path flag: a pause is pending (any reason)
    pending: AtomicBool,

    /// Completed stop-the-world epochs. Workers wait until this advances
    /// past the epoch they arrived at.
    epoch: AtomicU64,

    state: Mutex<CoordinatorState>,

    cv: Condvar,

    /// Statistics
    pub stats: SafepointStats,
}

impl SafepointCoordinator {
    /// Create a new SafepointCoordinator with the specified number of workers
    pub fn new(worker_count: usize) -> Self {
        Self {
            pending: AtomicBool::new(false),
            epoch: AtomicU64::new(0),
            state: Mutex::new(CoordinatorState {
                worker_count,
                arrived: 0,
                reason: None,
            }),
            cv: Condvar::new(),
            stats: SafepointStats::default(),
        }
    }

    /// Fast inline check - called frequently from interpreter
    #[inline(always)]
    pub fn poll(&self) {
        if self.is_pause_pending_fast() {
            self.enter_safepoint();
        }
    }

    #[inline(always)]
    fn is_pause_pending_fast(&self) -> bool {
        self.pending.load(Ordering::Acquire)
    }

    /// Slow path: arrive at the current epoch and wait for release.
    #[cold]
    #[inline(never)]
    fn enter_safepoint(&self) {
        let start = std::time::Instant::now();

        {
            let mut st = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            st.arrived += 1;
            // Wake the coordinator: arrival count changed.
            self.cv.notify_all();

            // Hold the position until the epoch is released.
            while st.reason.is_some() {
                st = self.cv.wait(st).unwrap_or_else(|poisoned| poisoned.into_inner());
            }
            st.arrived -= 1;
        }

        let elapsed = start.elapsed().as_micros() as usize;
        self.stats
            .total_pause_time_us
            .fetch_add(elapsed, Ordering::Relaxed);
        self.stats.total_safepoints.fetch_add(1, Ordering::Relaxed);

        let mut max = self.stats.max_pause_time_us.load(Ordering::Relaxed);
        while elapsed > max {
            match self.stats.max_pause_time_us.compare_exchange_weak(
                max,
                elapsed,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(current) => max = current,
            }
        }
    }

    /// Request a stop-the-world pause. Blocks until every registered worker
    /// has arrived at a safepoint. The world remains stopped until
    /// [`resume_from_pause`](Self::resume_from_pause) is called.
    ///
    /// # Panics
    /// Panics if a pause is already active.
    pub fn request_stw_pause(&self, reason: StopReason) {
        let mut st = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if st.reason.is_some() {
            panic!("Cannot request STW pause while another is active");
        }
        st.reason = Some(reason);
        drop(st);

        // Release-store makes the pause visible to poll()'s fast path.
        self.pending.store(true, Ordering::Release);

        let mut st = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        while st.arrived < st.worker_count {
            st = self.cv.wait(st).unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        // World stopped: every registered worker is parked in enter_safepoint.
    }

    /// Resume from STW pause. Releases all waiting workers by advancing the
    /// epoch; they cannot re-execute user code before this call.
    pub fn resume_from_pause(&self) {
        {
            let mut st = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            st.reason = None;
        }
        self.epoch.fetch_add(1, Ordering::AcqRel);
        self.pending.store(false, Ordering::Release);
        self.cv.notify_all();
    }

    /// Register a new worker thread. Adjusts the arrival target under the
    /// coordinator lock so an in-flight request accounts for the newcomer.
    pub fn register_worker(&self) {
        let mut st = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        st.worker_count += 1;
        drop(st);
        self.cv.notify_all();
    }

    /// Deregister a worker thread. May unblock an in-flight request whose
    /// arrival target just dropped.
    pub fn deregister_worker(&self) {
        let mut st = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        st.worker_count = st.worker_count.saturating_sub(1);
        drop(st);
        self.cv.notify_all();
    }

    /// Get current worker count
    pub fn worker_count(&self) -> usize {
        self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).worker_count
    }

    /// Get current workers at safepoint
    pub fn workers_at_safepoint(&self) -> usize {
        self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).arrived
    }

    /// Get current pause reason
    pub fn current_reason(&self) -> Option<StopReason> {
        self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).reason
    }

    /// Check if a pause is currently pending
    pub fn is_pause_pending(&self) -> bool {
        self.is_pause_pending_fast()
    }

    /// Completed stop-the-world epochs
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// Get safepoint statistics
    pub fn stats(&self) -> (usize, usize, usize) {
        (
            self.stats.total_safepoints(),
            self.stats.total_pause_time_us(),
            self.stats.max_pause_time_us(),
        )
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        self.stats.reset();
    }
}

impl Default for SafepointCoordinator {
    fn default() -> Self {
        Self::new(1)
    }
}

/// Arc-shared handle used by the scheduler and interpreter.
#[derive(Clone)]
pub struct SafepointCoordinatorHandle(Arc<SafepointCoordinator>);

impl SafepointCoordinatorHandle {
    pub fn new(coordinator: Arc<SafepointCoordinator>) -> Self {
        Self(coordinator)
    }

    pub fn get(&self) -> &SafepointCoordinator {
        &self.0
    }
}

impl std::ops::Deref for SafepointCoordinatorHandle {
    type Target = SafepointCoordinator;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_coordinator() {
        let coord = SafepointCoordinator::new(4);
        assert_eq!(coord.worker_count(), 4);
        assert_eq!(coord.workers_at_safepoint(), 0);
        assert!(!coord.is_pause_pending());
        assert_eq!(coord.current_reason(), None);
        assert_eq!(coord.epoch(), 0);
    }

    #[test]
    fn test_default_coordinator() {
        let coord = SafepointCoordinator::default();
        assert_eq!(coord.worker_count(), 1);
    }

    #[test]
    fn test_poll_no_pause() {
        let coord = SafepointCoordinator::new(1);
        coord.poll(); // must return immediately
        assert_eq!(coord.epoch(), 0);
    }

    #[test]
    fn test_worker_registration_is_dynamic() {
        let coord = SafepointCoordinator::new(2);
        coord.register_worker();
        assert_eq!(coord.worker_count(), 3);
        coord.deregister_worker();
        assert_eq!(coord.worker_count(), 2);
    }

    #[test]
    fn full_round_trip_two_workers() {
        let coord = Arc::new(SafepointCoordinator::new(2));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let handles: Vec<_> = (0..2)
            .map(|_| {
                let c = coord.clone();
                let stop = stop.clone();
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        c.poll();
                        std::thread::sleep(std::time::Duration::from_micros(50));
                    }
                })
            })
            .collect();

        // Give workers time to enter their poll loops.
        std::thread::sleep(std::time::Duration::from_millis(5));

        coord.request_stw_pause(StopReason::GarbageCollection);

        // World stopped: both workers parked, reason visible.
        assert_eq!(coord.current_reason(), Some(StopReason::GarbageCollection));
        coord.resume_from_pause();
        assert_eq!(coord.epoch(), 1);
        assert!(!coord.is_pause_pending());

        stop.store(true, Ordering::Relaxed);
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn deregister_unblocks_coordinator() {
        let coord = Arc::new(SafepointCoordinator::new(2));

        // Only one worker will ever poll; the other "worker" deregisters.
        let c2 = coord.clone();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let s = stop.clone();
        let poller = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                c2.poll();
                std::thread::sleep(std::time::Duration::from_micros(50));
            }
        });
        std::thread::sleep(std::time::Duration::from_millis(5));
        coord.deregister_worker();

        coord.request_stw_pause(StopReason::Snapshot);
        assert_eq!(coord.current_reason(), Some(StopReason::Snapshot));
        coord.resume_from_pause();

        stop.store(true, Ordering::Relaxed);
        poller.join().unwrap();
    }

    #[test]
    fn double_request_panics() {
        let coord = Arc::new(SafepointCoordinator::new(1));
        let c2 = coord.clone();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let s = stop.clone();
        let poller = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                c2.poll();
                std::thread::sleep(std::time::Duration::from_micros(50));
            }
        });
        std::thread::sleep(std::time::Duration::from_millis(5));

        coord.request_stw_pause(StopReason::Debug);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            coord.request_stw_pause(StopReason::GarbageCollection);
        }));
        assert!(result.is_err());

        coord.resume_from_pause();
        stop.store(true, Ordering::Relaxed);
        poller.join().unwrap();
    }

    #[test]
    fn zero_workers_stop_immediately() {
        let coord = SafepointCoordinator::new(0);
        coord.request_stw_pause(StopReason::GarbageCollection);
        coord.resume_from_pause();
        assert_eq!(coord.epoch(), 1);
    }

    #[test]
    fn statistics_track_pauses() {
        let coord = Arc::new(SafepointCoordinator::new(1));
        let c2 = coord.clone();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let s = stop.clone();
        let poller = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                c2.poll();
                std::thread::sleep(std::time::Duration::from_micros(50));
            }
        });

        std::thread::sleep(std::time::Duration::from_millis(5));
        coord.request_stw_pause(StopReason::GarbageCollection);
        coord.resume_from_pause();
        stop.store(true, Ordering::Relaxed);
        poller.join().unwrap();

        let (total, _time, _max) = coord.stats();
        assert!(total >= 1, "at least one safepoint recorded");
    }
}
