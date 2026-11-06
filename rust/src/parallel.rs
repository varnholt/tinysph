//! A small pool of worker threads that live as long as the solver, so a substep does not pay for spawning threads in
//! every phase.
//!
//! The work is split into chunks of [`CHUNK_SIZE`] elements that the workers (and the calling thread) pick up one by one,
//! like the c++ `parallelFor`. Every element is computed by the same function no matter which thread runs it, so results
//! never depend on the thread count.
//!
//! Workers outlive any borrow, so a task is a `'static` closure: it gets its inputs through an `Arc` and writes its
//! results into per chunk buffers that the caller copies out once [`Pool::map`] returns. By then every worker has dropped
//! its handle to the task, so the caller owns the inputs and the buffers alone again (`Arc::get_mut` succeeds).

use std::num::NonZeroUsize;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};

/// elements per chunk, as in the c++ code; small enough to balance the load, large enough to keep the overhead low
const CHUNK_SIZE: usize = 256;

type Task = Arc<dyn Fn(usize) + Send + Sync>;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[derive(Default)]
struct State {
    generation: u64,
    task: Option<Task>,
    chunk_count: usize,
    busy_workers: usize,
    panicked: bool,
    shutdown: bool,
}

#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    work_available: Condvar,
    work_done: Condvar,
    next_chunk: AtomicUsize,
}

impl Shared {
    /// Runs chunks until none are left; returns false if the task panicked.
    fn run_chunks(&self, task: &Task, chunk_count: usize) -> bool {
        panic::catch_unwind(AssertUnwindSafe(|| {
            loop {
                let chunk = self.next_chunk.fetch_add(1, Ordering::Relaxed);
                if chunk >= chunk_count {
                    break;
                }
                task(chunk);
            }
        }))
        .is_ok()
    }

    fn work(&self) {
        let mut seen_generation = 0;
        loop {
            let (task, chunk_count) = {
                let mut state = lock(&self.state);
                while state.generation == seen_generation && !state.shutdown {
                    state = self.work_available.wait(state).unwrap_or_else(PoisonError::into_inner);
                }
                if state.shutdown {
                    return;
                }
                seen_generation = state.generation;
                (state.task.clone(), state.chunk_count)
            };
            let succeeded = task.as_ref().is_none_or(|task| self.run_chunks(task, chunk_count));
            drop(task); // before reporting back, so the caller holds the only reference afterwards
            let mut state = lock(&self.state);
            state.panicked |= !succeeded;
            state.busy_workers -= 1;
            if state.busy_workers == 0 {
                self.work_done.notify_one();
            }
        }
    }
}

/// Per chunk result buffers of [`Pool::map`], kept between calls so a substep does not allocate.
pub(crate) struct ChunkBuffer<T> {
    chunks: Arc<Vec<Mutex<Vec<T>>>>,
    chunk_count: usize,
}

impl<T> Default for ChunkBuffer<T> {
    fn default() -> Self {
        Self { chunks: Arc::default(), chunk_count: 0 }
    }
}

impl<T> ChunkBuffer<T> {
    /// The results of the last [`Pool::map`], in index order.
    pub(crate) fn values(&mut self) -> impl Iterator<Item = &T> {
        let chunks = Arc::get_mut(&mut self.chunks).expect("no task holds the buffer after Pool::map");
        chunks[..self.chunk_count].iter_mut().flat_map(|chunk| chunk.get_mut().unwrap_or_else(PoisonError::into_inner).iter())
    }
}

/// Worker threads, started on first use and stopped when the pool is dropped.
#[derive(Default)]
pub(crate) struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl Pool {
    fn start_workers(&mut self) {
        let thread_count = thread::available_parallelism().map_or(1, NonZeroUsize::get);
        // the calling thread works as well
        for _ in 1..thread_count {
            let shared = Arc::clone(&self.shared);
            match thread::Builder::new().name("tinysph worker".into()).spawn(move || shared.work()) {
                Ok(worker) => self.workers.push(worker),
                Err(_) => break, // fewer workers only make it slower
            }
        }
    }

    /// Runs `task(chunk)` for every chunk in `0..chunk_count`, on the workers and the calling thread.
    fn run(&mut self, chunk_count: usize, task: &Task) {
        if self.workers.is_empty() && chunk_count > 1 {
            self.start_workers();
        }
        if self.workers.is_empty() || chunk_count <= 1 {
            for chunk in 0..chunk_count {
                task(chunk);
            }
            return;
        }
        self.shared.next_chunk.store(0, Ordering::Relaxed);
        {
            let mut state = lock(&self.shared.state);
            state.generation += 1;
            state.task = Some(Arc::clone(task));
            state.chunk_count = chunk_count;
            state.busy_workers = self.workers.len();
        }
        self.shared.work_available.notify_all();
        let mut succeeded = self.shared.run_chunks(task, chunk_count);
        let mut state = lock(&self.shared.state);
        while state.busy_workers > 0 {
            state = self.shared.work_done.wait(state).unwrap_or_else(PoisonError::into_inner);
        }
        state.task = None;
        succeeded &= !std::mem::take(&mut state.panicked);
        drop(state);
        assert!(succeeded, "a parallel task panicked");
    }

    /// Computes `function(index)` for every index in `0..count`; read the results with [`ChunkBuffer::values`].
    pub(crate) fn map<T, F>(&mut self, count: usize, buffer: &mut ChunkBuffer<T>, function: F)
    where
        T: Send + 'static,
        F: Fn(usize) -> T + Send + Sync + 'static,
    {
        let chunk_count = count.div_ceil(CHUNK_SIZE);
        let chunks = Arc::get_mut(&mut buffer.chunks).expect("no task holds the buffer between calls");
        if chunks.len() < chunk_count {
            chunks.resize_with(chunk_count, || Mutex::new(Vec::with_capacity(CHUNK_SIZE)));
        }
        buffer.chunk_count = chunk_count;
        let chunks = Arc::clone(&buffer.chunks);
        let task: Task = Arc::new(move |chunk| {
            let mut values = lock(&chunks[chunk]);
            values.clear();
            values.extend((chunk * CHUNK_SIZE..count.min((chunk + 1) * CHUNK_SIZE)).map(&function));
        });
        self.run(chunk_count, &task);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        lock(&self.shared.state).shutdown = true;
        self.shared.work_available.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}
