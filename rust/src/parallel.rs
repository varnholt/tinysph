//! Per-particle loops on scoped threads. Every element is computed by the same function no matter how the range is
//! split, so results never depend on the thread count.

use std::num::NonZeroUsize;
use std::sync::OnceLock;
use std::thread;

/// below this many elements per thread, spawning costs more than it saves
const MIN_CHUNK: usize = 1024;

fn thread_count() -> usize {
    static COUNT: OnceLock<usize> = OnceLock::new();
    *COUNT.get_or_init(|| thread::available_parallelism().map_or(1, NonZeroUsize::get))
}

fn chunk_size(count: usize) -> usize {
    let threads = thread_count().min(count.div_ceil(MIN_CHUNK)).max(1);
    count.div_ceil(threads).max(1)
}

/// `output[i] = function(i)` for every index.
pub(crate) fn fill<T, F>(output: &mut [T], function: F)
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    let chunk = chunk_size(output.len());
    if chunk >= output.len() {
        for (index, value) in output.iter_mut().enumerate() {
            *value = function(index);
        }
        return;
    }
    thread::scope(|scope| {
        for (chunk_index, slice) in output.chunks_mut(chunk).enumerate() {
            let function = &function;
            scope.spawn(move || {
                let first = chunk_index * chunk;
                for (offset, value) in slice.iter_mut().enumerate() {
                    *value = function(first + offset);
                }
            });
        }
    });
}

/// `function(i, &mut first[i], &mut second[i])` for every index; both slices have the same length.
pub(crate) fn for_each_pair<A, B, F>(first: &mut [A], second: &mut [B], function: F)
where
    A: Send,
    B: Send,
    F: Fn(usize, &mut A, &mut B) + Sync,
{
    debug_assert_eq!(first.len(), second.len());
    let chunk = chunk_size(first.len());
    if chunk >= first.len() {
        for (index, (a, b)) in first.iter_mut().zip(second.iter_mut()).enumerate() {
            function(index, a, b);
        }
        return;
    }
    thread::scope(|scope| {
        for (chunk_index, (slice_a, slice_b)) in first.chunks_mut(chunk).zip(second.chunks_mut(chunk)).enumerate() {
            let function = &function;
            scope.spawn(move || {
                let start = chunk_index * chunk;
                for (offset, (a, b)) in slice_a.iter_mut().zip(slice_b.iter_mut()).enumerate() {
                    function(start + offset, a, b);
                }
            });
        }
    });
}
