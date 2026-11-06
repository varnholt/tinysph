//! Wall clock time for `StepStatistics::wall_time`.
//!
//! `std::time::Instant` panics on `wasm32-unknown-unknown`, which has no clock without javascript, so there the wall
//! time is reported as zero.

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use std::time::Instant;

/// Stands in for `std::time::Instant` on the web.
#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy)]
pub(crate) struct Instant;

#[cfg(target_arch = "wasm32")]
impl Instant {
    pub(crate) fn now() -> Self {
        Instant
    }

    #[allow(clippy::unused_self)]
    pub(crate) fn elapsed(self) -> std::time::Duration {
        std::time::Duration::ZERO
    }
}
