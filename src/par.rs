//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

// Loops and sorts that run on rayon when the feature is on and on one thread when it is off.
// Every helper gives the same answer either way: sorts are stable or sort unique keys, and
// collected results keep their index order.

use std::cmp::Ordering;
#[cfg(feature = "rayon")] use rayon::prelude::*;

pub fn tabulate<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    #[cfg(feature = "rayon")] { (0..n).into_par_iter().map(f).collect() }
    #[cfg(not(feature = "rayon"))] { (0..n).map(f).collect() }
}

pub fn for_each(n: usize, f: impl Fn(usize) + Sync + Send) {
    #[cfg(feature = "rayon")] { (0..n).into_par_iter().for_each(f) }
    #[cfg(not(feature = "rayon"))] { (0..n).for_each(f) }
}

pub fn map<S: Sync, T: Send>(src: &[S], f: impl Fn(&S) -> T + Sync + Send) -> Vec<T> {
    #[cfg(feature = "rayon")] { src.par_iter().map(f).collect() }
    #[cfg(not(feature = "rayon"))] { src.iter().map(f).collect() }
}

pub fn for_each_mut<T: Send>(dst: &mut [T], f: impl Fn(usize, &mut T) + Sync + Send) {
    #[cfg(feature = "rayon")] { dst.par_iter_mut().enumerate().for_each(|(i, t)| f(i, t)) }
    #[cfg(not(feature = "rayon"))] { dst.iter_mut().enumerate().for_each(|(i, t)| f(i, t)) }
}

pub fn all<T: Sync>(src: &[T], f: impl Fn(usize, &T) -> bool + Sync + Send) -> bool {
    #[cfg(feature = "rayon")] { src.par_iter().enumerate().all(|(i, t)| f(i, t)) }
    #[cfg(not(feature = "rayon"))] { src.iter().enumerate().all(|(i, t)| f(i, t)) }
}

/// Maps each index to any number of items, kept in index order.
pub fn flat_tabulate<T: Send, I: IntoIterator<Item = T>>(n: usize, f: impl Fn(usize) -> I + Sync + Send) -> Vec<T> {
    #[cfg(feature = "rayon")] { (0..n).into_par_iter().flat_map_iter(f).collect() }
    #[cfg(not(feature = "rayon"))] { (0..n).flat_map(f).collect() }
}

/// Only for keys that never compare equal, so that an unstable sort is still deterministic.
pub fn sort_unique<T: Ord + Send>(v: &mut [T]) {
    #[cfg(feature = "rayon")] { v.par_sort_unstable() }
    #[cfg(not(feature = "rayon"))] { v.sort_unstable() }
}

/// Only for an order under which no two items compare equal.
pub fn sort_unique_by<T: Send>(v: &mut [T], f: impl Fn(&T, &T) -> Ordering + Sync) {
    #[cfg(feature = "rayon")] { v.par_sort_unstable_by(f) }
    #[cfg(not(feature = "rayon"))] { v.sort_unstable_by(f) }
}

pub fn sort_by_key<T: Send, K: Ord>(v: &mut [T], f: impl Fn(&T) -> K + Sync) {
    #[cfg(feature = "rayon")] { v.par_sort_by_key(f) }
    #[cfg(not(feature = "rayon"))] { v.sort_by_key(f) }
}

/// Slots of one slice that several threads write at once, each slot by one thread only.
pub struct Shared<T> { at: *mut T, len: usize }
unsafe impl<T: Send> Sync for Shared<T> {}
unsafe impl<T: Send> Send for Shared<T> {}

impl<T> Shared<T> {
    pub fn new(slice: &mut [T]) -> Self { Self { at: slice.as_mut_ptr(), len: slice.len() } }

    /// # Safety
    /// No other thread reads or writes slot `i` while this runs, and the slice outlives `self`.
    pub unsafe fn set(&self, i: usize, value: T) {
        assert!(i < self.len);
        unsafe { *self.at.add(i) = value; }
    }

    /// # Safety
    /// As for `set`.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get_mut(&self, i: usize) -> &mut T {
        assert!(i < self.len);
        unsafe { &mut *self.at.add(i) }
    }
}
