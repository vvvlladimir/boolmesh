//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

use crate::{next_of, Half, Vec3u};
#[cfg(feature = "rayon")] use rayon::prelude::*;

const REMOVE_FLAG: u32 = crate::NONE - 1;

#[cfg(not(feature = "rayon"))]
pub fn tri_halfs_single(ts: &[Vec3u]) -> Vec<Half> {
    let nh = ts.len() * 3;
    let ne = nh / 2;
    let nt = nh / 3;
    let mut hs = vec![Half::default(); nh];
    let mut is = (0..nh).collect::<Vec<_>>();
    let mut ky = vec![0u64; nh];

    for (tid, t) in ts.iter().enumerate() {
        for i in 0..3 {
            let j = (i + 1) % 3;
            let e = tid * 3 + i;
            let i0 = t[i];
            let i1 = t[j];
            hs[e].tail = i0 as u32;
            hs[e].head = i1 as u32;
            let a = std::cmp::min(i0, i1) as u64;
            let b = std::cmp::max(i0, i1) as u64;
            let f = if i0 < i1 { 1u64 } else { 0u64 } << 63;
            ky[e] = f | (a << 32) | b;
        }
    }

    is.sort_by_key(|&i| ky[i]);

    let mut ini = 0;
    for i in 0..ne { ini = step(&mut is, &mut hs, i, ini); }

    for i in 0..ne {
        let i0 = is[i];
        let i1 = is[i + ne];
        if hs[i0].pair != REMOVE_FLAG {
            hs[i0].pair = i1 as u32;
            hs[i1].pair = i0 as u32;
        } else {
            hs[i0] = Half::default();
            hs[i1] = Half::default();
        }
    }

    // reorder halfedges: step 1
    for t in 0..nt {
        let i = t * 3;
        let f = [hs[i].clone(), hs[i + 1].clone(), hs[i + 2].clone(), ];
        let mut mini = 0;
        if f[1].tail < f[mini].tail { mini = 1; }
        if f[2].tail < f[mini].tail { mini = 2; }
        for j in 0..3 { hs[i + j] = f[(mini + j) % 3].clone(); }
    }

    // reorder halfedges: step 2
    for t in 0..nt {
        for i in t * 3..(t + 1) * 3 {
            let tail = hs[i].tail;
            let pair = hs[i].pair;
            if pair == REMOVE_FLAG || pair >= ((hs.len()) as u32) { continue; }
            let j = (pair / 3) * 3;
            let f = (0..3).find(|&k| hs[(j + k) as usize].head == tail);
            if let Some(k) = f { hs[i].pair = j + k; }
        }
    }
    hs
}

#[cfg(feature = "rayon")]
pub fn tri_halfs_multi(ts: &[Vec3u]) -> Vec<Half> {
    let mut laps = crate::laps::Laps::new("pairs");
    let nh = ts.len() * 3;
    let ne = nh / 2;
    let mut hs  = vec![Half::default(); nh];
    let mut is = (0..nh).collect::<Vec<_>>();
    let mut ky = vec![0u64; nh];

    hs.par_chunks_mut(3)
        .zip(ky.par_chunks_mut(3))
        .zip(ts.par_iter())
        .for_each(|((hs_, ky_), t)| {
            for i in 0..3 {
                let j  = (i + 1) % 3;
                let i0 = t[i];
                let i1 = t[j];
                hs_[i].tail = i0 as u32;
                hs_[i].head = i1 as u32;
                let a = std::cmp::min(i0, i1) as u64;
                let b = std::cmp::max(i0, i1) as u64;
                let f = if i0 < i1 { 1u64 } else { 0u64 } << 63;
                ky_[i] = f | (a << 32) | b;
            }
        });

    // Sorting (key, index) pairs, unique by index, orders them as a stable sort by key would.
    let mut kis = ky.par_iter().zip(is.par_iter()).map(|(&k, &i)| (k, i)).collect::<Vec<_>>();
    kis.par_sort_unstable();
    is.par_iter_mut().zip(kis.par_iter()).for_each(|(i, ki)| *i = ki.1);
    drop(kis);
    laps.lap("sort");

    if aligned(&is, &hs) {
        step_runs(&mut is, &mut hs);
    } else {
        let mut ini = 0;
        for i in 0..ne { ini = step(&mut is, &mut hs, i, ini); }
    }

    laps.lap("step");
    let slots = crate::par::Shared::new(&mut hs);
    (0..ne).into_par_iter().for_each(|i| {
        let (i0, i1) = (is[i], is[i + ne]);
        // SAFETY: `is` is a permutation, so halfedges i0 and i1 belong to this pass alone.
        let (h0, h1) = unsafe { (slots.get_mut(i0), slots.get_mut(i1)) };
        if h0.pair != REMOVE_FLAG {
            h0.pair = i1 as u32;
            h1.pair = i0 as u32;
        } else {
            *h0 = Half::default();
            *h1 = Half::default();
        }
    });

    laps.lap("pair");
    // reorder halfedges: step 1
    hs.par_chunks_mut(3).for_each(|t| {
        let f = [t[0].clone(), t[1].clone(), t[2].clone()];
        let mut mini = 0;
        if f[1].tail < f[mini].tail { mini = 1; }
        if f[2].tail < f[mini].tail { mini = 2; }
        for j in 0..3 { t[j] = f[(mini + j) % 3].clone(); }
    });

    // reorder halfedges: step 2, reading only heads, which it does not write
    let pairs = (0..nh).into_par_iter().map(|i| {
        let tail = hs[i].tail;
        let pair = hs[i].pair;
        if pair == REMOVE_FLAG || pair >= ((hs.len()) as u32) { return pair; }
        let j = (pair / 3) * 3;
        (0..3).find(|&k| hs[(j + k) as usize].head == tail).map_or(pair, |k| j + k)
    }).collect::<Vec<_>>();
    hs.par_iter_mut().zip(pairs).for_each(|(h, p)| h.pair = p);
    laps.lap("reorder");
    hs
}


fn edge_of(h: &Half) -> (usize, usize) { (((h.tail.min(h.head)) as usize), ((h.tail.max(h.head)) as usize)) }

// Whether every edge has as many halfedges one way as the other, so that the two sorted halves
// line up edge by edge and each edge's run can be stepped through on its own.
#[cfg(feature = "rayon")]
fn aligned(is: &[usize], hs: &[Half]) -> bool {
    let ne = hs.len() / 2;
    (0..ne).into_par_iter().all(|i| edge_of(&hs[is[i]]) == edge_of(&hs[is[i + ne]]))
}

// What `step` does over all of `is`, one edge's run per task: a run reads and writes only its
// own halfedges and its own stretch of the second half, and `step` never looks past it.
#[cfg(feature = "rayon")]
fn step_runs(is: &mut [usize], hs: &mut [Half]) {
    let ne = hs.len() / 2;
    let starts = (0..ne).into_par_iter()
        .filter(|&i| i == 0 || edge_of(&hs[is[i]]) != edge_of(&hs[is[i - 1]]))
        .collect::<Vec<_>>();
    let changed = (0..starts.len()).into_par_iter().filter_map(|r| {
        let (s, e) = (starts[r], starts.get(r + 1).copied().unwrap_or(ne));
        let mut ahead = is[s + ne..e + ne].to_vec();
        let mut removed = vec![];
        for i in s..e {
            let i0 = is[i];
            if let Some(k) = (0..ahead.len()).find(|&k| hs[next_of(i0)].head == hs[next_of(ahead[k])].head) {
                removed.extend([i0, ahead[k]]);
                ahead.swap(i - s, k);
            }
        }
        (!removed.is_empty()).then_some((s, ahead, removed))
    }).collect::<Vec<_>>();
    for (s, ahead, removed) in changed {
        is[s + ne..s + ne + ahead.len()].copy_from_slice(&ahead);
        for h in removed { hs[h].pair = REMOVE_FLAG; }
    }
}

// By sorting forward and backward halfedges by key,
// now halfedges of the same mini ids are sorted in a sequence.
// It treats the triangle overlap case here, also considers 4-manifold case.
fn step(
    is: &mut [usize],
    hs: &mut [Half],
    i: usize,
    consecutive_ini: usize,
) -> usize {
    let nh = hs.len();
    let ne = nh / 2;
    let i0 = is[i];
    let h0 = hs[i0].clone();
    let j = i + ne;
    let mut k = consecutive_ini + ne;
    loop {
        if k >= nh { break; }
        let i1 = is[k];
        let h1 = hs[i1].clone();

        if !(h0.tail == h1.head && h0.head == h1.tail) { break; }
        if hs[next_of(i0)].head == hs[next_of(i1)].head { // overlap
            hs[i0].pair = REMOVE_FLAG;
            hs[i1].pair = REMOVE_FLAG;
            if k != j { is.swap(j, k); }
            break;
        }
        k += 1;
    }

    if i + 1 == ne { return consecutive_ini; }
    let i2 = is[i + 1];
    let h2 = hs[i2].clone();
    if h0.tail == h2.tail && h0.head == h2.head { consecutive_ini } else { i + 1 }
}
