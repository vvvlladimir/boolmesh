//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

pub mod dedup;
pub mod re_swap;
pub mod collapse;
use crate::{par, Data, Real, Half, Tref, next_of, Vec2, Vec3};
use collapse::{collapse_edge, collapse_short_edges, collapse_collinear_edges};
use dedup::dedupe_edges;
use re_swap::swap_degenerates;

pub fn simplify_topology<S: Data>(
    hs: &mut Vec<Half>,
    ps: &mut Vec<Vec3>,
    ns: &mut Vec<Vec3>,
    rs: &mut Vec<Tref>,
    vs: &mut Vec<S>,
    nv_from_p: usize,
    nv_from_q: usize,
    eps: Real,
) {
    let nv = nv_from_p + nv_from_q;
    let mut laps = crate::laps::Laps::new("simplify");
    split_pinched_vert(hs, ps);
    laps.lap("pinched");
    dedupe_edges(ps, hs, ns, rs);
    laps.lap("dedupe");
    collapse_short_edges(hs, ps, ns,     rs, nv, !vs.is_empty(), eps);
    laps.lap("short");
    collapse_collinear_edges(hs, ps, ns, rs, nv, !vs.is_empty(), eps);
    laps.lap("collinear");
    swap_degenerates(hs, ps, ns, rs, vs, nv, eps);
    laps.lap("swap");
}

fn head_of(hs: &[Half], i: usize) -> usize { hs[i].head as usize }
fn tail_of(hs: &[Half], i: usize) -> usize { hs[i].tail as usize }
fn pair_of(hs: &[Half], i: usize) -> usize { hs[i].pair as usize }
fn pair_up(hs: &mut[Half], i: usize, j: usize) { hs[i].pair = j as u32; hs[j].pair = i as u32; }
fn hids_of(i: usize) -> (usize, usize, usize) { let j = next_of(i); let k = next_of(j); (i, j, k) }

// When bgn halfedge and end halfedge are heading to the same vertex,
// and if collapsing the tail vertex as well, this function creates two loops.
// Beware the needless loop is not necessarily eliminated from the mesh because
// halfedges of the tail side might be connected to other triangles (would be folded though).
fn form_loops(
    hs: &mut [Half],
    ps: &mut Vec<Vec3>,
    bgn: usize,
    end: usize
) {
    ps.push(ps[tail_of(hs, bgn)]);
    ps.push(ps[head_of(hs, bgn)]);
    let bgn_vid = ps.len() - 2;
    let end_vid = ps.len() - 1;

    let bgn_pair = pair_of(hs, bgn);
    let end_pair = pair_of(hs, end);

    update_vid_around_star(hs, bgn_pair, end_pair, bgn_vid);
    update_vid_around_star(hs, end, bgn, end_vid);

    hs[bgn].pair = end_pair as u32;
    hs[end_pair].pair = bgn as u32;
    hs[end].pair = bgn_pair as u32;
    hs[bgn_pair].pair = end as u32;

    remove_if_folded(hs, ps, end);
}

// Removing fold paired triangles from the mesh. The process is either of:
// 1. Non-2-manifold, two triangles are completely isolated from the mesh
// 2. Non-2-manifold, one vertex is isolated from the mesh
// 3. Topologically valid, but just two vertex positions are the same
// Beware the case that triangles only connected at a vertex but not by halfedge
// are eliminated by split_pinched_vert and dedupe_edges functions.
fn remove_if_folded(
    hs: &mut [Half],
    ps: &mut [Vec3],
    hid: usize
) {
    let (i0, i1, i2) = hids_of(hid);
    let (j0, j1, j2) = hids_of(pair_of(hs, hid));

    if hs[i1].pair().is_none() || head_of(hs, i1) != head_of(hs, j1) { return; }

    match (pair_of(hs, i1) == j2, pair_of(hs, i2) == j1) {
        (true, true)  => for i in [i0, i1, i2] { ps[tail_of(hs, i)] = Vec3::NAN; },
        (true, false) => { ps[tail_of(hs, i1)] = Vec3::NAN; }
        (false, true) => { ps[tail_of(hs, j1)] = Vec3::NAN; }
        _ => {} // topo valid
    }
    pair_up(hs, hs[i1].pair as usize, hs[j2].pair as usize);
    pair_up(hs, hs[i2].pair as usize, hs[j1].pair as usize);
    for i in [i0, i1, i2] { hs[i] = Half::default(); }
    for j in [j0, j1, j2] { hs[j] = Half::default(); }
}

fn split_pinched_vert(
    hs: &mut [Half],
    ps: &mut Vec<Vec3>
) {
    // Only a vertex whose halfedges form more than one star is split; walking the others would
    // write back what they hold, so the walk below visits the halfedges of split vertices alone.
    let pinched = pinched_verts(hs, ps.len());
    let mut v_processed = vec![false; ps.len()];
    let mut h_processed = vec![false; hs.len()];

    for hid in pinched {
        if h_processed[hid] { continue; }
        let mut vid = hs[hid].tail;
        if v_processed[vid as usize] {
            ps.push(ps[vid as usize]);
            vid = (ps.len() - 1) as u32;
        } else { v_processed[vid as usize] = true; }

        // loop halfedges around their tail ccw way
        let mut cur = hid;
        loop {
            cur = next_of(hs[cur].pair as usize);
            h_processed[cur] = true;
            hs[cur].tail = vid;
            hs[hs[cur].pair as usize].head = vid;
            if cur == hid { break; }
        }
    }
}

/// The halfedges, in order, whose tail has more halfedges than the star around one of them holds.
fn pinched_verts(hs: &[Half], nv: usize) -> Vec<usize> {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    let count = (0..nv).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>();
    let first = (0..nv).map(|_| AtomicUsize::new(usize::MAX)).collect::<Vec<_>>();
    par::for_each(hs.len(), |h| {
        let v = hs[h].tail;
        if v == ((usize::MAX) as u32) { return; }
        count[v as usize].fetch_add(1, Relaxed);
        first[v as usize].fetch_min(h, Relaxed);
    });
    let split = par::tabulate(nv, |v| {
        let start = first[v].load(Relaxed);
        if start == usize::MAX { return false; }
        let (mut cur, mut n) = (start, 0);
        loop {
            cur = next_of(hs[cur].pair as usize);
            n += 1;
            if cur == start || n > hs.len() { break; }
        }
        n != count[v].load(Relaxed)
    });
    par::flat_tabulate(hs.len(), |h| (hs[h].tail != ((usize::MAX) as u32) && split[hs[h].tail as usize]).then_some(h))
}

fn update_vid_around_star(
    hs: &mut [Half],
    bgn: usize, // incoming bgn halfedge id (inclusive)
    end: usize, // incoming end halfedge id (exclusive)
    vid: usize, // alternative vid
) {
    let mut cur = bgn;
    while cur != end {
        hs[cur].head = vid as u32; cur = next_of(cur);
        hs[cur].tail = vid as u32; cur = pair_of(hs, cur);
        assert_ne!(cur, bgn);
    }
}

fn collapse_triangle(hs: &mut [Half], hids: &(usize, usize, usize)) {
    if hs[hids.1].pair().is_none() { return; }
    let pair1 = pair_of(hs, hids.1);
    let pair2 = pair_of(hs, hids.2);
    hs[pair1].pair = pair2 as u32;
    hs[pair2].pair = pair1 as u32;
    for i in [hids.0, hids.1, hids.2] { hs[i] = Half { prop: hs[i].prop, ..Half::default() }; }
}

fn is01_longest_2d(
    p0: &Vec2,
    p1: &Vec2,
    p2: &Vec2
) -> bool {
    let e01 = (*p1 - *p0).length_squared();
    let e12 = (*p2 - *p1).length_squared();
    let e20 = (*p0 - *p2).length_squared();
    e01 > e12 && e01 > e20
}

