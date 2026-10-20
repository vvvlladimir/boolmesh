//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

use std::collections::HashMap;
use crate::{par, Half, Tref, Vec3, next_of};
use super::{pair_up, tail_of, update_vid_around_star};

fn dedupe_edge(
    ps: &mut Vec<Vec3>,
    hs: &mut Vec<Half>,
    ns: &mut Vec<Vec3>,
    rs: &mut Vec<Tref>,
    hid: usize,
) {
    // 1: (imagine a Riemann surface)
    // Orbit around head vert to check that the surface crosses over at the same halfedge.
    // In that case, split the head vert and create new triangles at first.
    // It is still crossover as a surface, but it's not a 4-manifold anymore.
    // For here we do not care either the tail vert is attached or not (care it in the 3rd step).
    let tail = hs[hid].tail;
    let head = hs[hid].head;
    let val = hs[next_of(hid)].prop;
    let opp = hs[next_of(hid)].pair;
    let mut cur = hs[next_of(hid)].pair;
    while cur != (hid as u32) {
        if tail_of(hs, cur as usize) == (tail as usize) {
            ps.push(ps[head as usize]);
            let copy = ps.len() - 1;
            cur = hs[next_of(cur as usize)].pair;
            update_vid_around_star(hs, cur as usize, opp as usize, copy);

            let nh = hs.len();
            let pair = hs[cur as usize].pair;
            hs.push(Half::new(head as usize, copy, usize::MAX, val as usize));
            hs.push(Half::new(copy, tail_of(hs, cur as usize), usize::MAX, val as usize));
            hs.push(Half::new(tail_of(hs, cur as usize), head as usize, usize::MAX, hs[cur as usize].prop as usize));
            pair_up(hs, nh + 2, pair as usize);
            pair_up(hs, nh + 1, cur as usize);

            let nh = hs.len();
            let pair = hs[opp as usize].pair;
            hs.push(Half::new(copy, head as usize, usize::MAX, val as usize));
            hs.push(Half::new(head as usize, tail_of(hs, opp as usize), usize::MAX, val as usize));
            hs.push(Half::new(tail_of(hs, opp as usize), copy, usize::MAX, hs[opp as usize].prop as usize));
            pair_up(hs, nh + 2, pair as usize);
            pair_up(hs, nh + 1, opp as usize);

            // Pair first new halfedge of second tri with first of first tri
            pair_up(hs, nh, nh - 3);

            // Push per-face data if present
            rs.push(rs[(cur / 3) as usize]);
            ns.push(ns[(cur / 3) as usize]);
            rs.push(rs[(opp / 3) as usize]);
            ns.push(ns[(opp / 3) as usize]);
            break;
        }
        cur = hs[next_of(cur as usize)].pair;
    }

    // 2: Pinch the head vert if 1 does not happen.
    if cur == (hid as u32) {
        // Separate topological unit needs no new faces to be split
        let new_vert = ps.len();
        ps.push(ps[head as usize]);
        // Rewire the entire star around NextHalfedge(current) to new_vert
        let start = next_of(cur as usize);
        let mut e = start;
        loop {
            hs[e].tail = new_vert as u32;
            let p = hs[e].pair;
            hs[p as usize].head = new_vert as u32;
            e = next_of(p as usize);
            if e == start { break; }
        }
    }
    // 3: Pinch the tail vert anyway.
    let pair = hs[hid].pair;
    let mut curr = hs[next_of(pair as usize)].pair;
    while curr != pair {
        if hs[curr as usize].tail == head { break; }
        curr = hs[next_of(curr as usize)].pair;
    }
    if curr == pair {
        // Split the pinched vert the previous split created.
        let new_vert = ps.len();
        ps.push(ps[head as usize]);
        let bgn = next_of(curr as usize);
        let mut e = bgn;
        loop {
            hs[e].tail = new_vert as u32;
            let p = hs[e].pair;
            hs[p as usize].head = new_vert as u32;
            e = next_of(p as usize);
            if e == bgn { break; }
        }
    }
}

// Every halfedge sharing its tail and head with one of lower index in the same star; a star
// is gathered once, from its lowest halfedge.
fn duplicates(hs: &[Half], nv: usize) -> Vec<usize> {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    let valid = |h: &Half| h.tail().is_some() && h.head().is_some();
    let first = (0..nv).map(|_| AtomicUsize::new(usize::MAX)).collect::<Vec<_>>();
    par::for_each(hs.len(), |h| if valid(&hs[h]) { first[hs[h].tail as usize].fetch_min(h, Relaxed); });
    let starts = par::flat_tabulate(hs.len(), |h| {
        if !valid(&hs[h]) { return None; }
        // A star's lowest halfedge: the lowest of its tail when the tail has one star, which
        // split_pinched_vert has made sure of; otherwise found by walking the star.
        if first[hs[h].tail as usize].load(Relaxed) == h { return Some(h); }
        let mut cur = next_of(hs[h].pair as usize);
        while cur != h { if cur < h { return None; } cur = next_of(hs[cur].pair as usize); }
        Some(h)
    });
    par::flat_tabulate(starts.len(), |s| {
        let hid = starts[s];
        let mut heads = Vec::<(usize, usize)>::new();
        let mut map = HashMap::<usize, usize>::new();
        let mut cur = hid;
        loop {
            let head = hs[cur].head;
            if map.is_empty() {
                if let Some(p) = heads.iter_mut().find(|p| p.0 == (head as usize)) { p.1 = p.1.min(cur); }
                else {
                    heads.push((head as usize, cur));
                    if heads.len() > 32 { for (k, v) in heads.drain(..) { map.insert(k, v); } }
                }
            } else { map.entry(head as usize).and_modify(|m| { if cur < *m { *m = cur; } }).or_insert(cur); }
            cur = next_of(hs[cur].pair as usize);
            if cur == hid { break; }
        }
        let mut dups = vec![];
        loop {
            let head = hs[cur].head;
            let mini = if map.is_empty() { heads.iter().find(|p| p.0 == (head as usize)).map(|p| p.1) }
                       else { map.get(&(head as usize)).copied() };
            if mini.is_some_and(|id| id != cur) { dups.push(cur); }
            cur = next_of(hs[cur].pair as usize);
            if cur == hid { break; }
        }
        dups
    })
}

pub fn dedupe_edges(
    ps: &mut Vec<Vec3>,
    hs: &mut Vec<Half>,
    ns: &mut Vec<Vec3>,
    rs: &mut Vec<Tref>
) {
    if hs.is_empty() { return; }
    loop {
        let mut dups = duplicates(hs, ps.len());

        dups.sort_unstable();
        dups.dedup();

        let mut flag = 0usize;
        for &hid in &dups {
            dedupe_edge(ps, hs, ns, rs, hid);
            flag += 1;
        }

        if flag == 0 { break; }

        #[cfg(feature = "verbose")]
        println!("{} dedup", flag);
    }
}
