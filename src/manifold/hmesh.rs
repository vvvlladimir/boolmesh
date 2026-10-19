//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.
#![allow(clippy::needless_range_loop)]

use std::f64::consts::PI;
use crate::{par, Vec3, Vec2u, Vec3u, Real};

/// Hmesh preserves the order of pos and idx in any cases.
/// Edges are ordered so as the edge is forward (tail idx < head idx)
#[derive(Debug, Clone)]
pub(in crate::manifold) struct Hmesh {
    pub nv: usize,
    pub nf: usize,
    pub nh: usize,
    pub twin: Vec<usize>,
    pub head: Vec<usize>,
    pub tail: Vec<usize>,
    pub half: Vec<usize>,
    pub vns: Vec<Vec3>,
    pub fns: Vec<Vec3>,
}

fn edge_topology(
    pos: &[Vec3],
    idx: &[Vec3u],
    e2v: &mut Vec<Vec2u>,
    e2f: &mut Vec<Vec2u>,
    f2e: &mut Vec<Vec3u>,
) -> Result<(), String> {
    if pos.is_empty() { return Err("empty pos matrix".into()); }
    if idx.is_empty() { return Err("empty idx matrix".into()); }
    if pos.len() > u32::MAX as usize || 3 * idx.len() > u32::MAX as usize { return Err("mesh too large".into()); }

    // (v1 << 32 | v2, face << 2 | side) sorts as the tuple (v1, v2, face, side) and is half its size.
    let mut ett: Vec<(u64, u64)> = par::tabulate(3 * idx.len(), |k| {
        let (i, j) = (k / 3, k % 3);
        let mut v1 = idx[i][j];
        let mut v2 = idx[i][(j + 1) % 3];
        if v1 > v2 { std::mem::swap(&mut v1, &mut v2); }
        (((v1 as u64) << 32) | v2 as u64, ((i as u64) << 2) | j as u64)
    });
    par::sort_unique(&mut ett);
    let row = |k: usize| {
        let (e, f) = ett[k];
        [(e >> 32) as usize, (e & 0xFFFF_FFFF) as usize, (f >> 2) as usize, (f & 3) as usize]
    };

    let mut ne = 1;
    for i in 0..ett.len() - 1 {
        if ett[i].0 != ett[i + 1].0 { ne += 1; }
    }

    e2v.resize(ne, Vec2u::MAX);
    e2f.resize(ne, Vec2u::MAX);
    f2e.resize(idx.len(), Vec3u::MAX);
    ne = 0;

    let mut i = 0;
    while i < ett.len() {
        if i == ett.len() - 1 || ett[i].0 != ett[i + 1].0 {
            // Border edge
            let [v1, v2, i, j] = row(i);
            e2v[ne][0] = v1;
            e2v[ne][1] = v2;
            e2f[ne][0] = i;
            f2e[i][j]  = ne;
        } else {
            let r1 = row(i);
            let r2 = row(i + 1);
            e2v[ne][0] = r1[0];
            e2v[ne][1] = r1[1];
            e2f[ne][0] = r1[2];
            e2f[ne][1] = r2[2];
            f2e[r1[2]][r1[3]] = ne;
            f2e[r2[2]][r2[3]] = ne;
            i += 1; // skip the next one
        }
        ne += 1;
        i += 1;
    }

    let e2v: &[Vec2u] = e2v;
    par::for_each_mut(e2f, |i, ef| {
        let fid = ef[0];
        let flip = !(0..3).any(|j| idx[fid][j] == e2v[i][0] && idx[fid][(j + 1) % 3] == e2v[i][1]);
        if flip { *ef = Vec2u::new(ef[1], ef[0]); }
    });
    Ok(())
}

impl Hmesh {
    pub fn new(
        pos: &[Vec3],
        idx: &[Vec3u],
    ) -> Result<Self, String> {
        let mut e2v = Default::default();
        let mut e2f = Default::default();
        let mut f2e = Default::default();
        edge_topology(pos, idx, &mut e2v, &mut e2f, &mut f2e)?;

        let nv = pos.len();
        let nf = idx.len();
        let nh = e2v.len() * 2;
        let nc = 3 * nf;

        let head = par::tabulate(nh, |ih| if ih < nc { idx[ih / 3][(ih + 1) % 3] } else { usize::MAX });
        let tail = par::tabulate(nh, |ih| if ih < nc { idx[ih / 3][ih % 3] } else { usize::MAX });
        // The other halfedge on the same edge: an edge holds at most two, in the faces it lists.
        let twin = par::tabulate(nh, |ih| {
            if ih >= nc { return usize::MAX; }
            let ie = f2e[ih / 3][ih % 3];
            for f in [e2f[ie][0], e2f[ie][1]] {
                if f == usize::MAX { continue; }
                if let Some(k) = (0..3).find(|&k| 3 * f + k != ih && f2e[f][k] == ie) { return 3 * f + k; }
            }
            usize::MAX
        });

        if twin.iter().any(|v| v == &usize::MAX) {
            return Err("Input mesh must not contain boundary edges.".into());
        }

        let half = (0..nh).collect();
        let fns = par::tabulate(nf, |i| {
            let p2 = pos[idx[i][1]];
            let p1 = pos[idx[i][0]];
            let p0 = pos[idx[i][2]];
            let x = p2 - p1;
            let t = (p1 - p0) * -1.;
            x.cross(t).normalize()
        });
        let vns = vert_normals(pos, idx, &fns);

        Ok(Hmesh{ nv, nf, nh, twin, head, tail, half, vns, fns })
    }
}

// Angle-weighted; each vertex sums its corners in face order, so threads do not change the sum.
fn vert_normals(pos: &[Vec3], idx: &[Vec3u], fns: &[Vec3]) -> Vec<Vec3> {
    let nv = pos.len();
    let mut start = vec![0usize; nv + 1];
    for f in idx { for j in 0..3 { start[f[j] + 1] += 1; } }
    for v in 0..nv { start[v + 1] += start[v]; }
    let mut fill = start.clone();
    let mut corners = vec![0u32; 3 * idx.len()];
    for (i, f) in idx.iter().enumerate() {
        for j in 0..3 {
            corners[fill[f[j]]] = (3 * i + j) as u32;
            fill[f[j]] += 1;
        }
    }

    par::tabulate(nv, |v| {
        let mut n = Vec3::ZERO;
        for &c in &corners[start[v]..start[v + 1]] {
            let (i, j) = (c as usize / 3, c as usize % 3);
            let v_prev = pos[idx[i][(j + 2) % 3]];
            let v_curr = pos[v];
            let v_next = pos[idx[i][(j + 1) % 3]];
            let e_curr = (v_next - v_curr).normalize();
            let e_prev = (v_curr - v_prev).normalize();
            if e_curr.is_nan() || e_prev.is_nan() { continue; }
            let dot = -e_prev.dot(e_curr);
            let phi = if      dot >=  1. { 0. }
                      else if dot <= -1. { PI as Real }
                      else               { dot.acos() };
            n += fns[i] * phi;
        }
        n.normalize_or_zero()
    })
}
