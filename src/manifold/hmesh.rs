//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

use std::f64::consts::PI;
use crate::{par, Vec3, Vec3u, Real};

/// Hmesh preserves the order of pos and idx in any cases.
#[derive(Debug, Clone)]
pub(in crate::manifold) struct Hmesh {
    pub twin: Vec<usize>,
    pub vns: Vec<Vec3>,
    pub fns: Vec<Vec3>,
}

// The halfedge across each corner's edge. Corners are sorted by their edge, then face and
// side; within an edge's run they pair off two by two in that order, and a run left odd has a
// corner with nothing across it.
fn twins(pos: &[Vec3], idx: &[Vec3u]) -> Result<Vec<usize>, String> {
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
    let corner = |k: usize| { let f = ett[k].1; 3 * (f >> 2) as usize + (f & 3) as usize };

    let starts = par::flat_tabulate(ett.len(), |k| (k == 0 || ett[k].0 != ett[k - 1].0).then_some(k));
    let run_end = |r: usize| starts.get(r + 1).copied().unwrap_or(ett.len());
    if !par::all(&starts, |r, &s| run_end(r) - s <= 2) {
        return Err("Input mesh has an edge shared by more than two faces.".into());
    }

    let mut twin = vec![usize::MAX; ett.len()];
    let slots = par::Shared::new(&mut twin);
    par::for_each(starts.len(), |r| {
        let (s, e) = (starts[r], run_end(r));
        for k in (s..e - 1).step_by(2) {
            // SAFETY: every corner sits in one run and one place in it.
            unsafe {
                slots.set(corner(k), corner(k + 1));
                slots.set(corner(k + 1), corner(k));
            }
        }
    });
    Ok(twin)
}

impl Hmesh {
    pub fn new(
        pos: &[Vec3],
        idx: &[Vec3u],
    ) -> Result<Self, String> {
        let mut laps = crate::laps::Laps::new("topology");
        let twin = twins(pos, idx)?;
        laps.lap("twins");
        if twin.iter().any(|v| v == &usize::MAX) {
            return Err("Input mesh must not contain boundary edges.".into());
        }

        let fns = face_normals(pos, idx);
        laps.lap("fns");
        let vns = vert_normals(pos, idx, &fns);
        laps.lap("vns");

        Ok(Hmesh{ twin, vns, fns })
    }
}

pub(in crate::manifold) fn face_normals(pos: &[Vec3], idx: &[Vec3u]) -> Vec<Vec3> {
    par::map(idx, |f| {
        let p2 = pos[f[1] as usize];
        let p1 = pos[f[0] as usize];
        let p0 = pos[f[2] as usize];
        let x = p2 - p1;
        let t = (p1 - p0) * -1.;
        x.cross(t).normalize()
    })
}

// Angle-weighted; each vertex sums its corners in face order, so threads do not change the sum.
pub(in crate::manifold) fn vert_normals(pos: &[Vec3], idx: &[Vec3u], fns: &[Vec3]) -> Vec<Vec3> {
    let nv = pos.len();
    let mut start = vec![0usize; nv + 1];
    for f in idx { for j in 0..3 { start[(f[j] + 1) as usize] += 1; } }
    for v in 0..nv { start[v + 1] += start[v]; }
    let mut fill = start.clone();
    let mut corners = vec![0u32; 3 * idx.len()];
    for (i, f) in idx.iter().enumerate() {
        for j in 0..3 {
            corners[fill[f[j] as usize]] = (3 * i + j) as u32;
            fill[f[j] as usize] += 1;
        }
    }

    par::tabulate(nv, |v| {
        let mut n = Vec3::ZERO;
        for &c in &corners[start[v]..start[v + 1]] {
            let (i, j) = (c as usize / 3, c as usize % 3);
            let v_prev = pos[(idx[i][(j + 2) % 3]) as usize];
            let v_curr = pos[v];
            let v_next = pos[(idx[i][(j + 1) % 3]) as usize];
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
