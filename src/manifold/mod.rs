//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

pub mod hmesh;
pub mod bounds;
pub mod collider;

use std::cmp::Ordering;
use bounds::BBox;
use crate::collider::{morton_code, MortonCollider, K_NO_CODE};
use crate::{par, Data, Real, Half, Vec3, Vec3u, K_PRECISION, next_of};
use super::hmesh::Hmesh;
#[cfg(feature = "rayon")] use rayon::prelude::*;

// The core struct for all boolean operations. Avoid modifying properties directly, as the struct
// maintains internal face index sorting based on vertex positions—a prerequisite for all subsequent operations.
#[derive(Clone, Debug)]
pub struct Manifold<S = Real>  {
    pub ps: Vec<Vec3>,            // positions
    pub hs: Vec<Half>,            // halfedges
    pub nv: usize,                // number of vertices
    pub nf: usize,                // number of faces
    pub nh: usize,                // number of halfedges
    pub bb: BBox,                 // bounding box
    pub eps: Real,                // epsilon
    pub tol: Real,                // tolerance
    pub fns: Vec<Vec3>,           // face normals
    pub vns: Vec<Vec3>,           // vert normals
    pub vals: Vec<S>,             // values on each corner (num of faces * 3)
    pub collider: MortonCollider, //
    pub coplanar: Vec<i32>,       // indices of coplanar faces
}

impl Manifold {
    pub fn new(pos: &[f64], idx: &[usize]) -> Result<Self, String> { Self::new_with_values(pos, idx, &[]) }
}

impl<S: Data> Manifold<S> {
    pub fn new_with_values(pos: &[f64], idx: &[usize], val: &[S]) -> Result<Self, String> {
        if !pos.len().is_multiple_of(3) { return Err("pos must be a multiple of 3".into()); }
        if !idx.len().is_multiple_of(3) { return Err("idx must be a multiple of 3".into()); }
        if !val.is_empty() && val.len() != idx.len() { return Err("val must hold one value per corner".into()); }

        let (weld, rmap) = weld_verts(pos);

        // create halfedge container
        let hs = par::flat_tabulate(idx.len() / 3, |t| {
            let v = [rmap[idx[3 * t]], rmap[idx[3 * t + 1]], rmap[idx[3 * t + 2]]];
            let keep = v[0] != v[1] && v[1] != v[2] && v[2] != v[0];
            keep.then(|| [
                Half::new(v[0], v[1], usize::MAX, 3 * t),
                Half::new(v[1], v[2], usize::MAX, 3 * t + 1),
                Half::new(v[2], v[0], usize::MAX, 3 * t + 2),
            ]).into_iter().flatten()
        });

        let mut mfd = Self::new_impl(weld, hs, val.to_vec(), None, None)?;
        if !mfd.vals.is_empty() { dedupe_values(&mut mfd.hs, &mfd.vals); }
        Ok(mfd)
    }

    pub(crate) fn new_impl(
        ps: Vec<Vec3>,
        hs: Vec<Half>,
        vs: Vec<S>,
        eps: Option<Real>,
        tol: Option<Real>,
    ) -> Result<Self, String> {
        let bb = BBox::new(None, &ps);
        let idx = par::tabulate(hs.len() / 3, |f| Vec3u::new(hs[3 * f].tail, hs[3 * f + 1].tail, hs[3 * f + 2].tail));
        let (mut f_bb, mut f_mt) = compute_face_morton(&ps, &idx, &bb);
        let (hm, map) = sort_faces(&ps, &idx, &mut f_bb, &mut f_mt)?;
        let prop = |i: usize| hs[3 * map[i / 3] + i % 3].prop;
        let hs = par::map(&hm.half, |&i| Half::new(hm.tail[i], hm.head[i], hm.twin[i], if vs.is_empty() { hm.tail[i] } else { prop(i) }));
        let vals = if vs.is_empty() { vec![] } else { par::map(&hs, |h| vs[h.prop].clone()) };

        let mut e = K_PRECISION * bb.scale();
        e = if e.is_finite() { e } else { -1. };
        let eps = if let Some(e_) = eps { e_ } else { e };
        let tol = if let Some(t_) = tol { t_ } else { e };
        let collider = MortonCollider::new(&f_bb, &f_mt);
        let coplanar = compute_coplanar_idx(&ps, &hm.fns, &hs, eps);

        let mfd = Manifold { hs, ps, vals, nv: hm.nv, nf: hm.nf, nh: hm.nh, bb, vns: hm.vns, fns: hm.fns, eps, tol, collider, coplanar };
        if !mfd.is_manifold() { return Err("The input mesh is not manifold".into()); }
        Ok(mfd)
    }

    pub fn is_manifold(&self) -> bool {
        par::all(&self.hs, |i, h| {
            if h.tail().is_none() || h.head().is_none() { return true; }
            match h.pair() {
                None => { false },
                Some(pair) => {
                    let mut good = true;
                    good &= self.hs[pair].pair() == Some(i);
                    good &= h.tail != h.head;
                    good &= h.tail == self.hs[pair].head;
                    good &= h.head == self.hs[pair].tail;
                    good
                }
            }
        })
    }
}

// Merges bit-identical positions, numbering the kept ones in order of first appearance.
fn weld_verts(pos: &[f64]) -> (Vec<Vec3>, Vec<usize>) {
    let n = pos.len() / 3;
    let ps = par::tabulate(n, |i| Vec3::new(pos[3 * i] as Real, pos[3 * i + 1] as Real, pos[3 * i + 2] as Real));
    let mut keyed = par::tabulate(n, |i| ((ps[i].x.to_bits(), ps[i].y.to_bits(), ps[i].z.to_bits()), i));
    par::sort_unique(&mut keyed);

    // The lowest index of each position, which sorts first among its equals.
    let mut first = vec![0; n];
    for run in keyed.chunk_by(|a, b| a.0 == b.0) {
        for &(_, i) in run { first[i] = run[0].1; }
    }
    drop(keyed);

    let mut weld = Vec::with_capacity(n);
    let mut rmap = vec![0; pos.len()];
    for i in 0..n {
        if first[i] == i { rmap[i] = weld.len(); weld.push(ps[i]); }
        else { rmap[i] = rmap[first[i]]; }
    }
    (weld, rmap)
}

fn compute_face_morton(
    pos: &[Vec3],
    idx: &[Vec3u],
    bb: &BBox
) -> (Vec<BBox>, Vec<u32>) {
    let n = idx.len();
    let mut bbs = vec![BBox::default(); n];
    let mut mts = vec![0; n];

    #[cfg(feature = "rayon")] {
        bbs.par_iter_mut()
            .zip(mts.par_iter_mut())
            .zip(idx.par_iter())
            .for_each(|((bb_, mt_), f)| {
                let p0 = pos[f.x];
                let p1 = pos[f.y];
                let p2 = pos[f.z];
                bb_.union(&p0);
                bb_.union(&p1);
                bb_.union(&p2);
                *mt_ = morton_code(&((p0 + p1 + p2) / 3.), bb);
            });
    }

    #[cfg(not(feature = "rayon"))] {
        for (i, f) in idx.iter().enumerate() {
            let p0 = pos[f.x];
            let p1 = pos[f.y];
            let p2 = pos[f.z];
            bbs[i].union(&p0);
            bbs[i].union(&p1);
            bbs[i].union(&p2);
            mts[i] = morton_code(&((p0 + p1 + p2) / 3.), bb);
        }
    }


    (bbs, mts)
}

fn sort_faces(
    pos: &[Vec3],
    idx: &[Vec3u],
    face_bboxes: &mut Vec<BBox>,
    face_morton: &mut Vec<u32>
) -> Result<(Hmesh, Vec<usize>), String> {
    let mut map = (0..face_morton.len()).collect::<Vec<_>>();
    par::sort_by_key(&mut map, |&i| face_morton[i]);
    *face_bboxes = par::map(&map, |&i| face_bboxes[i].clone());
    *face_morton = par::map(&map, |&i| face_morton[i]);
    Ok((Hmesh::new(pos, &par::map(&map, |&i| idx[i]))?, map))
}

fn compute_coplanar_idx(
    ps: &[Vec3],
    ns: &[Vec3],
    hs: &[Half],
    tol: Real
) -> Vec<i32> {
    let nt = hs.len() / 3;
    let mut res = vec![-1; nt];
    let mut priority = par::tabulate(nt, |t| {
        let i = t * 3;
        let area = if hs[i].tail().is_none() { 0.} else {
            let p0 = ps[hs[i].tail];
            let p1 = ps[hs[i].head];
            let p2 = ps[hs[i + 1].head];
            (p1 - p0).cross(p2 - p0).length_squared()
        };
        (area, t)
    });

    par::sort_by(&mut priority, |a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

    let mut interior = vec![];
    for (_, t) in priority.iter() {
        if res[*t] != -1 { continue; }
        res[*t] = *t as i32;

        let i = t * 3;
        let p = ps[hs[i].tail];
        let n = ns[*t];

        interior.clear();
        interior.extend_from_slice(&[i, i + 1, i + 2]);

        while let Some(hi) = interior.pop() {
            let h1 = next_of(hs[hi].pair);
            let t1 = h1 / 3;

            if res[t1] != -1 { continue; }

            if (ps[hs[h1].head] - p).dot(n).abs() < tol {
                res[t1] = *t as i32;
                if interior.last().copied() == Some(hs[h1].pair) { interior.pop(); }
                else { interior.push(h1); }
                interior.push(next_of(h1));
            }
        }
    }
    res
}

pub(crate) fn cleanup_unused_verts(
    ps: &mut Vec<Vec3>,
    hs: &mut Vec<Half>
) {
    let bb = BBox::new(None, ps);
    let mt = par::map(ps, |p| morton_code(p, &bb));

    let mut new2old = (0..ps.len()).collect::<Vec<_>>();
    let mut old2new = vec![0; ps.len()];
    par::sort_by_key(&mut new2old, |&i| mt[i]);
    for (new, &old) in new2old.iter().enumerate() { old2new[old] = new; }

    // reindex verts
    par::for_each_mut(hs, |_, h| {
        if h.pair().is_none() { return; }
        h.tail = old2new[h.tail];
        h.head = old2new[h.head];
    });

    // truncate pos container
    let nv = new2old
        .iter()
        .position(|&v| mt[v] >= K_NO_CODE)
        .unwrap_or(new2old.len());

    new2old.truncate(nv);

    *ps = par::map(&new2old, |&i| ps[i]);
    *hs = hs.iter().filter(|h| h.pair().is_some()).cloned().collect();
}

fn dedupe_values<S: Data>(hs: &mut [Half], vs: &[S]) {
    let mut root = (0..=hs.iter().map(|h| h.prop).max().unwrap_or(0)).collect::<Vec<_>>();
    fn find(r: &mut [usize], mut i: usize) -> usize { while r[i] != i { r[i] = r[r[i]]; i = r[i]; } i }

    for e in 0..hs.len() {
        if hs[e].pair == usize::MAX { continue; }
        let f = next_of(hs[e].pair);
        if vs[e] == vs[f] { let (a, b) = (find(&mut root, hs[e].prop), find(&mut root, hs[f].prop)); root[a.min(b)] = a.max(b); }
    }

    for h in hs.iter_mut() { h.prop = find(&mut root, h.prop); }
}

