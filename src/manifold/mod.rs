//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

pub mod hmesh;
pub mod bounds;
pub mod collider;

use std::sync::OnceLock;
use bounds::BBox;
use crate::collider::{morton_code, MortonCollider, K_NO_CODE};
use crate::{par, id, Data, Real, Half, Vec3, Vec3u, K_PRECISION, next_of};
use super::hmesh::{Hmesh, face_normals, vert_normals};

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
    coplanar: OnceLock<Vec<i32>>, // see `coplanar()`
}

impl Manifold {
    pub fn new(pos: &[f64], idx: &[usize]) -> Result<Self, String> { Self::new_with_values(pos, idx, &[]) }
}

impl<S> Manifold<S> {
    /// Which faces lie in one plane: each face holds the index of the face its group grew from.
    /// Worked out the first time a boolean needs it, since a result read only for its mesh never does.
    pub fn coplanar(&self) -> &[i32] {
        self.coplanar.get_or_init(|| compute_coplanar_idx(&self.ps, &self.fns, &self.hs, self.eps))
    }
}

impl<S: Data> Manifold<S> {
    pub fn new_with_values(pos: &[f64], idx: &[usize], val: &[S]) -> Result<Self, String> {
        if !pos.len().is_multiple_of(3) { return Err("pos must be a multiple of 3".into()); }
        if !idx.len().is_multiple_of(3) { return Err("idx must be a multiple of 3".into()); }
        if !val.is_empty() && val.len() != idx.len() { return Err("val must hold one value per corner".into()); }
        let nv = pos.len() / 3;
        if idx.iter().any(|&i| i >= nv) { return Err("idx must refer to vertices in pos".into()); }

        let mut laps = crate::laps::Laps::new("read");
        let (weld, rmap) = weld_verts(pos)?;
        laps.lap("weld");

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

        laps.lap("halfedges");
        let mut mfd = Self::new_impl(weld, hs, val.to_vec(), None, None)?;
        if !mfd.vals.is_empty() { dedupe_values(&mut mfd.hs, &mfd.vals); }
        Ok(mfd)
    }

    /// A manifold from triangles whose halfedges are not paired yet.
    pub(crate) fn new_impl(
        ps: Vec<Vec3>,
        hs: Vec<Half>,
        vs: Vec<S>,
        eps: Option<Real>,
        tol: Option<Real>,
    ) -> Result<Self, String> {
        let mut laps = crate::laps::Laps::new("finish");
        let bb = BBox::new(None, &ps);
        let idx = par::tabulate(hs.len() / 3, |f| Vec3u::new(hs[3 * f].tail, hs[3 * f + 1].tail, hs[3 * f + 2].tail));
        let (mut f_bb, mut f_mt) = compute_face_morton(&ps, &idx, &bb);
        laps.lap("morton");
        let (hm, idx, map) = sort_faces(&ps, &idx, &mut f_bb, &mut f_mt)?;
        laps.lap("sort+topology");
        let hs = par::tabulate(3 * idx.len(), |i| {
            let tail = idx[i / 3][i % 3];
            let prop = if vs.is_empty() { tail } else { hs[3 * map[i / 3] + i % 3].prop };
            Half { tail, head: idx[i / 3][(i + 1) % 3], pair: id(hm.twin[i]), prop }
        });
        laps.lap("halfedges");
        Self::finished(ps, hs, &vs, hm.fns, hm.vns, bb, f_bb, f_mt, eps, tol, laps)
    }

    /// The parts side by side as one manifold, with no boolean between them: for parts whose
    /// bounding boxes do not meet, which is the union. Their epsilon and tolerance are the
    /// largest of theirs; values come along only when every part has them.
    pub fn compose(parts: &[&Self]) -> Result<Self, String> {
        let with_values = parts.iter().filter(|m| !m.vals.is_empty()).count();
        if with_values != 0 && with_values != parts.len() { return Err("values on some parts only".into()); }
        let (mut ps, mut hs, mut vs) = (vec![], vec![], vec![]);
        for m in parts {
            let (dv, dh) = (ps.len(), hs.len());
            ps.extend_from_slice(&m.ps);
            hs.extend(m.hs.iter().map(|h| Half::new(h.tail as usize + dv, h.head as usize + dv, h.pair as usize + dh, h.prop as usize + dh)));
            vs.extend_from_slice(&m.vals);
        }
        let eps = parts.iter().map(|m| m.eps).fold(Real::MIN, Real::max);
        let tol = parts.iter().map(|m| m.tol).fold(Real::MIN, Real::max);
        // Without values a corner's property is its vertex; with them, the corner itself.
        if vs.is_empty() { for h in hs.iter_mut() { h.prop = h.tail; } }
        else { for (i, h) in hs.iter_mut().enumerate() { h.prop = id(i); } }
        Self::from_paired(ps, hs, vs, Some(eps), Some(tol))
    }

    /// A manifold from a boolean's halfedges, already paired: its faces are put in Morton
    /// order and the pairs follow them, with no edge table built again.
    pub(crate) fn from_paired(
        ps: Vec<Vec3>,
        hs: Vec<Half>,
        vs: Vec<S>,
        eps: Option<Real>,
        tol: Option<Real>,
    ) -> Result<Self, String> {
        let mut laps = crate::laps::Laps::new("finish");
        if ps.is_empty() { return Err("empty pos matrix".into()); }
        if hs.is_empty() { return Err("empty idx matrix".into()); }
        if ps.len() > u32::MAX as usize || hs.len() > u32::MAX as usize { return Err("mesh too large".into()); }
        if hs.iter().any(|h| h.pair().is_none()) { return Err("Input mesh must not contain boundary edges.".into()); }
        let bb = BBox::new(None, &ps);
        let nf = hs.len() / 3;
        let idx = par::tabulate(nf, |f| Vec3u::new((hs[3 * f].tail as usize) as u32, (hs[3 * f + 1].tail as usize) as u32, (hs[3 * f + 2].tail as usize) as u32));
        let (f_bb, f_mt) = compute_face_morton(&ps, &idx, &bb);
        laps.lap("morton");
        let mut map = (0..nf).collect::<Vec<_>>();
        par::sort_by_key(&mut map, |&i| f_mt[i]);
        let mut inv = vec![0; nf];
        for (new, &old) in map.iter().enumerate() { inv[old] = new; }
        let f_bb = par::map(&map, |&i| f_bb[i].clone());
        let f_mt = par::map(&map, |&i| f_mt[i]);
        let idx = par::map(&map, |&i| idx[i]);
        laps.lap("sort");
        let hs = par::tabulate(3 * nf, |i| {
            let h = &hs[3 * map[i / 3] + i % 3];
            let pair = 3 * inv[(h.pair / 3) as usize] + ((h.pair % 3) as usize);
            Half::new(h.tail as usize, h.head as usize, pair, if vs.is_empty() { h.tail as usize } else { h.prop as usize })
        });
        let fns = face_normals(&ps, &idx);
        let vns = vert_normals(&ps, &idx, &fns);
        laps.lap("halfedges");
        Self::finished(ps, hs, &vs, fns, vns, bb, f_bb, f_mt, eps, tol, laps)
    }

    fn finished(
        ps: Vec<Vec3>,
        hs: Vec<Half>,
        vs: &[S],
        fns: Vec<Vec3>,
        vns: Vec<Vec3>,
        bb: BBox,
        f_bb: Vec<BBox>,
        f_mt: Vec<u32>,
        eps: Option<Real>,
        tol: Option<Real>,
        mut laps: crate::laps::Laps,
    ) -> Result<Self, String> {
        let vals = if vs.is_empty() { vec![] } else { par::map(&hs, |h| vs[h.prop as usize].clone()) };
        let mut e = K_PRECISION * bb.scale();
        e = if e.is_finite() { e } else { -1. };
        let eps = if let Some(e_) = eps { e_ } else { e };
        let tol = if let Some(t_) = tol { t_ } else { e };
        let collider = MortonCollider::new(&f_bb, &f_mt);
        laps.lap("collider");

        let (nv, nf, nh) = (ps.len(), hs.len() / 3, hs.len());
        let mfd = Manifold { hs, ps, vals, nv, nf, nh, bb, vns, fns, eps, tol, collider, coplanar: OnceLock::new() };
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
fn weld_verts(pos: &[f64]) -> Result<(Vec<Vec3>, Vec<usize>), String> {
    let n = pos.len() / 3;
    let ps = par::tabulate(n, |i| Vec3::new(pos[3 * i] as Real, pos[3 * i + 1] as Real, pos[3 * i + 2] as Real));
    if !par::all(&ps, |_, p| p.is_finite()) { return Err("pos must not contain NaN or infinite coordinates".into()); }
    let mut keyed = par::tabulate(n, |i| ((ps[i].x.to_bits(), ps[i].y.to_bits(), ps[i].z.to_bits()), i));
    par::sort_unique(&mut keyed);

    // The lowest index of each position, which sorts first among its equals.
    let mut first = vec![0; n];
    for run in keyed.chunk_by(|a, b| a.0 == b.0) {
        for &(_, i) in run { first[i] = run[0].1; }
    }
    drop(keyed);

    let mut weld = Vec::with_capacity(n);
    let mut rmap = vec![0; n];
    for i in 0..n {
        if first[i] == i { rmap[i] = weld.len(); weld.push(ps[i]); }
        else { rmap[i] = rmap[first[i]]; }
    }
    Ok((weld, rmap))
}

fn compute_face_morton(
    pos: &[Vec3],
    idx: &[Vec3u],
    bb: &BBox
) -> (Vec<BBox>, Vec<u32>) {
    let corners = |f: &Vec3u| [pos[f.x as usize], pos[f.y as usize], pos[f.z as usize]];
    let bbs = par::map(idx, |f| BBox::new(None, &corners(f)));
    let mts = par::map(idx, |f| { let [p0, p1, p2] = corners(f); morton_code(&((p0 + p1 + p2) / 3.), bb) });
    (bbs, mts)
}

fn sort_faces(
    pos: &[Vec3],
    idx: &[Vec3u],
    face_bboxes: &mut Vec<BBox>,
    face_morton: &mut Vec<u32>
) -> Result<(Hmesh, Vec<Vec3u>, Vec<usize>), String> {
    let mut map = (0..face_morton.len()).collect::<Vec<_>>();
    par::sort_by_key(&mut map, |&i| face_morton[i]);
    *face_bboxes = par::map(&map, |&i| face_bboxes[i].clone());
    *face_morton = par::map(&map, |&i| face_morton[i]);
    let idx = par::map(&map, |&i| idx[i]);
    Ok((Hmesh::new(pos, &idx)?, idx, map))
}

// Faces are taken largest first, ties by index, and each grows its group over the neighbours
// in its plane that no earlier face holds. Which neighbours lie in a face's own plane is found
// for all faces at once; a face with none grows nothing, so only the others are sorted and
// walked, and a walk passes over a face that comes earlier, as it would already hold itself.
fn compute_coplanar_idx(
    ps: &[Vec3],
    ns: &[Vec3],
    hs: &[Half],
    tol: Real
) -> Vec<i32> {
    let nt = hs.len() / 3;
    let area = par::tabulate(nt, |t| {
        let i = t * 3;
        if hs[i].tail().is_none() { return 0.; }
        let p0 = ps[hs[i].tail as usize];
        let p1 = ps[hs[i].head as usize];
        let p2 = ps[hs[i + 1].head as usize];
        (p1 - p0).cross(p2 - p0).length_squared()
    });
    let in_plane = |t: usize, hi: usize| {
        let h1 = next_of(hs[hi].pair as usize);
        (ps[hs[h1].head as usize] - ps[hs[3 * t].tail as usize]).dot(ns[t]).abs() < tol
    };
    // Bit k: the face across halfedge 3t + k lies in t's plane.
    let ring = par::tabulate(nt, |t| (0..3).fold(0u8, |bits, k| bits | (in_plane(t, 3 * t + k) as u8) << k));
    let order = |u: usize, t: usize| area[t].total_cmp(&area[u]).then(u.cmp(&t));
    let mut seeds = par::flat_tabulate(nt, |t| (ring[t] != 0).then_some((area[t], t)));
    par::sort_unique_by(&mut seeds, |a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));

    let mut laps = crate::laps::Laps::new("coplanar");
    let mut res = vec![-1; nt];
    let mut interior = vec![];
    for &(_, t) in &seeds {
        if res[t] != -1 { continue; }
        res[t] = t as i32;
        let i = t * 3;
        interior.clear();
        interior.extend_from_slice(&[i, i + 1, i + 2]);

        while let Some(hi) = interior.pop() {
            // The seed's own edges were tested above; the rest against the seed's plane here.
            if hi / 3 == t && ring[t] >> (hi % 3) & 1 == 0 { continue; }
            let h1 = next_of(hs[hi].pair as usize);
            let t1 = h1 / 3;
            if res[t1] != -1 || order(t1, t).is_lt() { continue; }
            if hi / 3 == t || in_plane(t, hi) {
                res[t1] = t as i32;
                if interior.last().copied() == Some(hs[h1].pair as usize) { interior.pop(); }
                else { interior.push(h1); }
                interior.push(next_of(h1));
            }
        }
    }
    laps.lap("walk");
    par::for_each_mut(&mut res, |t, r| if *r == -1 { *r = t as i32; });
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
        h.tail = old2new[h.tail as usize] as u32;
        h.head = old2new[h.head as usize] as u32;
    });

    // truncate pos container
    let nv = new2old
        .iter()
        .position(|&v| mt[v] >= K_NO_CODE)
        .unwrap_or(new2old.len());

    new2old.truncate(nv);

    *ps = par::map(&new2old, |&i| ps[i]);
    // Removed halfedges leave gaps; the pairs of the kept ones are renumbered past them.
    let mut kept = vec![usize::MAX; hs.len()];
    let mut n = 0;
    for (i, h) in hs.iter().enumerate() { if h.pair().is_some() { kept[i] = n; n += 1; } }
    *hs = par::flat_tabulate(hs.len(), |i| {
        let h = &hs[i];
        h.pair().map(|p| Half::new(h.tail as usize, h.head as usize, kept[p], h.prop as usize))
    });
}

fn dedupe_values<S: Data>(hs: &mut [Half], vs: &[S]) {
    let mut root = (0..=hs.iter().map(|h| h.prop as usize).max().unwrap_or(0)).collect::<Vec<_>>();
    fn find(r: &mut [usize], mut i: usize) -> usize { while r[i] != i { r[i] = r[r[i]]; i = r[i]; } i }

    for e in 0..hs.len() {
        if hs[e].pair == ((usize::MAX) as u32) { continue; }
        let f = next_of(hs[e].pair as usize);
        if vs[e] == vs[f] { let (a, b) = (find(&mut root, hs[e].prop as usize), find(&mut root, hs[f].prop as usize)); root[a.min(b)] = a.max(b); }
    }

    for h in hs.iter_mut() { h.prop = (find(&mut root, h.prop as usize)) as u32; }
}

