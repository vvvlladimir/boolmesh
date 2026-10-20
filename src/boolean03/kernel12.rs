//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

use crate::{par, Manifold, Half, Real, Vec3};
use crate::bounds::{BBox, Query};
use super::kernel01::intersect;
use super::kernel02::Kernel02;
use super::kernel11::Kernel11;

pub struct Kernel12<'a> {
    pub hs_p: &'a[Half],
    pub hs_q: &'a[Half],
    pub ps_p: &'a[Vec3],
    pub k02: Kernel02<'a>,
    pub k11: Kernel11<'a>,
    pub fwd: bool,
}

impl<'a> Kernel12<'a> {
    pub fn op (&self, p1: usize, q2: usize) -> Option<(i32, Vec3)> {
        let mut x12 = 0;
        let mut xzy_lr0 = [Vec3::ZERO; 2];
        let mut xzy_lr1 = [Vec3::ZERO; 2];
        let mut shadow_ = false;
        let h = self.hs_p[p1].clone();

        let mut k = 0;

        for vid in [h.tail, h.head].iter() {
            if let Some((s, z)) = self.k02.op((*vid) as usize, q2) {
                let f = (*vid == h.tail) == self.fwd;
                x12 += s * if f { 1 } else { -1 };
                if k < 2 && (k == 0 || (s != 0) != shadow_) {
                    shadow_ = s != 0;
                    xzy_lr0[k] = self.ps_p[(*vid) as usize];
                    let temp = xzy_lr0[k].y;
                    xzy_lr0[k].y = xzy_lr0[k].z;
                    xzy_lr0[k].z = temp;
                    xzy_lr1[k] = xzy_lr0[k];
                    xzy_lr1[k].y = z;
                    k += 1;
                }
            }
        }

        for i in 0..3 {
            let q1 = 3 * q2 + i;
            let h = &self.hs_q[q1];
            let q1f = if h.is_fwd() { q1 } else { h.pair as usize };
            let op = if self.fwd { self.k11.op(p1, q1f) } else { self.k11.op(q1f, p1) };
            if let Some((s, xyzz)) = op {
                x12 -= s * if h.is_fwd() { 1 } else { -1 };
                if k < 2 && (k == 0 || (s != 0) != shadow_) {
                    shadow_ = s != 0;
                    xzy_lr0[k].x = xyzz.x;
                    xzy_lr0[k].y = xyzz.z;
                    xzy_lr0[k].z = xyzz.y;
                    xzy_lr1[k] = xzy_lr0[k];
                    xzy_lr1[k].y = xyzz.w;
                    if !self.fwd { std::mem::swap(&mut xzy_lr0[k].y, &mut xzy_lr1[k].y); }
                    k += 1;
                }
            }
        }

        if x12 == 0 { return None }

        assert_eq!(k, 2, "Boolean manifold error: v12");
        let xzyy = intersect(xzy_lr0[0], xzy_lr0[1], xzy_lr1[0], xzy_lr1[1]);
        Some((x12, Vec3::new(xzyy[0], xzyy[2], xzyy[1])))
    }
}

pub fn intersect12<S> (
    mp: &Manifold<S>,
    mq: &Manifold<S>,
    p1q2: &mut Vec<[usize; 2]>,
    expand: Real,
    fwd: bool
) -> (Vec<i32>, Vec<Vec3>) {
    let ma = if fwd { mp } else { mq };
    let mb = if fwd { mq } else { mp };

    let k02 = Kernel02{ ps_p: &ma.ps, ps_q: &mb.ps, hs_q: &mb.hs, ns: &mp.vns, expand, fwd };
    let k11 = Kernel11{ ps_p: &mp.ps, ps_q: &mq.ps, hs_p: &mp.hs, hs_q: &mq.hs, ns: &mp.vns, expand };
    let k12 = Kernel12{ ps_p: &ma.ps, hs_p: &ma.hs, hs_q: &mb.hs, fwd, k02, k11 };

    let bbs = ma.hs.iter()
        .enumerate()
        .filter(|(_, h)| h.is_fwd())
        .map(|(i, h)| Query::Bb(BBox::new(Some(i), &[ma.ps[h.tail as usize], ma.ps[h.head as usize]])))
        .collect::<Vec<Query>>();

    let mut hits = mb.collider.collision_map(&bbs, |a, b| {
        k12.op(a, b).map(|(x, v)| (if fwd { [a, b] } else { [b, a] }, x, v))
    });
    par::sort_by_key(&mut hits, |h| h.0);

    p1q2.extend(hits.iter().map(|h| h.0));
    let x12 = hits.iter().map(|h| h.1).collect();
    let v12 = hits.iter().map(|h| h.2).collect();
    (x12, v12)
}