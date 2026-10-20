//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

#[cfg(feature = "f32")]
mod precision {
    pub type Vec2 = glam::Vec2;
    pub type Vec3 = glam::Vec3A;
    pub type Vec4 = glam::Vec4;
    pub type Mat3 = glam::Mat3A;
    pub type Real = f32;
    pub const K_PRECISION: Real = 1e-4;
}

#[cfg(not(feature = "f32"))]
mod precision {
    pub type Vec2 = glam::DVec2;
    pub type Vec3 = glam::DVec3;
    pub type Vec4 = glam::DVec4;
    pub type Mat3 = glam::DMat3;
    pub type Real = f64;
    pub const K_PRECISION: f64 = 1e-12;
}

pub type Vec3u = glam::UVec3;
pub use precision::{Real, Vec2, Vec3, Vec4, Mat3, K_PRECISION};
pub const K_BEST: Real = Real::MIN;


#[derive(PartialEq)]
pub enum OpType { Add, Subtract, Intersect }

/// What an index field holds when it points nowhere.
pub const NONE: u32 = u32::MAX;

// Indices are `u32`, half of what `usize` takes: a mesh past four billion halfedges is
// refused when it is read.
#[derive(Clone, Copy, Debug)]
pub struct Half {
    pub tail: u32,
    pub head: u32,
    pub pair: u32,
    pub prop: u32,
}

impl Default for Half {
    fn default() -> Self { Self { tail: NONE, head: NONE, pair: NONE, prop: NONE } }
}

/// `i` as a stored index; `usize::MAX`, and `NONE` read back as a `usize`, are `NONE`.
pub fn id(i: usize) -> u32 { if i == usize::MAX { NONE } else { debug_assert!(i <= NONE as usize); i as u32 } }

impl Half {
    pub fn new(tail: usize, head: usize, pair: usize, prop: usize) -> Self { Self { tail: id(tail), head: id(head), pair: id(pair), prop: id(prop) } }
    pub fn tmp(tail: usize, head: usize) -> Self { Self { tail: id(tail), head: id(head), pair: NONE, prop: NONE } }
    pub fn tail(&self) -> Option<usize> { if self.tail == NONE {None} else {Some(self.tail as usize)} }
    pub fn head(&self) -> Option<usize> { if self.head == NONE {None} else {Some(self.head as usize)} }
    pub fn pair(&self) -> Option<usize> { if self.pair == NONE {None} else {Some(self.pair as usize)} }
    pub fn is_fwd(&self) -> bool { self.tail < self.head }
}

pub fn face_of(hid: usize) -> usize { hid / 3 }
pub fn next_of(hid: usize) -> usize { let mut i = hid + 1; if i.is_multiple_of(3) { i -= 3;} i }


#[derive(Clone, Debug, Copy)]
pub struct Tref {
    pub mid: u32, // mesh id
    pub fid: u32, // face id
    pub pid: i32, // planer id
}

impl Default for Tref {
    fn default() -> Self {
        Self {
            mid: NONE,
            fid: NONE,
            pid: -1
        }
    }
}

pub fn det2x2(a: &Vec2, b: &Vec2) -> Real { a.x * b.y - a.y * b.x }

pub fn get_aa_proj_matrix(n: &Vec3) -> (Vec3, Vec3) {
    let a = n.abs();
    let m: Real;
    let r1: Vec3;
    let r2: Vec3;

    if a.z > a.x && a.z > a.y { r1 = Vec3::new(1., 0., 0.); r2 = Vec3::new(0., 1., 0.); m = n.z; } // preserve x, y
    else if a.y > a.x         { r1 = Vec3::new(0., 0., 1.); r2 = Vec3::new(1., 0., 0.); m = n.y; } // preserve z, x
    else                      { r1 = Vec3::new(0., 1., 0.); r2 = Vec3::new(0., 0., 1.); m = n.x; } // preserve y, z

    if m < 0. { (-r1, r2) } else { (r1, r2) }
}

pub fn compute_aa_proj(p: &(Vec3, Vec3), v: &Vec3) -> Vec2 {
    Vec2::new(p.0.dot(*v), p.1.dot(*v))
}

pub fn is_ccw_2d(p0: &Vec2, p1: &Vec2, p2: &Vec2, t: Real) -> i32 {
    let v1 = p1 - p0;
    let v2 = p2 - p0;
    let area = v1.x * v2.y - v1.y * v2.x;
    let base = v1.length_squared().max(v2.length_squared());
    if area.powi(2) * 4. <= base * t.powi(2) { return 0; }
    if area > 0. { 1 } else { -1 }
}

pub fn is_ccw_3d(p0: &Vec3, p1: &Vec3, p2: &Vec3, n: &Vec3, t: Real) -> i32 {
    let p = get_aa_proj_matrix(n);
    is_ccw_2d(
        &compute_aa_proj(&p, p0),
        &compute_aa_proj(&p, p1),
        &compute_aa_proj(&p, p2),
        t
    )
}

pub fn safe_normalize(v: Vec2) -> Vec2 {
    let n = v.normalize();
    if n.x.is_finite() && !n.x.is_nan() &&
       n.y.is_finite() && !n.y.is_nan() { n }
    else { Vec2::new(0., 0.) }
}
