// Meshes the benchmark and the golden example share: closed, with known shapes.
#![allow(dead_code)]

use boolmesh::prelude::*;

pub struct Soup {
    pub pos: Vec<f64>,
    pub idx: Vec<usize>,
}

impl Soup {
    pub fn manifold(&self) -> Manifold { Manifold::new(&self.pos, &self.idx).expect("the shapes are closed") }

    pub fn join(parts: &[Soup]) -> Soup {
        let mut out = Soup { pos: vec![], idx: vec![] };
        for p in parts {
            let base = out.pos.len() / 3;
            out.pos.extend_from_slice(&p.pos);
            out.idx.extend(p.idx.iter().map(|i| i + base));
        }
        out
    }

    pub fn faces(&self) -> usize { self.idx.len() / 3 }
}

/// A latitude-longitude sphere: `2 * stacks * (slices - 1)` faces.
pub fn ball(c: [f64; 3], r: f64, stacks: usize, slices: usize) -> Soup {
    let mut pos = vec![c[0], c[1], c[2] + r, c[0], c[1], c[2] - r];
    for i in 1..stacks {
        let phi = std::f64::consts::PI * i as f64 / stacks as f64;
        for j in 0..slices {
            let th = 2. * std::f64::consts::PI * j as f64 / slices as f64;
            pos.extend_from_slice(&[c[0] + r * phi.sin() * th.cos(), c[1] + r * phi.sin() * th.sin(), c[2] + r * phi.cos()]);
        }
    }
    let ring = |i: usize, j: usize| 2 + (i - 1) * slices + j % slices;
    let mut idx = vec![];
    for j in 0..slices {
        idx.extend_from_slice(&[0, ring(1, j), ring(1, j + 1)]);
        idx.extend_from_slice(&[1, ring(stacks - 1, j + 1), ring(stacks - 1, j)]);
    }
    for i in 1..stacks - 1 {
        for j in 0..slices {
            let (a, b, c2, d) = (ring(i, j), ring(i, j + 1), ring(i + 1, j), ring(i + 1, j + 1));
            idx.extend_from_slice(&[a, c2, d, a, d, b]);
        }
    }
    Soup { pos, idx }
}

pub fn cuboid(lo: [f64; 3], hi: [f64; 3]) -> Soup {
    let mut pos = vec![];
    for i in 0..8 {
        pos.push(if i & 1 == 0 { lo[0] } else { hi[0] });
        pos.push(if i & 2 == 0 { lo[1] } else { hi[1] });
        pos.push(if i & 4 == 0 { lo[2] } else { hi[2] });
    }
    let idx = vec![0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5];
    Soup { pos, idx }
}

/// A 10 x 10 grid of thin square bars, all the way through a ball of radius 20 at the origin.
pub fn prisms() -> Soup {
    let mut parts = vec![];
    for i in 0..10 {
        for j in 0..10 {
            let (x, y) = (-15. + 3. * i as f64, -15. + 3. * j as f64);
            parts.push(cuboid([x, y, -30.], [x + 1., y + 1., 30.]));
        }
    }
    Soup::join(&parts)
}

/// Seven blocks sunk into the top of a ball of radius 20 at the origin.
pub fn letters() -> Soup {
    Soup::join(&(0..7).map(|l| {
        let x = -10.5 + 3. * l as f64;
        cuboid([x, -3., 19.], [x + 2., 3., 21.])
    }).collect::<Vec<_>>())
}

/// A ball of `stacks` x `stacks` with the stacks to its faces.
pub fn dense_ball(stacks: usize) -> Soup { ball([0.; 3], 20., stacks, stacks) }

/// The ball's faces for a target face count, roughly.
pub fn stacks_for(faces: usize) -> usize { ((faces as f64 / 2.).sqrt()).round() as usize }

/// The ball with every vertex moved off it by up to `amount`, so no two faces share a plane.
pub fn jittered(mut s: Soup, amount: f64) -> Soup {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    for c in s.pos.iter_mut() {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        *c += amount * ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5);
    }
    s
}
