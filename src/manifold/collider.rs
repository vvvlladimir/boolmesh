//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

use std::sync::atomic::{AtomicU32, Ordering};
use crate::{par, Vec3};
use crate::bounds::{union_bbs, BBox, Query};

pub const K_NO_CODE: u32 = 0xFFFFFFFF;
const K_INITIAL_LENGTH: i32 = 128;
const K_LENGTH_MULTIPLE: i32 = 4;
const K_ROOT: i32 = 1;

fn spread_bits_3(v: u32) -> u32 {
    assert!(v <= 1023);
    let mut v = v;
    v = 0xFF0000FFu32 & v.wrapping_mul(0x00010001u32);
    v = 0x0F00F00Fu32 & v.wrapping_mul(0x00000101u32);
    v = 0xC30C30C3u32 & v.wrapping_mul(0x00000011u32);
    v = 0x49249249u32 & v.wrapping_mul(0x00000005u32);
    v
}

pub fn morton_code(p: &Vec3, bb: &BBox) -> u32 {
    if p.x.is_nan() { return K_NO_CODE; }
    let mut xyz = (p - bb.min) / (bb.max - bb.min);
    xyz = (1024. * xyz).max(Vec3::ZERO).min(Vec3::new(1023., 1023., 1023.));
    let x = spread_bits_3(xyz.x as u32);
    let y = spread_bits_3(xyz.y as u32);
    let z = spread_bits_3(xyz.z as u32);
    x * 4 + y * 2 + z
}

fn node2intl(node: i32) -> Option<i32> { if node % 2 == 1 { Some((node - 1) / 2) } else { None } }
fn node2leaf(node: i32) -> Option<i32> { if node % 2 == 0 { Some(node / 2) } else { None } }
fn intl2node(intl: i32) -> i32 { intl * 2 + 1 }
fn leaf2node(leaf: i32) -> i32 { leaf * 2 }
fn prefix_length(a: u32, b: u32) -> u32 { (a ^ b).leading_zeros() }

struct RadixTree<'a> {
    leaf_morton: &'a [u32],
}

impl<'a> RadixTree<'a> {
    fn prefix_length(&self, i: i32, j: i32) -> i32 {
        if j < 0 || j >= self.leaf_morton.len() as i32 { return -1; }
        let lmi = self.leaf_morton[i as usize];
        let lmj = self.leaf_morton[j as usize];
        if lmi == lmj { return 32 + prefix_length(i as u32, j as u32) as i32; }
        prefix_length(lmi, lmj) as i32
    }

   fn range_end(&self, i: i32) -> i32 {
        let mut dir = self.prefix_length(i, i + 1) - self.prefix_length(i, i - 1);
        dir = if dir > 0 { 1 } else if dir < 0 { -1 } else { 0 };

        let common = self.prefix_length(i, i - dir);
        let mut max = K_INITIAL_LENGTH;
        while self.prefix_length(i, i + dir * max) > common { max *= K_LENGTH_MULTIPLE; }

        // compute precise range length with binary search
        let mut len = 0;
        let mut stp = max / 2;
        while stp > 0 {
            if self.prefix_length(i, i + dir * (len + stp)) > common { len += stp; }
            stp /= 2;
        }
        i + dir * len
   }

    fn find_split(&self, bgn: i32, end: i32) -> i32 {
        let common = self.prefix_length(bgn, end);
        // Find the furthest object that shares more than common_prefix bits
        // with the first one, using binary search.
        let mut split = bgn;
        let mut step = end - bgn;

        loop {
            step = (step + 1) >> 1; // divide by 2, rounding up
            let new_split = split + step;
            if new_split < end && self.prefix_length(bgn, new_split) > common { split = new_split; }
            if step <= 1 { break; }
        }

        split
    }

    // The children of one internal node, found from the leaf codes alone (Karras 2012).
    fn op(&self, intl: i32) -> (i32, i32) {
        let mut bgn = intl;
        let mut end = self.range_end(bgn);
        if bgn > end { std::mem::swap(&mut bgn, &mut end); }

        let mut s = self.find_split(bgn, end);
        let child1 = if s == bgn { leaf2node(s) } else { intl2node(s) };
        s += 1;
        let child2 = if s == end { leaf2node(s) } else { intl2node(s) };

        (child1, child2)
    }
}

// Box slots written from several threads at once; see `update_boxes` for why no two touch one.
struct Slots(*mut BBox);
unsafe impl Sync for Slots {}
unsafe impl Send for Slots {}

#[derive(Clone, Debug)]
pub struct MortonCollider {
    pub node_bb: Vec<BBox>,
    pub node_parent: Vec<i32>,
    pub intl_children: Vec<(i32, i32)>,
}

impl MortonCollider {
    fn num_intl(&self) -> usize { self.intl_children.len() }
    fn num_leaf(&self) -> usize { if self.intl_children.is_empty() { 0 } else { self.num_intl() + 1 } }

    // Each leaf walks up; the second arrival at an internal node is the only one to write it,
    // after both children are written. A box is a min and max, so any order gives the same one.
    fn update_boxes(&mut self) {
        if self.num_intl() == 0 { return; }
        let counter = (0..self.num_intl()).map(|_| AtomicU32::new(0)).collect::<Vec<_>>();
        let slots = Slots(self.node_bb.as_mut_ptr());
        let (parent, children) = (&self.node_parent, &self.intl_children);
        par::for_each(self.num_leaf(), |leaf| {
            let slots = &slots;
            let mut node = leaf2node(leaf as i32);
            while node != K_ROOT {
                node = parent[node as usize];
                let intl = node2intl(node).unwrap() as usize;
                if counter[intl].fetch_add(1, Ordering::AcqRel) == 0 { return; }
                let (c1, c2) = children[intl];
                // SAFETY: indices lie inside `node_bb`; the counter hands `node` to this thread
                // alone, and its acquire orders the children's writes before these reads.
                unsafe {
                    let b = union_bbs(&*slots.0.add(c1 as usize), &*slots.0.add(c2 as usize));
                    *slots.0.add(node as usize) = b;
                }
            }
        });
    }

    pub fn new(
        leaf_bb: &[BBox],
        leaf_morton: &[u32]
    ) -> Self {
        let n_intl = leaf_bb.len() - 1;
        let n_node = 2 * leaf_bb.len() - 1;
        let mut node_parent = vec![-1; n_node];
        let tree = RadixTree { leaf_morton };
        let intl_children = par::tabulate(n_intl, |i| tree.op(i as i32));
        for (i, &(c1, c2)) in intl_children.iter().enumerate() {
            node_parent[c1 as usize] = intl2node(i as i32);
            node_parent[c2 as usize] = intl2node(i as i32);
        }

        let mut res = MortonCollider {
            node_bb: par::tabulate(n_node, |n| if n % 2 == 0 { leaf_bb[n / 2].clone() } else { BBox::default() }),
            node_parent,
            intl_children
        };

        res.update_boxes();
        res
    }

    /// What `f` makes of every (query id, leaf) pair that overlaps, in the order the queries
    /// and then the leaves are visited, so the result does not depend on threads.
    pub fn collision_map<T: Send>(&self, queries: &[Query], f: impl Fn(usize, usize) -> Option<T> + Sync + Send) -> Vec<T> {
        par::flat_tabulate(queries.len(), |i| {
            let mut out = vec![];
            let mut record = |q, l| if let Some(t) = f(q, l) { out.push(t); };
            find_collisions(queries, &self.node_bb, &self.intl_children, i, &mut record, false);
            out
        })
    }

    /// The sum of `f` over the leaves each query overlaps, one sum per query in query order.
    pub fn collision_sum(&self, queries: &[Query], f: impl Fn(usize, usize) -> i32 + Sync + Send) -> Vec<i32> {
        par::tabulate(queries.len(), |i| {
            let mut sum = 0;
            let mut record = |q, l| sum += f(q, l);
            find_collisions(queries, &self.node_bb, &self.intl_children, i, &mut record, false);
            sum
        })
    }
}

fn find_collisions<F>(
    queries: &[Query],
    node_bb: &[BBox],
    children: &[(i32, i32)],
    query_idx: usize,
    record: &mut F,
    self_collision: bool,
) where F: FnMut(usize, usize) {
    // depth-first search
    let mut stack = [0; 64];
    let mut top = -1i32;
    let mut node = K_ROOT;

    let mut rec = |node: i32| {
        let q = &queries[query_idx];
        let overlap = node_bb[node as usize].overlaps(q);
        if overlap && let Some(il) = node2leaf(node) {
            if !self_collision || il != query_idx as i32 {
                match q {
                    Query::Bb(q) => { if let Some(iq) = q.id { record(iq, il as usize); }},
                    Query::Pt(q) => { if let Some(iq) = q.id { record(iq, il as usize); }},
                }
            }
        }
        overlap && node2intl(node).is_some() //should traverse into node
    };

    loop {
        let intl = node2intl(node).unwrap();
        let (c1, c2) = children[intl as usize];
        let traverse1 = rec(c1);
        let traverse2 = rec(c2);
        if !traverse1 && !traverse2 {
            if top < 0 { break; } // done
            node = stack[top as usize];
            top -= 1;
        } else {
            node = if traverse1 { c1 } else { c2 }; // go here next
            if traverse1 && traverse2 {
                top += 1;
                stack[top as usize] = c2; // save the other for later
            }
        }
    }
}

