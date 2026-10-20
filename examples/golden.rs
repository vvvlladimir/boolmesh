// Prints a fingerprint of every result in a fixed set of booleans, and how long each took.
// Run before and after a change that must not alter any output, and diff the two:
// cargo run --release --features rayon --example golden

#[path = "../benches/shapes.rs"]
mod shapes;

use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::Instant;
use boolmesh::prelude::*;
use shapes::*;

fn fingerprint(m: &Manifold) -> u64 {
    let mut h = DefaultHasher::new();
    for p in &m.ps { [p.x as f64, p.y as f64, p.z as f64].map(f64::to_bits).hash(&mut h); }
    for e in &m.hs { [e.tail as usize, e.head as usize, e.pair as usize].hash(&mut h); }
    h.finish()
}

fn run(name: &str, a: &Soup, others: &[(Soup, OpType)]) {
    let t = Instant::now();
    let mut acc = a.manifold();
    let read = t.elapsed().as_secs_f64();
    let mut op = 0.;
    for (b, kind) in others {
        let mb = b.manifold();
        let t = Instant::now();
        let k = match kind { OpType::Add => OpType::Add, OpType::Subtract => OpType::Subtract, OpType::Intersect => OpType::Intersect };
        acc = compute_boolean(&acc, &mb, k).expect("the booleans have an answer");
        op += t.elapsed().as_secs_f64();
    }
    println!("{name:<12} {:>8} faces  {:016x}  read {read:.3} s  op {op:.3} s", acc.nf, fingerprint(&acc));
}

fn main() {
    let only = std::env::args().nth(1);
    let wanted = |name: &str| only.as_deref().is_none_or(|o| name.starts_with(o));
    let sub = |s: Soup| (s, OpType::Subtract);

    if wanted("prisms-200k") { run("prisms-200k", &dense_ball(316), &[sub(prisms())]); }
    if wanted("prisms-1m") { run("prisms-1m", &dense_ball(708), &[sub(prisms())]); }
    if wanted("letters") { run("letters", &dense_ball(512), &[(letters(), OpType::Add)]); }
    if wanted("floor-1m") { run("floor-1m", &dense_ball(708), &[sub(cuboid([-30., -30., -30.], [30., 30., -15.]))]); }
    if wanted("spheres") {
        run("spheres", &ball([0.; 3], 20., 200, 200), &[(ball([5., 3., 2.], 18., 190, 190), OpType::Intersect)]);
    }
    if wanted("chain") {
        let holes = (0..20).map(|i| {
            let a = i as f64 * 0.31;
            let (x, y) = (14. * a.cos(), 14. * a.sin());
            sub(cuboid([x - 2., y - 2., -25.], [x + 2., y + 2., 25.]))
        }).collect::<Vec<_>>();
        run("chain", &dense_ball(224), &holes);
    }
    if wanted("coplanar") {
        run("coplanar", &cuboid([0.; 3], [10.; 3]), &[
            (cuboid([10., 0., 0.], [20., 10., 10.]), OpType::Add),
            sub(cuboid([5., 5., 0.], [15., 15., 10.])),
            (cuboid([0., 0., 0.], [20., 5., 5.]), OpType::Intersect),
        ]);
    }
}
