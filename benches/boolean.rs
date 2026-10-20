// Reading a mesh and one boolean on it, at the sizes a print model reaches.
// cargo bench --features rayon --bench boolean

mod shapes;

use std::time::Duration;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use boolmesh::prelude::*;
use shapes::*;

fn read(c: &mut Criterion) {
    let mut g = c.benchmark_group("read");
    g.sample_size(10).measurement_time(Duration::from_secs(10));
    for faces in [200_000, 1_000_000] {
        let soup = dense_ball(stacks_for(faces));
        g.bench_with_input(BenchmarkId::from_parameter(faces), &soup, |b, s| b.iter(|| s.manifold()));
    }
    g.finish();
}

fn boolean(c: &mut Criterion) {
    let mut g = c.benchmark_group("boolean");
    g.sample_size(10).measurement_time(Duration::from_secs(10));
    let prisms = prisms().manifold();
    for faces in [200_000, 1_000_000] {
        let ball = dense_ball(stacks_for(faces)).manifold();
        g.bench_with_input(BenchmarkId::new("ball-minus-prisms", faces), &ball, |b, a| {
            b.iter(|| compute_boolean(a, &prisms, OpType::Subtract).expect("the prisms cross the ball"))
        });
    }
    let a = ball([0.; 3], 20., 200, 200).manifold();
    let s = ball([5., 3., 2.], 18., 190, 190).manifold();
    g.bench_function("spheres-intersect-80k", |b| {
        b.iter(|| compute_boolean(&a, &s, OpType::Intersect).expect("the balls overlap"))
    });
    let small = dense_ball(stacks_for(20_000)).manifold();
    let block = cuboid([-5., -5., -25.], [5., 5., 25.]).manifold();
    g.bench_function("small-20k", |b| {
        b.iter(|| compute_boolean(&small, &block, OpType::Subtract).expect("the block crosses the ball"))
    });
    g.finish();
}

fn coplanar(c: &mut Criterion) {
    let mut g = c.benchmark_group("coplanar");
    g.sample_size(10).measurement_time(Duration::from_secs(5));
    let stacks = stacks_for(1_000_000);
    for (name, soup) in [("quads", dense_ball(stacks)), ("organic", jittered(dense_ball(stacks), 0.01))] {
        let m = soup.manifold();
        g.bench_function(name, |b| b.iter_batched(|| m.clone(), |m| m.coplanar().len(), BatchSize::LargeInput));
    }
    g.finish();
}

criterion_group!(benches, read, boolean, coplanar);
criterion_main!(benches);
