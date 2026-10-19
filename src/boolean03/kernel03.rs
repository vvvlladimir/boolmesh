//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

use super::kernel02::Kernel02;
use crate::bounds::{BPos, Query};
use crate::{Real, Vec2, Manifold};

pub fn winding03<S>(
    mp: &Manifold<S>,
    mq: &Manifold<S>,
    expand: Real,
    fwd: bool
) -> Vec<i32> {
    let ma = if fwd { mp } else { mq };
    let mb = if fwd { mq } else { mp };

    let k02 = Kernel02 {
        ps_p: &ma.ps,
        ps_q: &mb.ps,
        hs_q: &mb.hs,
        ns: &mp.vns,
        expand,
        fwd,
    };

    let queries = ma.ps.iter().enumerate()
        .map(|(i, p)| Query::Pt(BPos{id: Some(i), pos: Vec2::new(p.x, p.y)}))
        .collect::<Vec<_>>();
    mb.collider.collision_sum(&queries, |a, b| k02.op(a, b).map_or(0, |(s, _)| s * if fwd { 1 } else { -1 }))
}
