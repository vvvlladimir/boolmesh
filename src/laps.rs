//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

// Wall time of each stage on stderr when the `timings` feature is on; nothing otherwise.

#[cfg(feature = "timings")]
pub struct Laps { what: &'static str, at: std::time::Instant }

#[cfg(feature = "timings")]
impl Laps {
    pub fn new(what: &'static str) -> Self { Self { what, at: std::time::Instant::now() } }
    pub fn lap(&mut self, stage: &str) {
        eprintln!("{:>10} {stage:<14} {:.4} s", self.what, self.at.elapsed().as_secs_f64());
        self.at = std::time::Instant::now();
    }
}

#[cfg(not(feature = "timings"))]
pub struct Laps;

#[cfg(not(feature = "timings"))]
impl Laps {
    pub fn new(_: &'static str) -> Self { Self }
    pub fn lap(&mut self, _: &str) {}
}
