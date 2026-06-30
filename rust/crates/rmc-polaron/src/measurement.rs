use rmc_core::mc::Measurement;
use rmc_core::Merge;
use rmc_grids::{Grid1d, LinearGrid};

use crate::diagram::{norm0, Diagram};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct BatchedSum {
    n_batches: usize,
    batch_sums: Vec<f64>,
    batch_counts: Vec<u64>,
    next: usize,
}

impl BatchedSum {
    pub fn new(n_batches: usize) -> Self {
        assert!(n_batches >= 2, "jackknife needs at least two batches");
        Self {
            n_batches,
            batch_sums: vec![0.0; n_batches],
            batch_counts: vec![0; n_batches],
            next: 0,
        }
    }

    pub fn push(&mut self, value: f64) {
        let batch = self.next % self.n_batches;
        self.batch_sums[batch] += value;
        self.batch_counts[batch] += 1;
        self.next += 1;
    }

    pub fn reset(&mut self) {
        self.batch_sums.fill(0.0);
        self.batch_counts.fill(0);
        self.next = 0;
    }

    pub fn n_batches(&self) -> usize {
        self.n_batches
    }

    pub fn total_count(&self) -> u64 {
        self.batch_counts.iter().sum()
    }

    pub fn total_sum(&self) -> f64 {
        self.batch_sums.iter().sum()
    }

    pub fn mean(&self) -> Option<f64> {
        let count = self.total_count();
        (count > 0).then_some(self.total_sum() / count as f64)
    }

    fn batch_mean(&self, batch: usize) -> Option<f64> {
        let count = self.batch_counts[batch];
        (count > 0).then_some(self.batch_sums[batch] / count as f64)
    }
}

impl Merge for BatchedSum {
    fn merge(self, other: Self) -> Self {
        assert_eq!(self.n_batches, other.n_batches);
        let batch_sums = self
            .batch_sums
            .into_iter()
            .zip(other.batch_sums)
            .map(|(lhs, rhs)| lhs + rhs)
            .collect();
        let batch_counts = self
            .batch_counts
            .into_iter()
            .zip(other.batch_counts)
            .map(|(lhs, rhs)| lhs + rhs)
            .collect();
        Self {
            n_batches: self.n_batches,
            batch_sums,
            batch_counts,
            next: self.next + other.next,
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct GridSpec {
    pub first: f64,
    pub last: f64,
    pub len: usize,
}

impl GridSpec {
    pub fn new(first: f64, last: f64, len: usize) -> Self {
        Self { first, last, len }
    }

    pub fn grid(self) -> LinearGrid {
        LinearGrid::new(self.first, self.last, self.len).expect("measurement grid must be valid")
    }

    pub fn bin_count(self) -> usize {
        self.len - 1
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct Estimate {
    pub mean: f64,
    pub stderr: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct SeriesEstimate {
    pub tau: Vec<f64>,
    pub mean: Vec<f64>,
    pub stderr: Vec<f64>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct PolaronStats {
    pub zeroth: BatchedSum,
    pub exact: Vec<BatchedSum>,
    pub hist: Vec<BatchedSum>,
    pub energy: BatchedSum,
    pub a: BatchedSum,
    pub order: BatchedSum,
    pub grid: GridSpec,
    pub energy_estimate: f64,
    pub energy_estimates: Vec<f64>,
    pub self_consistent_count: usize,
    pub self_consistent_period: usize,
    pub self_consistent_periods: Vec<usize>,
    pub period_multiplier: f64,
    pub alpha: f64,
    pub mu: f64,
    pub momentum: f64,
    pub max_tau: f64,
    pub sample_count: usize,
}

impl PolaronStats {
    pub fn jackknife_selfenergy(&self) -> SeriesEstimate {
        let grid = self.grid.grid();
        let dispersion = self.dispersion();
        let n0 = norm0(self.max_tau, dispersion);
        let binsize = grid.step().abs();
        let mut mean = Vec::with_capacity(self.exact.len());
        let mut stderr = Vec::with_capacity(self.exact.len());
        for exact in &self.exact {
            let estimate = jackknife_ratio(exact, &self.zeroth, |num, den| {
                if den == 0.0 {
                    f64::NAN
                } else {
                    num * n0 / (den * binsize)
                }
            });
            mean.push(estimate.mean);
            stderr.push(estimate.stderr);
        }
        SeriesEstimate {
            tau: grid.bin_centers().collect(),
            mean,
            stderr,
        }
    }

    pub fn jackknife_energy(&self) -> Estimate {
        let n0 = norm0(self.max_tau, self.dispersion());
        jackknife_ratio(&self.energy, &self.zeroth, |energy, zeroth| {
            if zeroth == 0.0 {
                f64::NAN
            } else {
                energy * n0 / zeroth
            }
        })
    }

    pub fn jackknife_quasiparticle_weight(&self) -> Estimate {
        let n0 = norm0(self.max_tau, self.dispersion());
        jackknife_ratio(&self.a, &self.zeroth, |a, zeroth| {
            if zeroth == 0.0 {
                f64::NAN
            } else {
                1.0 / (1.0 + a * n0 / zeroth)
            }
        })
    }

    pub fn get_exact(&self) -> Vec<f64> {
        let grid = self.grid.grid();
        let n0 = norm0(self.max_tau, self.dispersion());
        let zeroth = self.zeroth.mean().unwrap_or(f64::NAN);
        self.exact
            .iter()
            .enumerate()
            .map(|(idx, exact)| {
                let binsize = grid.bin_width(idx).expect("bin must exist");
                exact.mean().unwrap_or(0.0) / binsize * n0 / zeroth
            })
            .collect()
    }

    fn dispersion(&self) -> f64 {
        self.momentum * self.momentum / (2.0 * Diagram::MASS) - self.mu
    }
}

impl Merge for PolaronStats {
    fn merge(self, other: Self) -> Self {
        assert_eq!(self.grid, other.grid);
        assert_eq!(self.exact.len(), other.exact.len());
        assert_eq!(self.hist.len(), other.hist.len());
        Self {
            zeroth: self.zeroth.merge(other.zeroth),
            exact: self
                .exact
                .into_iter()
                .zip(other.exact)
                .map(|(lhs, rhs)| lhs.merge(rhs))
                .collect(),
            hist: self
                .hist
                .into_iter()
                .zip(other.hist)
                .map(|(lhs, rhs)| lhs.merge(rhs))
                .collect(),
            energy: self.energy.merge(other.energy),
            a: self.a.merge(other.a),
            order: self.order.merge(other.order),
            grid: self.grid,
            energy_estimate: other.energy_estimate,
            energy_estimates: [self.energy_estimates, other.energy_estimates].concat(),
            self_consistent_count: self.self_consistent_count + other.self_consistent_count,
            self_consistent_period: other.self_consistent_period,
            self_consistent_periods: [self.self_consistent_periods, other.self_consistent_periods]
                .concat(),
            period_multiplier: self.period_multiplier,
            alpha: self.alpha,
            mu: self.mu,
            momentum: self.momentum,
            max_tau: self.max_tau,
            sample_count: self.sample_count + other.sample_count,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PolaronMeasurement {
    stats: PolaronStats,
}

impl PolaronMeasurement {
    pub fn new(
        num_bins: usize,
        max_tau: f64,
        n_batches: usize,
        energy_estimate: f64,
        self_consistent_period: usize,
        period_multiplier: f64,
        template: &Diagram,
    ) -> Self {
        let grid = GridSpec::new(0.0, max_tau, num_bins + 1);
        let exact = (0..num_bins).map(|_| BatchedSum::new(n_batches)).collect();
        let hist = (0..num_bins).map(|_| BatchedSum::new(n_batches)).collect();
        Self {
            stats: PolaronStats {
                zeroth: BatchedSum::new(n_batches),
                exact,
                hist,
                energy: BatchedSum::new(n_batches),
                a: BatchedSum::new(n_batches),
                order: BatchedSum::new(n_batches),
                grid,
                energy_estimate,
                energy_estimates: vec![energy_estimate],
                self_consistent_count: 0,
                self_consistent_period,
                self_consistent_periods: vec![self_consistent_period],
                period_multiplier,
                alpha: template.alpha,
                mu: template.mu,
                momentum: template.momentum,
                max_tau: template.max_tau,
                sample_count: 0,
            },
        }
    }

    pub fn stats(&self) -> &PolaronStats {
        &self.stats
    }

    fn reevaluate_energy_estimate(&mut self, d: &Diagram) {
        let estimate = self.stats.jackknife_energy();
        if !estimate.mean.is_finite() {
            return;
        }
        let new_estimate = Diagram::bare_dispersion(&d.momentum_out()) + estimate.mean;
        self.stats.energy_estimate = new_estimate;
        self.stats.energy_estimates.push(new_estimate);
        self.stats.energy.reset();
        self.stats.a.reset();
        self.stats.self_consistent_count = 0;
        self.stats.self_consistent_period =
            ((self.stats.self_consistent_period as f64) * self.stats.period_multiplier) as usize;
        self.stats
            .self_consistent_periods
            .push(self.stats.self_consistent_period);
    }
}

impl Measurement<Diagram> for PolaronMeasurement {
    type Output = PolaronStats;

    fn measure(&mut self, d: &Diagram) {
        let grid = self.stats.grid.grid();
        let Some(index) = grid.bin_index(d.tau()) else {
            return;
        };

        let is_zeroth = d.order == 0;
        let t0 = grid.bin_center(index).expect("bin center must exist");
        let exact_value = if is_zeroth {
            0.0
        } else {
            d.exact_estimator(t0)
        };
        let exp_energy = if is_zeroth {
            0.0
        } else {
            ((self.stats.energy_estimate - d.mu) * d.tau()).exp()
        };

        self.stats.zeroth.push(if is_zeroth { 1.0 } else { 0.0 });
        self.stats
            .energy
            .push(if is_zeroth { 0.0 } else { -exp_energy });
        self.stats
            .a
            .push(if is_zeroth { 0.0 } else { d.tau() * exp_energy });
        self.stats.order.push(d.order as f64);

        for bin in 0..self.stats.grid.bin_count() {
            self.stats
                .hist
                .get_mut(bin)
                .expect("hist bin must exist")
                .push(if !is_zeroth && bin == index { 1.0 } else { 0.0 });
            self.stats
                .exact
                .get_mut(bin)
                .expect("exact bin must exist")
                .push(if !is_zeroth && bin == index {
                    exact_value
                } else {
                    0.0
                });
        }

        self.stats.sample_count += 1;
        self.stats.self_consistent_count += 1;
        if self.stats.self_consistent_count > self.stats.self_consistent_period {
            self.reevaluate_energy_estimate(d);
        }
    }

    fn finish(self) -> Self::Output {
        self.stats
    }
}

pub fn jackknife_ratio<F>(num: &BatchedSum, den: &BatchedSum, f: F) -> Estimate
where
    F: Fn(f64, f64) -> f64,
{
    assert_eq!(num.n_batches(), den.n_batches());
    let mut batch_num = Vec::new();
    let mut batch_den = Vec::new();
    for batch in 0..num.n_batches() {
        if let (Some(num_mean), Some(den_mean)) = (num.batch_mean(batch), den.batch_mean(batch)) {
            batch_num.push(num_mean);
            batch_den.push(den_mean);
        }
    }

    let n = batch_num.len();
    if n < 2 {
        return Estimate {
            mean: f64::NAN,
            stderr: f64::NAN,
        };
    }

    let total_num: f64 = batch_num.iter().sum();
    let total_den: f64 = batch_den.iter().sum();
    let mut theta = Vec::with_capacity(n);
    for batch in 0..n {
        let loo_num = (total_num - batch_num[batch]) / (n - 1) as f64;
        let loo_den = (total_den - batch_den[batch]) / (n - 1) as f64;
        theta.push(f(loo_num, loo_den));
    }

    let finite = theta
        .into_iter()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let n = finite.len();
    if n < 2 {
        return Estimate {
            mean: f64::NAN,
            stderr: f64::NAN,
        };
    }

    let mean = finite.iter().sum::<f64>() / n as f64;
    let variance_sum = finite
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>();
    Estimate {
        mean,
        stderr: (((n - 1) as f64 / n as f64) * variance_sum).sqrt(),
    }
}
