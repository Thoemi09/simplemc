use rmc_core::mc::{
    run_typed, Measurement, MetropolisKernel, SimulationParams, SingleUpdateSet, Update,
};
use rmc_core::random::{uniform_index, ChainId, SeedSource};

#[derive(Clone, Debug)]
struct Ising2x2 {
    spins: [i8; 4],
}

impl Ising2x2 {
    fn ordered() -> Self {
        Self { spins: [1; 4] }
    }

    fn magnetization_per_spin(&self) -> f64 {
        self.spins.iter().map(|spin| f64::from(*spin)).sum::<f64>() / self.spins.len() as f64
    }

    fn flip(&mut self, idx: usize) {
        self.spins[idx] *= -1;
    }
}

#[derive(Clone, Debug)]
struct LazyInfiniteTemperatureFlip {
    proposed_idx: usize,
}

impl LazyInfiniteTemperatureFlip {
    fn new() -> Self {
        Self { proposed_idx: 0 }
    }
}

impl Update<Ising2x2> for LazyInfiniteTemperatureFlip {
    fn attempt<R: rand::Rng + ?Sized>(&mut self, state: &mut Ising2x2, rng: &mut R) -> f64 {
        self.proposed_idx = uniform_index(rng, state.spins.len());
        0.5
    }

    fn accept(&mut self, state: &mut Ising2x2) {
        state.flip(self.proposed_idx);
    }
}

#[derive(Clone, Debug, Default)]
struct MagnetizationMoments {
    samples: u64,
    signed_sum: f64,
    abs_sum: f64,
}

impl Measurement<Ising2x2> for MagnetizationMoments {
    type Output = Self;

    fn measure(&mut self, state: &Ising2x2) {
        let magnetization = state.magnetization_per_spin();
        self.samples += 1;
        self.signed_sum += magnetization;
        self.abs_sum += magnetization.abs();
    }

    fn finish(self) -> Self::Output {
        self
    }
}

impl MagnetizationMoments {
    fn mean_signed(&self) -> f64 {
        self.signed_sum / self.samples as f64
    }

    fn mean_abs(&self) -> f64 {
        self.abs_sum / self.samples as f64
    }
}

#[test]
fn infinite_temperature_ising_2x2_matches_exact_magnetization() {
    let mut rng = SeedSource::new(0x15_1eaf).rng_for(ChainId(0));
    let mut kernel =
        MetropolisKernel::new(SingleUpdateSet::new(LazyInfiniteTemperatureFlip::new()));

    let (_state, stats, moments) = run_typed(
        Ising2x2::ordered(),
        &mut rng,
        &mut kernel,
        MagnetizationMoments::default(),
        SimulationParams {
            max_steps: 200_000,
            steps_per_cycle: 4,
            cycles_per_check: 1,
        },
    )
    .unwrap();

    assert_eq!(stats.cycles_done, 50_000);
    assert!(
        moments.mean_signed().abs() < 0.02,
        "signed magnetization should vanish at beta=0, got {}",
        moments.mean_signed()
    );
    assert!(
        (moments.mean_abs() - exact_infinite_temperature_abs_magnetization()).abs() < 0.02,
        "absolute magnetization mean={} expected={}",
        moments.mean_abs(),
        exact_infinite_temperature_abs_magnetization()
    );
}

fn exact_infinite_temperature_abs_magnetization() -> f64 {
    let mut total = 0.0;
    for mask in 0_u8..16 {
        let magnetization = (0..4)
            .map(|bit| if mask & (1 << bit) == 0 { -1.0 } else { 1.0 })
            .sum::<f64>()
            / 4.0;
        total += magnetization.abs();
    }
    total / 16.0
}
