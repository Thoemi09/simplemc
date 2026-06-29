use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rmc_core::mc::{
    run, run_typed, DynMeasurementSet, DynMetropolisKernel, DynUpdateSet, Measurement,
    MeasurementEntry, MetropolisKernel, SimulationParams, SingleUpdateSet, Update, UpdateEntry,
};
use rmc_core::random::{ChainId, SeedSource};

#[derive(Clone)]
struct IncrementStateUpdate;

impl Update<u64> for IncrementStateUpdate {
    fn attempt<R: rand::Rng + ?Sized>(&mut self, state: &mut u64, rng: &mut R) -> f64 {
        *state = state.wrapping_add(rng.next_u64() & 1);
        1.0
    }

    fn accept(&mut self, state: &mut u64) {
        *state = state.wrapping_add(1);
    }
}

#[derive(Clone)]
struct IncrementUnitUpdate {
    value: u64,
}

impl Update<()> for IncrementUnitUpdate {
    fn attempt<R: rand::Rng + ?Sized>(&mut self, _state: &mut (), rng: &mut R) -> f64 {
        self.value = self.value.wrapping_add(rng.next_u64() & 1);
        1.0
    }

    fn accept(&mut self, _state: &mut ()) {
        self.value = self.value.wrapping_add(1);
    }
}

struct FinalStateMeasurement;

impl Measurement<u64> for FinalStateMeasurement {
    type Output = u64;

    fn measure(&mut self, _state: &u64) {}

    fn finish(self) -> Self::Output {
        0
    }
}

#[derive(Clone)]
struct NoopDynMeasurement;

impl Measurement<()> for NoopDynMeasurement {
    type Output = ();

    fn measure(&mut self, _state: &()) {}

    fn finish(self) -> Self::Output {}
}

fn params() -> SimulationParams {
    SimulationParams {
        max_steps: 10_000,
        steps_per_cycle: 10_000,
        cycles_per_check: 1,
    }
}

fn bench_static_single_update(c: &mut Criterion) {
    c.bench_function("static_single_update_10k_steps", |b| {
        b.iter_batched(
            || {
                let rng = SeedSource::new(0x5eed).rng_for(ChainId(0));
                let kernel = MetropolisKernel::new(SingleUpdateSet::new(IncrementStateUpdate));
                (rng, kernel)
            },
            |(mut rng, mut kernel)| {
                let (state, stats, output) = run_typed(
                    0_u64,
                    &mut rng,
                    &mut kernel,
                    FinalStateMeasurement,
                    params(),
                )
                .unwrap();
                black_box((state, stats, output));
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_dyn_single_update(c: &mut Criterion) {
    c.bench_function("dyn_single_update_10k_steps", |b| {
        b.iter_batched(
            || {
                let rng = SeedSource::new(0x5eed).rng_for(ChainId(0));
                let mut updates = DynUpdateSet::new();
                updates
                    .add(UpdateEntry::new(IncrementUnitUpdate { value: 0 }, "inc", 1.0).unwrap())
                    .unwrap();
                let kernel = DynMetropolisKernel::new(updates);
                let mut measurements = DynMeasurementSet::new();
                measurements
                    .add(MeasurementEntry::new(NoopDynMeasurement, "noop").unwrap())
                    .unwrap();
                (rng, kernel, measurements)
            },
            |(mut rng, mut kernel, mut measurements)| {
                let (state, stats) =
                    run((), &mut rng, &mut kernel, &mut measurements, params()).unwrap();
                black_box((state, stats));
            },
            BatchSize::SmallInput,
        )
    });
}

fn hot_path(c: &mut Criterion) {
    bench_static_single_update(c);
    bench_dyn_single_update(c);
}

criterion_group!(benches, hot_path);
criterion_main!(benches);
