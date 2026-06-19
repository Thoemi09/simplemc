use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use rmc_core::mc::{
    run_parallel, run_typed, Measurement, ParallelConfig, SimulationParams, SingleUpdateSet,
    StaticMetropolisKernel, Update,
};
use rmc_core::random::{ChainId, SeedSource};
use rmc_core::Merge;

#[derive(Clone)]
struct RandomWalkUpdate {
    position: Arc<AtomicI64>,
    proposed_delta: i64,
}

impl RandomWalkUpdate {
    fn new(position: Arc<AtomicI64>) -> Self {
        Self {
            position,
            proposed_delta: 0,
        }
    }
}

impl Update for RandomWalkUpdate {
    fn attempt(&mut self, rng: &mut dyn rand::RngCore) -> f64 {
        self.proposed_delta = if rng.next_u64() & 1 == 0 { -1 } else { 1 };
        1.0
    }

    fn accept(&mut self) {
        self.position
            .fetch_add(self.proposed_delta, Ordering::Relaxed);
    }
}

struct WalkMeasurement {
    position: Arc<AtomicI64>,
    cycles: u64,
}

impl WalkMeasurement {
    fn new(position: Arc<AtomicI64>) -> Self {
        Self {
            position,
            cycles: 0,
        }
    }
}

impl Measurement for WalkMeasurement {
    type Output = WalkSummary;

    fn measure(&mut self) {
        self.cycles += 1;
    }

    fn finish(self) -> Self::Output {
        WalkSummary {
            final_position_sum: self.position.load(Ordering::Relaxed),
            measured_cycles: self.cycles,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct WalkSummary {
    final_position_sum: i64,
    measured_cycles: u64,
}

impl Merge for WalkSummary {
    fn merge(self, other: Self) -> Self {
        Self {
            final_position_sum: self.final_position_sum + other.final_position_sum,
            measured_cycles: self.measured_cycles + other.measured_cycles,
        }
    }
}

fn build_chain(
    _chain: ChainId,
) -> (
    StaticMetropolisKernel<SingleUpdateSet<RandomWalkUpdate>>,
    WalkMeasurement,
) {
    let position = Arc::new(AtomicI64::new(0));
    let update = RandomWalkUpdate::new(Arc::clone(&position));
    let measurement = WalkMeasurement::new(position);
    (
        StaticMetropolisKernel::new(SingleUpdateSet::new(update)),
        measurement,
    )
}

fn params() -> SimulationParams {
    SimulationParams {
        max_steps: 1_000,
        steps_per_cycle: 10,
        cycles_per_check: 1,
    }
}

fn main() -> rmc_core::Result<()> {
    let seed = SeedSource::new(0x5eed);
    let mut rng = seed.rng_for(ChainId(0));
    let (mut kernel, measurement) = build_chain(ChainId(0));
    let (single_stats, single_summary) = run_typed(&mut rng, &mut kernel, measurement, params())?;

    println!(
        "single chain: steps={}, cycles={}, final_position={}",
        single_stats.steps_done, single_stats.cycles_done, single_summary.final_position_sum
    );

    let chains = 8;
    let (parallel_stats, parallel_summary) = run_parallel(
        ParallelConfig {
            chains,
            seed,
            params: params(),
        },
        build_chain,
    )?;

    println!(
        "parallel: chains={}, steps={}, cycles={}, mean_final_position={:.3}",
        chains,
        parallel_stats.steps_done,
        parallel_stats.cycles_done,
        parallel_summary.final_position_sum as f64 / chains as f64
    );

    Ok(())
}
