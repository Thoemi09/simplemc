use rayon::prelude::*;
use rayon::ThreadPool;

use crate::random::{ChainId, SeedSource};
use crate::{Merge, Result, RmcError};

use super::run::{run_stateful_typed, run_typed, SimulationParams, SimulationStats};
use super::traits::{Kernel, Measurement, StatefulKernel, StatefulMeasurement};

#[derive(Clone, Copy, Debug)]
pub struct ParallelConfig {
    pub chains: u64,
    pub seed: SeedSource,
    pub params: SimulationParams,
}

impl ParallelConfig {
    pub fn validate(&self) -> Result<()> {
        if self.chains == 0 {
            return Err(RmcError::InvalidArgument("chains must be > 0".to_string()));
        }
        self.params.validate()
    }
}

pub fn run_parallel<K, M, B>(
    config: ParallelConfig,
    build: B,
) -> Result<(SimulationStats, M::Output)>
where
    K: Kernel<crate::random::DefaultRng> + Send,
    M: Measurement + Send,
    M::Output: Merge + Send,
    B: Fn(ChainId) -> (K, M) + Send + Sync,
{
    run_parallel_impl(config, &build)
}

pub fn run_parallel_in_pool<K, M, B>(
    pool: &ThreadPool,
    config: ParallelConfig,
    build: B,
) -> Result<(SimulationStats, M::Output)>
where
    K: Kernel<crate::random::DefaultRng> + Send,
    M: Measurement + Send,
    M::Output: Merge + Send,
    B: Fn(ChainId) -> (K, M) + Send + Sync,
{
    pool.install(|| run_parallel_impl(config, &build))
}

pub fn run_stateful_parallel<State, K, M, B>(
    config: ParallelConfig,
    build: B,
) -> Result<(SimulationStats, M::Output)>
where
    State: Send,
    K: StatefulKernel<State, crate::random::DefaultRng> + Send,
    M: StatefulMeasurement<State> + Send,
    M::Output: Merge + Send,
    B: Fn(ChainId) -> (State, K, M) + Send + Sync,
{
    run_stateful_parallel_impl(config, &build)
}

pub fn run_stateful_parallel_in_pool<State, K, M, B>(
    pool: &ThreadPool,
    config: ParallelConfig,
    build: B,
) -> Result<(SimulationStats, M::Output)>
where
    State: Send,
    K: StatefulKernel<State, crate::random::DefaultRng> + Send,
    M: StatefulMeasurement<State> + Send,
    M::Output: Merge + Send,
    B: Fn(ChainId) -> (State, K, M) + Send + Sync,
{
    pool.install(|| run_stateful_parallel_impl(config, &build))
}

fn run_parallel_impl<K, M>(
    config: ParallelConfig,
    build: &(impl Fn(ChainId) -> (K, M) + Sync),
) -> Result<(SimulationStats, M::Output)>
where
    K: Kernel<crate::random::DefaultRng> + Send,
    M: Measurement + Send,
    M::Output: Merge + Send,
{
    config.validate()?;

    let partials = (0..config.chains)
        .into_par_iter()
        .map(|chain| {
            let chain_id = ChainId(chain);
            let mut rng = config.seed.rng_for(chain_id);
            let (mut kernel, measurement) = build(chain_id);
            run_typed(&mut rng, &mut kernel, measurement, config.params)
        })
        .collect::<Vec<_>>();

    let mut merged: Option<(SimulationStats, M::Output)> = None;
    for partial in partials {
        let (stats, output) = partial?;
        merged = Some(match merged {
            Some((acc_stats, acc_output)) => (acc_stats.merge(stats), acc_output.merge(output)),
            None => (stats, output),
        });
    }

    merged.ok_or_else(|| RmcError::InvalidState("parallel run produced no chains".to_string()))
}

fn run_stateful_parallel_impl<State, K, M>(
    config: ParallelConfig,
    build: &(impl Fn(ChainId) -> (State, K, M) + Sync),
) -> Result<(SimulationStats, M::Output)>
where
    State: Send,
    K: StatefulKernel<State, crate::random::DefaultRng> + Send,
    M: StatefulMeasurement<State> + Send,
    M::Output: Merge + Send,
{
    config.validate()?;

    let partials = (0..config.chains)
        .into_par_iter()
        .map(|chain| {
            let chain_id = ChainId(chain);
            let mut rng = config.seed.rng_for(chain_id);
            let (state, mut kernel, measurement) = build(chain_id);
            run_stateful_typed(state, &mut rng, &mut kernel, measurement, config.params)
                .map(|(_state, stats, output)| (stats, output))
        })
        .collect::<Vec<_>>();

    let mut merged: Option<(SimulationStats, M::Output)> = None;
    for partial in partials {
        let (stats, output) = partial?;
        merged = Some(match merged {
            Some((acc_stats, acc_output)) => (acc_stats.merge(stats), acc_output.merge(output)),
            None => (stats, output),
        });
    }

    merged.ok_or_else(|| {
        RmcError::InvalidState("stateful parallel run produced no chains".to_string())
    })
}
