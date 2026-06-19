use crate::{Result, RmcError};

use super::sets::DynMeasurementSet;
use super::traits::{Kernel, Measurement, RunCallbacks, StatefulKernel, StatefulMeasurement};

#[derive(Clone, Copy, Debug)]
pub struct SimulationParams {
    pub max_steps: u64,
    pub steps_per_cycle: u64,
    pub cycles_per_check: u64,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            max_steps: 1,
            steps_per_cycle: 1,
            cycles_per_check: 1,
        }
    }
}

impl SimulationParams {
    pub fn validate(&self) -> Result<()> {
        if self.steps_per_cycle == 0 {
            return Err(RmcError::InvalidArgument(
                "steps_per_cycle must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SimulationStats {
    pub steps_done: u64,
    pub cycles_done: u64,
}

impl crate::Merge for SimulationStats {
    fn merge(self, other: Self) -> Self {
        Self {
            steps_done: self.steps_done + other.steps_done,
            cycles_done: self.cycles_done + other.cycles_done,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SimulationCtx {
    pub steps_done: u64,
    pub cycles_done: u64,
    pub steps_in_cycle: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NoopCallbacks;

impl RunCallbacks<SimulationCtx> for NoopCallbacks {}

pub fn run<R, K>(
    rng: &mut R,
    kernel: &mut K,
    measurements: &mut DynMeasurementSet,
    params: SimulationParams,
) -> Result<SimulationStats>
where
    K: Kernel<R>,
{
    let mut callbacks = NoopCallbacks;
    run_with_callbacks(rng, kernel, measurements, params, &mut callbacks)
}

pub fn run_with_callbacks<R, K, C>(
    rng: &mut R,
    kernel: &mut K,
    measurements: &mut DynMeasurementSet,
    params: SimulationParams,
    callbacks: &mut C,
) -> Result<SimulationStats>
where
    K: Kernel<R>,
    C: RunCallbacks<SimulationCtx>,
{
    params.validate()?;
    kernel.prepare()?;
    measurements.refresh_active();

    let mut stats = SimulationStats::default();
    while stats.steps_done < params.max_steps {
        for steps_in_cycle in 0..params.steps_per_cycle {
            if stats.steps_done >= params.max_steps {
                break;
            }

            kernel.step(rng)?;
            stats.steps_done += 1;
            callbacks.on_step(&SimulationCtx {
                steps_done: stats.steps_done,
                cycles_done: stats.cycles_done,
                steps_in_cycle: steps_in_cycle + 1,
            });
        }

        measurements.measure_all();
        stats.cycles_done += 1;
        let ctx = SimulationCtx {
            steps_done: stats.steps_done,
            cycles_done: stats.cycles_done,
            steps_in_cycle: 0,
        };
        callbacks.on_cycle(&ctx);

        if params.cycles_per_check > 0 && stats.cycles_done % params.cycles_per_check == 0 {
            callbacks.on_checkpoint(&ctx);
            if callbacks.stop_when(&ctx) {
                break;
            }
        }
    }

    Ok(stats)
}

pub fn run_typed<R, K, M>(
    rng: &mut R,
    kernel: &mut K,
    measurement: M,
    params: SimulationParams,
) -> Result<(SimulationStats, M::Output)>
where
    K: Kernel<R>,
    M: Measurement,
{
    let mut callbacks = NoopCallbacks;
    run_typed_with_callbacks(rng, kernel, measurement, params, &mut callbacks)
}

pub fn run_typed_with_callbacks<R, K, M, C>(
    rng: &mut R,
    kernel: &mut K,
    mut measurement: M,
    params: SimulationParams,
    callbacks: &mut C,
) -> Result<(SimulationStats, M::Output)>
where
    K: Kernel<R>,
    M: Measurement,
    C: RunCallbacks<SimulationCtx>,
{
    params.validate()?;
    kernel.prepare()?;

    let mut stats = SimulationStats::default();
    while stats.steps_done < params.max_steps {
        for steps_in_cycle in 0..params.steps_per_cycle {
            if stats.steps_done >= params.max_steps {
                break;
            }

            kernel.step(rng)?;
            stats.steps_done += 1;
            callbacks.on_step(&SimulationCtx {
                steps_done: stats.steps_done,
                cycles_done: stats.cycles_done,
                steps_in_cycle: steps_in_cycle + 1,
            });
        }

        measurement.measure();
        stats.cycles_done += 1;
        let ctx = SimulationCtx {
            steps_done: stats.steps_done,
            cycles_done: stats.cycles_done,
            steps_in_cycle: 0,
        };
        callbacks.on_cycle(&ctx);

        if params.cycles_per_check > 0 && stats.cycles_done % params.cycles_per_check == 0 {
            callbacks.on_checkpoint(&ctx);
            if callbacks.stop_when(&ctx) {
                break;
            }
        }
    }

    Ok((stats, measurement.finish()))
}

pub fn run_stateful_typed<State, R, K, M>(
    state: State,
    rng: &mut R,
    kernel: &mut K,
    measurement: M,
    params: SimulationParams,
) -> Result<(State, SimulationStats, M::Output)>
where
    K: StatefulKernel<State, R>,
    M: StatefulMeasurement<State>,
{
    let mut callbacks = NoopCallbacks;
    run_stateful_typed_with_callbacks(state, rng, kernel, measurement, params, &mut callbacks)
}

pub fn run_stateful_typed_with_callbacks<State, R, K, M, C>(
    mut state: State,
    rng: &mut R,
    kernel: &mut K,
    mut measurement: M,
    params: SimulationParams,
    callbacks: &mut C,
) -> Result<(State, SimulationStats, M::Output)>
where
    K: StatefulKernel<State, R>,
    M: StatefulMeasurement<State>,
    C: RunCallbacks<SimulationCtx>,
{
    params.validate()?;
    kernel.prepare(&mut state)?;

    let mut stats = SimulationStats::default();
    while stats.steps_done < params.max_steps {
        for steps_in_cycle in 0..params.steps_per_cycle {
            if stats.steps_done >= params.max_steps {
                break;
            }

            kernel.step(&mut state, rng)?;
            stats.steps_done += 1;
            callbacks.on_step(&SimulationCtx {
                steps_done: stats.steps_done,
                cycles_done: stats.cycles_done,
                steps_in_cycle: steps_in_cycle + 1,
            });
        }

        measurement.measure(&state);
        stats.cycles_done += 1;
        let ctx = SimulationCtx {
            steps_done: stats.steps_done,
            cycles_done: stats.cycles_done,
            steps_in_cycle: 0,
        };
        callbacks.on_cycle(&ctx);

        if params.cycles_per_check > 0 && stats.cycles_done % params.cycles_per_check == 0 {
            callbacks.on_checkpoint(&ctx);
            if callbacks.stop_when(&ctx) {
                break;
            }
        }
    }

    Ok((state, stats, measurement.finish()))
}
