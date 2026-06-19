use rand::RngCore;

use crate::Result;

pub trait Measurement {
    type Output;

    fn measure(&mut self);
    fn finish(self) -> Self::Output;
}

pub trait StatefulMeasurement<State> {
    type Output;

    fn measure(&mut self, state: &State);
    fn finish(self) -> Self::Output;
}

pub trait Update {
    fn attempt(&mut self, rng: &mut dyn RngCore) -> f64;
    fn accept(&mut self);
    fn reject(&mut self) {}
}

pub trait StatefulUpdate<State> {
    fn attempt(&mut self, state: &mut State, rng: &mut dyn RngCore) -> f64;
    fn accept(&mut self, state: &mut State);
    fn reject(&mut self, _state: &mut State) {}
}

pub struct StepOutcome {
    pub update_index: usize,
    pub probability: f64,
    pub accepted: bool,
    pub impossible: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UpdateStats {
    pub nprops: u64,
    pub naccs: u64,
    pub nimps: u64,
}

pub trait UpdateSet {
    fn stats(&self) -> &[UpdateStats];
}

pub trait SteppingUpdateSet<R>: UpdateSet {
    fn prepare(&mut self) -> Result<()>;
    fn select_and_step(&mut self, rng: &mut R) -> Result<StepOutcome>;
}

pub trait SteppingStatefulUpdateSet<State, R>: UpdateSet {
    fn prepare(&mut self, state: &mut State) -> Result<()>;
    fn select_and_step(&mut self, state: &mut State, rng: &mut R) -> Result<StepOutcome>;
}

pub trait Kernel<R> {
    fn prepare(&mut self) -> Result<()> {
        Ok(())
    }

    fn step(&mut self, rng: &mut R) -> Result<StepOutcome>;
}

pub trait StatefulKernel<State, R> {
    fn prepare(&mut self, _state: &mut State) -> Result<()> {
        Ok(())
    }

    fn step(&mut self, state: &mut State, rng: &mut R) -> Result<StepOutcome>;
}

pub trait RunCallbacks<C> {
    fn on_step(&mut self, _ctx: &C) {}
    fn on_cycle(&mut self, _ctx: &C) {}
    fn on_checkpoint(&mut self, _ctx: &C) {}
    fn stop_when(&mut self, _ctx: &C) -> bool {
        false
    }
}
