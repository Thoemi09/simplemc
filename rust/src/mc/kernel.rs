use rand::Rng;

use crate::Result;

use super::sets::DynUpdateSet;
use super::traits::{
    Kernel, StatefulKernel, StepOutcome, SteppingStatefulUpdateSet, SteppingUpdateSet,
};

pub struct DynMetropolisKernel {
    updates: DynUpdateSet,
}

pub struct StaticMetropolisKernel<S> {
    updates: S,
}

pub struct StatefulMetropolisKernel<S> {
    updates: S,
}

impl<S> StaticMetropolisKernel<S> {
    pub fn new(updates: S) -> Self {
        Self { updates }
    }

    pub fn updates(&self) -> &S {
        &self.updates
    }

    pub fn updates_mut(&mut self) -> &mut S {
        &mut self.updates
    }

    pub fn into_updates(self) -> S {
        self.updates
    }
}

impl<R, S> Kernel<R> for StaticMetropolisKernel<S>
where
    R: Rng,
    S: SteppingUpdateSet<R>,
{
    fn prepare(&mut self) -> Result<()> {
        self.updates.prepare()
    }

    fn step(&mut self, rng: &mut R) -> Result<StepOutcome> {
        self.updates.select_and_step(rng)
    }
}

impl<S> StatefulMetropolisKernel<S> {
    pub fn new(updates: S) -> Self {
        Self { updates }
    }

    pub fn updates(&self) -> &S {
        &self.updates
    }

    pub fn updates_mut(&mut self) -> &mut S {
        &mut self.updates
    }

    pub fn into_updates(self) -> S {
        self.updates
    }
}

impl<State, R, S> StatefulKernel<State, R> for StatefulMetropolisKernel<S>
where
    R: Rng,
    S: SteppingStatefulUpdateSet<State, R>,
{
    fn prepare(&mut self, state: &mut State) -> Result<()> {
        self.updates.prepare(state)
    }

    fn step(&mut self, state: &mut State, rng: &mut R) -> Result<StepOutcome> {
        self.updates.select_and_step(state, rng)
    }
}

impl DynMetropolisKernel {
    pub fn new(updates: DynUpdateSet) -> Self {
        Self { updates }
    }

    pub fn updates(&self) -> &DynUpdateSet {
        &self.updates
    }

    pub fn updates_mut(&mut self) -> &mut DynUpdateSet {
        &mut self.updates
    }

    pub fn into_updates(self) -> DynUpdateSet {
        self.updates
    }
}

impl<R> Kernel<R> for DynMetropolisKernel
where
    R: Rng,
{
    fn prepare(&mut self) -> Result<()> {
        self.updates.rebuild_distribution()
    }

    fn step(&mut self, rng: &mut R) -> Result<StepOutcome> {
        self.updates.select_and_step(rng)
    }
}
