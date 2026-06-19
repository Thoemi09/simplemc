use rand::RngCore;

use crate::{Result, RmcError};

use super::named_set::NamedEntry;
use super::traits::{Measurement, Update};
use super::wrappers::{DynMeasurement, DynUpdate};

#[derive(Clone)]
pub struct MeasurementEntry {
    pub wrapped: DynMeasurement,
    pub name: String,
    pub is_active: bool,
}

impl MeasurementEntry {
    pub fn new<M>(measurement: M, name: impl Into<String>) -> Result<Self>
    where
        M: Measurement<Output = ()> + Clone + 'static,
    {
        Self::new_with_active(measurement, name, true)
    }

    pub fn new_with_active<M>(
        measurement: M,
        name: impl Into<String>,
        is_active: bool,
    ) -> Result<Self>
    where
        M: Measurement<Output = ()> + Clone + 'static,
    {
        Self::from_dyn(DynMeasurement::new(measurement), name, is_active)
    }

    pub fn from_dyn(
        wrapped: DynMeasurement,
        name: impl Into<String>,
        is_active: bool,
    ) -> Result<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(RmcError::InvalidArgument(
                "measurement name must be non-empty".to_string(),
            ));
        }

        Ok(Self {
            wrapped,
            name,
            is_active,
        })
    }
}

impl Measurement for MeasurementEntry {
    type Output = ();

    fn measure(&mut self) {
        self.wrapped.measure();
    }

    fn finish(self) -> Self::Output {}
}

impl NamedEntry for MeasurementEntry {
    fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Clone)]
pub struct UpdateEntry {
    pub wrapped: DynUpdate,
    pub name: String,
    pub inv_name: String,
    pub weight: f64,
    pub ratio: f64,
    pub nprops: u64,
    pub naccs: u64,
    pub nimps: u64,
    pub cumulative_nprops: u64,
    pub cumulative_naccs: u64,
    pub cumulative_nimps: u64,
}

impl UpdateEntry {
    pub fn new<U>(update: U, name: impl Into<String>, weight: f64) -> Result<Self>
    where
        U: Update + Clone + 'static,
    {
        Self::from_dyn(DynUpdate::new(update), name, weight)
    }

    pub fn from_dyn(wrapped: DynUpdate, name: impl Into<String>, weight: f64) -> Result<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(RmcError::InvalidArgument(
                "update name must be non-empty".to_string(),
            ));
        }
        if weight < 0.0 {
            return Err(RmcError::InvalidArgument(format!(
                "update weight must be >= 0 (got {weight} on '{name}')"
            )));
        }

        Ok(Self {
            wrapped,
            inv_name: name.clone(),
            name,
            weight,
            ratio: 1.0,
            nprops: 0,
            naccs: 0,
            nimps: 0,
            cumulative_nprops: 0,
            cumulative_naccs: 0,
            cumulative_nimps: 0,
        })
    }

    pub fn reset_run_counters(&mut self) {
        self.nprops = 0;
        self.naccs = 0;
        self.nimps = 0;
    }

    pub fn accumulate_counters(&mut self) {
        self.cumulative_nprops += self.nprops;
        self.cumulative_naccs += self.naccs;
        self.cumulative_nimps += self.nimps;
        self.reset_run_counters();
    }
}

impl Update for UpdateEntry {
    fn attempt(&mut self, rng: &mut dyn RngCore) -> f64 {
        self.wrapped.attempt(rng)
    }

    fn accept(&mut self) {
        self.wrapped.accept();
    }

    fn reject(&mut self) {
        self.wrapped.reject();
    }
}

impl NamedEntry for UpdateEntry {
    fn name(&self) -> &str {
        &self.name
    }
}
