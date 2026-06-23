use rand::RngCore;

use crate::{Result, RmcError};

use super::named_set::NamedEntry;
use super::traits::{Measurement, Update};
use super::wrappers::{DynMeasurement, DynUpdate};

// The boxed entry types below carry type-erased *stateless* updates/measurements (`Update<()>` /
// `Measurement<(), Output = ()>`). The monomorphized static path is the one that threads a real
// `State`; the runtime-flexible boxed path is specialized to `State = ()`.

#[derive(Clone)]
pub struct MeasurementEntry {
    pub wrapped: DynMeasurement,
    pub name: String,
    pub is_active: bool,
}

impl MeasurementEntry {
    pub fn new<M>(measurement: M, name: impl Into<String>) -> Result<Self>
    where
        M: Measurement<(), Output = ()> + Clone + 'static,
    {
        Self::new_with_active(measurement, name, true)
    }

    pub fn new_with_active<M>(
        measurement: M,
        name: impl Into<String>,
        is_active: bool,
    ) -> Result<Self>
    where
        M: Measurement<(), Output = ()> + Clone + 'static,
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

impl MeasurementEntry {
    /// Run the type-erased measurement (side-effecting; produces no typed output).
    pub fn measure(&mut self) {
        self.wrapped.measure();
    }
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
        U: Update<()> + Clone + 'static,
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

impl UpdateEntry {
    /// Forward an attempt to the type-erased inner update.
    ///
    /// Inherent (not an `Update` impl) for the same reason as [`DynUpdate::attempt`]: `Update` is
    /// object-unsafe under the generic-RNG signature, and the dyn boxed path needs a concrete RNG.
    pub fn attempt<R: RngCore>(&mut self, rng: &mut R) -> f64 {
        self.wrapped.attempt(rng)
    }

    pub fn accept(&mut self) {
        self.wrapped.accept();
    }

    pub fn reject(&mut self) {
        self.wrapped.reject();
    }
}

impl NamedEntry for UpdateEntry {
    fn name(&self) -> &str {
        &self.name
    }
}
