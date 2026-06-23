use rand::RngCore;

use super::traits::{Measurement, Update};

trait MeasurementObject {
    fn clone_box(&self) -> Box<dyn MeasurementObject>;
    fn measure(&mut self);
}

impl<T> MeasurementObject for T
where
    T: Measurement<(), Output = ()> + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn MeasurementObject> {
        Box::new(self.clone())
    }

    fn measure(&mut self) {
        Measurement::measure(self, &());
    }
}

/// Type-erased stateless measurement (`Measurement<(), Output = ()>`).
///
/// The boxed measurement path is intentionally side-effect-only — it cannot return a typed result
/// (see [`DynMeasurementSet`](crate::mc::DynMeasurementSet)). Use the typed
/// [`run_typed`](crate::mc::run_typed) path when results must flow back by ownership.
pub struct DynMeasurement {
    object: Box<dyn MeasurementObject>,
}

impl DynMeasurement {
    pub fn new<M>(measurement: M) -> Self
    where
        M: Measurement<(), Output = ()> + Clone + 'static,
    {
        Self {
            object: Box::new(measurement),
        }
    }

    pub fn measure(&mut self) {
        self.object.measure();
    }
}

impl Clone for DynMeasurement {
    fn clone(&self) -> Self {
        Self {
            object: self.object.clone_box(),
        }
    }
}

/// Object-safe bridge for type-erased stateless updates.
///
/// [`Update::attempt`] is generic over the RNG (object-unsafe), so the boxed path cannot store
/// `dyn Update<()>` directly. Instead it stores `dyn UpdateObject`, whose `attempt` takes a concrete
/// `&mut dyn RngCore`. The blanket impl below instantiates the generic `Update::attempt` with
/// `R = dyn RngCore`, which is valid because `dyn RngCore: Rng + ?Sized` (rand blanket impl). The
/// `()` state is folded away here since the boxed path is stateless.
trait UpdateObject {
    fn clone_box(&self) -> Box<dyn UpdateObject>;
    fn attempt(&mut self, rng: &mut dyn RngCore) -> f64;
    fn accept(&mut self);
    fn reject(&mut self);
}

impl<T> UpdateObject for T
where
    T: Update<()> + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn UpdateObject> {
        Box::new(self.clone())
    }

    fn attempt(&mut self, rng: &mut dyn RngCore) -> f64 {
        Update::attempt(self, &mut (), rng)
    }

    fn accept(&mut self) {
        Update::accept(self, &mut ());
    }

    fn reject(&mut self) {
        Update::reject(self, &mut ());
    }
}

/// Type-erased stateless update (`Update<()>`), used by the runtime-flexible boxed update set.
pub struct DynUpdate {
    object: Box<dyn UpdateObject>,
}

impl DynUpdate {
    pub fn new<U>(update: U) -> Self
    where
        U: Update<()> + Clone + 'static,
    {
        Self {
            object: Box::new(update),
        }
    }

    /// Forward an attempt to the type-erased inner update.
    ///
    /// Inherent (not an `Update` impl) because `Update::attempt` is generic over `R: ?Sized`, which
    /// cannot be coerced to the `&mut dyn RngCore` the boxed object requires. Callers always hold a
    /// concrete (sized) RNG here, so the coercion at the call boundary is valid.
    pub fn attempt<R: RngCore>(&mut self, rng: &mut R) -> f64 {
        self.object.attempt(rng)
    }

    pub fn accept(&mut self) {
        self.object.accept();
    }

    pub fn reject(&mut self) {
        self.object.reject();
    }
}

impl Clone for DynUpdate {
    fn clone(&self) -> Self {
        Self {
            object: self.object.clone_box(),
        }
    }
}
