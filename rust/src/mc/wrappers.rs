use rand::RngCore;

use super::traits::{Measurement, Update};

trait MeasurementObject {
    fn clone_box(&self) -> Box<dyn MeasurementObject>;
    fn measure(&mut self);
}

impl<T> MeasurementObject for T
where
    T: Measurement<Output = ()> + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn MeasurementObject> {
        Box::new(self.clone())
    }

    fn measure(&mut self) {
        Measurement::measure(self);
    }
}

pub struct DynMeasurement {
    object: Box<dyn MeasurementObject>,
}

impl DynMeasurement {
    pub fn new<M>(measurement: M) -> Self
    where
        M: Measurement<Output = ()> + Clone + 'static,
    {
        Self {
            object: Box::new(measurement),
        }
    }
}

impl Clone for DynMeasurement {
    fn clone(&self) -> Self {
        Self {
            object: self.object.clone_box(),
        }
    }
}

impl Measurement for DynMeasurement {
    type Output = ();

    fn measure(&mut self) {
        self.object.measure();
    }

    fn finish(self) -> Self::Output {}
}

trait UpdateObject {
    fn clone_box(&self) -> Box<dyn UpdateObject>;
    fn attempt(&mut self, rng: &mut dyn RngCore) -> f64;
    fn accept(&mut self);
    fn reject(&mut self);
}

impl<T> UpdateObject for T
where
    T: Update + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn UpdateObject> {
        Box::new(self.clone())
    }

    fn attempt(&mut self, rng: &mut dyn RngCore) -> f64 {
        Update::attempt(self, rng)
    }

    fn accept(&mut self) {
        Update::accept(self);
    }

    fn reject(&mut self) {
        Update::reject(self);
    }
}

pub struct DynUpdate {
    object: Box<dyn UpdateObject>,
}

impl DynUpdate {
    pub fn new<U>(update: U) -> Self
    where
        U: Update + Clone + 'static,
    {
        Self {
            object: Box::new(update),
        }
    }
}

impl Clone for DynUpdate {
    fn clone(&self) -> Self {
        Self {
            object: self.object.clone_box(),
        }
    }
}

impl Update for DynUpdate {
    fn attempt(&mut self, rng: &mut dyn RngCore) -> f64 {
        self.object.attempt(rng)
    }

    fn accept(&mut self) {
        self.object.accept();
    }

    fn reject(&mut self) {
        self.object.reject();
    }
}
