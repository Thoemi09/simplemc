use std::cell::Cell;
use std::rc::Rc;

use rmc_core::mc::{DynMeasurement, DynUpdate, Measurement, MeasurementEntry, Update, UpdateEntry};

#[derive(Clone)]
struct CounterMeasurement {
    count: Rc<Cell<i32>>,
}

impl CounterMeasurement {
    fn new() -> Self {
        Self {
            count: Rc::new(Cell::new(0)),
        }
    }
}

impl Measurement for CounterMeasurement {
    type Output = i32;

    fn measure(&mut self) {
        self.count.set(self.count.get() + 1);
    }

    fn finish(self) -> Self::Output {
        self.count.get()
    }
}

#[derive(Clone)]
struct ToyUpdate {
    accepted: Rc<Cell<i32>>,
    rejected: Rc<Cell<i32>>,
    prob: f64,
}

impl ToyUpdate {
    fn new(prob: f64) -> Self {
        Self {
            accepted: Rc::new(Cell::new(0)),
            rejected: Rc::new(Cell::new(0)),
            prob,
        }
    }
}

impl Update for ToyUpdate {
    fn attempt(&mut self, _rng: &mut dyn rand::RngCore) -> f64 {
        self.prob
    }

    fn accept(&mut self) {
        self.accepted.set(self.accepted.get() + 1);
    }

    fn reject(&mut self) {
        self.rejected.set(self.rejected.get() + 1);
    }
}

#[derive(Clone)]
struct MinimalUpdate {
    accepted: Rc<Cell<i32>>,
}

impl MinimalUpdate {
    fn new() -> Self {
        Self {
            accepted: Rc::new(Cell::new(0)),
        }
    }
}

impl Update for MinimalUpdate {
    fn attempt(&mut self, _rng: &mut dyn rand::RngCore) -> f64 {
        1.0
    }

    fn accept(&mut self) {
        self.accepted.set(self.accepted.get() + 1);
    }
}

#[test]
fn dyn_measurement_wraps_and_clones_unit_output_measurements() {
    let src = CounterMeasurement::new();
    let count = Rc::clone(&src.count);
    let mut first = DynMeasurement::new(UnitMeasurement(src));
    let mut second = first.clone();

    first.measure();
    second.measure();

    assert_eq!(count.get(), 2);
}

#[derive(Clone)]
struct UnitMeasurement(CounterMeasurement);

impl Measurement for UnitMeasurement {
    type Output = ();

    fn measure(&mut self) {
        self.0.measure();
    }

    fn finish(self) -> Self::Output {}
}

#[test]
fn measurement_entry_validates_and_forwards() {
    let src = CounterMeasurement::new();
    let count = Rc::clone(&src.count);
    let mut entry = MeasurementEntry::new_with_active(UnitMeasurement(src), "m", false).unwrap();

    assert_eq!(entry.name, "m");
    assert!(!entry.is_active);

    entry.measure();
    assert_eq!(count.get(), 1);
    assert!(MeasurementEntry::new(UnitMeasurement(CounterMeasurement::new()), "").is_err());
}

#[test]
fn dyn_update_wraps_and_clones() {
    let src = ToyUpdate::new(0.75);
    let accepted = Rc::clone(&src.accepted);
    let rejected = Rc::clone(&src.rejected);
    let mut first = DynUpdate::new(src);
    let mut second = first.clone();
    let mut rng = rmc_core::random::SeedSource::new(123).rng_for(rmc_core::random::ChainId(0));

    assert_eq!(first.attempt(&mut rng), 0.75);
    first.accept();
    second.reject();

    assert_eq!(accepted.get(), 1);
    assert_eq!(rejected.get(), 1);
}

#[test]
fn update_reject_defaults_to_noop() {
    let src = MinimalUpdate::new();
    let accepted = Rc::clone(&src.accepted);
    let mut update = DynUpdate::new(src);

    update.reject();

    assert_eq!(accepted.get(), 0);
}

#[test]
fn update_entry_validates_forwards_and_accumulates_counters() {
    let src = ToyUpdate::new(0.25);
    let accepted = Rc::clone(&src.accepted);
    let mut entry = UpdateEntry::new(src, "u", 2.5).unwrap();
    let mut rng = rmc_core::random::SeedSource::new(123).rng_for(rmc_core::random::ChainId(0));

    assert_eq!(entry.name, "u");
    assert_eq!(entry.inv_name, "u");
    assert_eq!(entry.weight, 2.5);
    assert_eq!(entry.ratio, 1.0);
    assert_eq!(entry.attempt(&mut rng), 0.25);

    entry.accept();
    assert_eq!(accepted.get(), 1);

    entry.nprops = 10;
    entry.naccs = 6;
    entry.nimps = 1;
    entry.accumulate_counters();
    assert_eq!(entry.cumulative_nprops, 10);
    assert_eq!(entry.cumulative_naccs, 6);
    assert_eq!(entry.cumulative_nimps, 1);
    assert_eq!(entry.nprops, 0);

    assert!(UpdateEntry::new(ToyUpdate::new(1.0), "", 1.0).is_err());
    assert!(UpdateEntry::new(ToyUpdate::new(1.0), "u", -1.0).is_err());
}
