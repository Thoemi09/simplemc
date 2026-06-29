use std::cell::Cell;
use std::rc::Rc;

use rmc_core::mc::{DynMeasurementSet, Measurement, MeasurementEntry};

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

impl Measurement<()> for CounterMeasurement {
    type Output = ();

    fn measure(&mut self, _state: &()) {
        self.count.set(self.count.get() + 1);
    }

    fn finish(self) -> Self::Output {}
}

#[test]
fn measurement_set_adds_and_finds_entries() {
    let mut set = DynMeasurementSet::new();

    set.add(MeasurementEntry::new(CounterMeasurement::new(), "a").unwrap())
        .unwrap();
    set.add(MeasurementEntry::new_with_active(CounterMeasurement::new(), "b", false).unwrap())
        .unwrap();

    assert_eq!(set.len(), 2);
    assert!(!set.is_empty());
    assert_eq!(set.find("a"), Some(0));
    assert_eq!(set.find("missing"), None);
    assert!(set[0].is_active);
    assert!(!set[1].is_active);
}

#[test]
fn measurement_set_rejects_duplicate_names_and_missing_active_set() {
    let mut set = DynMeasurementSet::new();
    set.add(MeasurementEntry::new(CounterMeasurement::new(), "a").unwrap())
        .unwrap();

    assert!(set
        .add(MeasurementEntry::new(CounterMeasurement::new(), "a").unwrap())
        .is_err());
    assert!(set.set_active("missing", false).is_err());
}

#[test]
fn measurement_set_refreshes_active_cache_explicitly() {
    let mut set = DynMeasurementSet::new();
    set.add(MeasurementEntry::new(CounterMeasurement::new(), "a").unwrap())
        .unwrap();
    set.add(MeasurementEntry::new_with_active(CounterMeasurement::new(), "b", false).unwrap())
        .unwrap();
    set.add(MeasurementEntry::new(CounterMeasurement::new(), "c").unwrap())
        .unwrap();

    set.refresh_active();
    assert_eq!(set.active_indices(), &[0, 2]);

    set.set_active("a", false).unwrap();
    assert_eq!(set.active_indices(), &[0, 2]);

    set.refresh_active();
    assert_eq!(set.active_indices(), &[2]);

    set.clear_active();
    assert!(set.active_indices().is_empty());
}

#[test]
fn measurement_set_measures_only_cached_active_entries() {
    let first = CounterMeasurement::new();
    let second = CounterMeasurement::new();
    let third = CounterMeasurement::new();
    let first_count = Rc::clone(&first.count);
    let second_count = Rc::clone(&second.count);
    let third_count = Rc::clone(&third.count);

    let mut set = DynMeasurementSet::new();
    set.add(MeasurementEntry::new(first, "a").unwrap()).unwrap();
    set.add(MeasurementEntry::new_with_active(second, "b", false).unwrap())
        .unwrap();
    set.add(MeasurementEntry::new(third, "c").unwrap()).unwrap();

    set.refresh_active();
    set.measure_all();
    set.measure_all();

    assert_eq!(first_count.get(), 2);
    assert_eq!(second_count.get(), 0);
    assert_eq!(third_count.get(), 2);
}
