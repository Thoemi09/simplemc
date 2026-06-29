use rmc_core::mc::{Measurement, MeasurementEntry, NamedSet};

#[derive(Clone)]
struct NoopMeasurement;

impl Measurement<()> for NoopMeasurement {
    type Output = ();

    fn measure(&mut self, _state: &()) {}

    fn finish(self) -> Self::Output {}
}

#[test]
fn named_set_registers_and_finds_entries_in_order() {
    let mut set = NamedSet::new();

    set.add(
        MeasurementEntry::new(NoopMeasurement, "a").unwrap(),
        "measurement",
    )
    .unwrap();
    set.add(
        MeasurementEntry::new_with_active(NoopMeasurement, "b", false).unwrap(),
        "measurement",
    )
    .unwrap();

    assert_eq!(set.len(), 2);
    assert_eq!(set.find("a"), Some(0));
    assert_eq!(set.find("b"), Some(1));
    assert_eq!(set.find("missing"), None);
    assert!(set[0].is_active);
    assert!(!set[1].is_active);
}

#[test]
fn named_set_rejects_duplicate_names() {
    let mut set = NamedSet::new();

    set.add(
        MeasurementEntry::new(NoopMeasurement, "a").unwrap(),
        "measurement",
    )
    .unwrap();
    let err = set
        .add(
            MeasurementEntry::new(NoopMeasurement, "a").unwrap(),
            "measurement",
        )
        .unwrap_err();

    assert_eq!(
        err.to_string(),
        "invalid argument: measurement 'a' is already registered"
    );
}

#[test]
fn named_set_mutates_entries_by_name() {
    let mut set = NamedSet::new();
    set.add(
        MeasurementEntry::new(NoopMeasurement, "a").unwrap(),
        "measurement",
    )
    .unwrap();

    set.get_by_name_mut("a").unwrap().is_active = false;

    assert!(!set.get_by_name("a").unwrap().is_active);
}
