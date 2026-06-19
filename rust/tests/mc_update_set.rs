use std::cell::Cell;
use std::rc::Rc;

use rmc_core::mc::{DynUpdateSet, Update, UpdateEntry};

#[derive(Clone)]
struct ToyUpdate {
    accepts: Rc<Cell<i32>>,
    prob: f64,
}

impl ToyUpdate {
    fn new(prob: f64) -> Self {
        Self {
            accepts: Rc::new(Cell::new(0)),
            prob,
        }
    }
}

impl Update for ToyUpdate {
    fn attempt(&mut self, _rng: &mut dyn rand::RngCore) -> f64 {
        self.prob
    }

    fn accept(&mut self) {
        self.accepts.set(self.accepts.get() + 1);
    }

    fn reject(&mut self) {}
}

fn update(name: &str, weight: f64) -> UpdateEntry {
    UpdateEntry::new(ToyUpdate::new(0.5), name, weight).unwrap()
}

#[test]
fn update_set_adds_and_finds_entries() {
    let mut set = DynUpdateSet::new();
    set.add(update("a", 1.0)).unwrap();
    set.add(update("b", 2.0)).unwrap();

    assert_eq!(set.len(), 2);
    assert!(!set.is_empty());
    assert_eq!(set.find("a"), Some(0));
    assert_eq!(set.find("b"), Some(1));
    assert_eq!(set.find("missing"), None);
    assert_eq!(set[0].name, "a");
    assert_eq!(set[1].weight, 2.0);
    assert_eq!(set[0].inv_name, "a");
    assert_eq!(set[0].ratio, 1.0);
}

#[test]
fn update_set_rejects_duplicates_and_invalid_weights() {
    let mut set = DynUpdateSet::new();
    set.add(update("a", 1.0)).unwrap();

    assert!(set.add(update("a", 1.0)).is_err());
    assert!(set.set_weight("a", -1.0).is_err());
    assert!(set.set_weight("missing", 1.0).is_err());
}

#[test]
fn update_set_add_pair_cross_links_and_recomputes_ratios() {
    let mut set = DynUpdateSet::new();
    set.add_pair(update("f", 2.0), update("b", 3.0)).unwrap();

    assert_eq!(set.len(), 2);
    assert_eq!(set[0].inv_name, "b");
    assert_eq!(set[1].inv_name, "f");

    set.rebuild_distribution().unwrap();
    assert_eq!(set[0].ratio, 1.5);
    assert_eq!(set[1].ratio, 2.0 / 3.0);

    set.set_weight("f", 1.0).unwrap();
    set.rebuild_distribution().unwrap();
    assert_eq!(set[0].ratio, 3.0);
    assert_eq!(set[1].ratio, 1.0 / 3.0);
}

#[test]
fn update_set_validates_inverse_pairs() {
    let mut set = DynUpdateSet::new();
    assert!(set.add_pair(update("f", 0.0), update("b", 1.0)).is_err());
    assert!(set.add_pair(update("x", 1.0), update("x", 1.0)).is_err());

    set.add(update("a", 1.0)).unwrap();
    assert!(set.add_pair(update("a", 1.0), update("b", 1.0)).is_err());
}

#[test]
fn update_set_rebuild_handles_zero_pairs_and_all_zero_failure() {
    let mut set = DynUpdateSet::new();
    set.add(update("a", 1.0)).unwrap();
    set.add_pair(update("f", 0.0), update("b", 0.0)).unwrap();

    set.rebuild_distribution().unwrap();
    assert_eq!(set[1].ratio, 1.0);
    assert_eq!(set[2].ratio, 1.0);

    set.set_weight("b", 2.0).unwrap();
    assert!(set.rebuild_distribution().is_err());

    let mut all_zero = DynUpdateSet::new();
    all_zero.add(update("z", 0.0)).unwrap();
    assert!(all_zero.rebuild_distribution().is_err());
}

#[test]
fn update_set_accumulates_and_resets_counters() {
    let mut set = DynUpdateSet::new();
    set.add(update("a", 1.0)).unwrap();
    set.add(update("b", 1.0)).unwrap();

    set[0].nprops = 10;
    set[0].naccs = 6;
    set[1].nprops = 4;
    set.accumulate_counters();

    assert_eq!(set[0].cumulative_nprops, 10);
    assert_eq!(set[0].cumulative_naccs, 6);
    assert_eq!(set[1].cumulative_nprops, 4);
    assert_eq!(set[0].nprops, 0);
    assert_eq!(set[1].nprops, 0);

    set[0].nprops = 2;
    set.reset_run_counters();
    assert_eq!(set[0].nprops, 0);
    assert_eq!(set[0].cumulative_nprops, 10);
}
