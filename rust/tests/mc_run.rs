use std::cell::Cell;
use std::rc::Rc;

use rmc_core::mc::{
    run, run_typed, run_with_callbacks, DynMeasurementSet, DynMetropolisKernel, DynUpdateSet,
    Measurement, MeasurementEntry, RunCallbacks, SimulationCtx, SimulationParams, SingleUpdateSet,
    StaticMetropolisKernel, SteppingUpdateSet, TwoUpdateSet, Update, UpdateEntry, UpdateSet,
};
use rmc_core::random::{ChainId, SeedSource};

#[derive(Clone)]
struct IncrementUpdate {
    value: Rc<Cell<i32>>,
}

impl Update for IncrementUpdate {
    fn attempt(&mut self, _rng: &mut dyn rand::RngCore) -> f64 {
        1.0
    }

    fn accept(&mut self) {
        self.value.set(self.value.get() + 1);
    }
}

#[derive(Clone)]
struct CounterMeasurement {
    count: Rc<Cell<i32>>,
}

impl Measurement for CounterMeasurement {
    type Output = ();

    fn measure(&mut self) {
        self.count.set(self.count.get() + 1);
    }

    fn finish(self) -> Self::Output {}
}

#[derive(Clone)]
struct TypedCounterMeasurement {
    count: u64,
}

impl Measurement for TypedCounterMeasurement {
    type Output = u64;

    fn measure(&mut self) {
        self.count += 1;
    }

    fn finish(self) -> Self::Output {
        self.count
    }
}

#[test]
fn run_executes_dynamic_metropolis_prototype() {
    let value = Rc::new(Cell::new(0));
    let measurement_count = Rc::new(Cell::new(0));

    let mut updates = DynUpdateSet::new();
    updates
        .add(
            UpdateEntry::new(
                IncrementUpdate {
                    value: Rc::clone(&value),
                },
                "inc",
                1.0,
            )
            .unwrap(),
        )
        .unwrap();

    let mut measurements = DynMeasurementSet::new();
    measurements
        .add(
            MeasurementEntry::new(
                CounterMeasurement {
                    count: Rc::clone(&measurement_count),
                },
                "counter",
            )
            .unwrap(),
        )
        .unwrap();

    let mut kernel = DynMetropolisKernel::new(updates);
    let mut rng = SeedSource::new(123).rng_for(ChainId(0));
    let stats = run(
        &mut rng,
        &mut kernel,
        &mut measurements,
        SimulationParams {
            max_steps: 10,
            steps_per_cycle: 2,
            cycles_per_check: 1,
        },
    )
    .unwrap();

    assert_eq!(value.get(), 10);
    assert_eq!(measurement_count.get(), 5);
    assert_eq!(stats.steps_done, 10);
    assert_eq!(stats.cycles_done, 5);

    let updates = kernel.into_updates();
    assert_eq!(updates[0].nprops, 10);
    assert_eq!(updates[0].naccs, 10);
    assert_eq!(updates[0].nimps, 0);
}

#[derive(Default)]
struct StopAfterTwoCycles;

impl RunCallbacks<SimulationCtx> for StopAfterTwoCycles {
    fn stop_when(&mut self, ctx: &SimulationCtx) -> bool {
        ctx.cycles_done >= 2
    }
}

#[test]
fn run_callbacks_can_stop_the_loop() {
    let value = Rc::new(Cell::new(0));
    let mut updates = DynUpdateSet::new();
    updates
        .add(
            UpdateEntry::new(
                IncrementUpdate {
                    value: Rc::clone(&value),
                },
                "inc",
                1.0,
            )
            .unwrap(),
        )
        .unwrap();

    let mut measurements = DynMeasurementSet::new();
    let mut kernel = DynMetropolisKernel::new(updates);
    let mut rng = SeedSource::new(123).rng_for(ChainId(0));
    let mut callbacks = StopAfterTwoCycles;

    let stats = run_with_callbacks(
        &mut rng,
        &mut kernel,
        &mut measurements,
        SimulationParams {
            max_steps: 100,
            steps_per_cycle: 3,
            cycles_per_check: 1,
        },
        &mut callbacks,
    )
    .unwrap();

    assert_eq!(stats.steps_done, 6);
    assert_eq!(stats.cycles_done, 2);
    assert_eq!(value.get(), 6);
}

#[test]
fn run_typed_returns_measurement_output_by_ownership() {
    let value = Rc::new(Cell::new(0));
    let mut updates = DynUpdateSet::new();
    updates
        .add(
            UpdateEntry::new(
                IncrementUpdate {
                    value: Rc::clone(&value),
                },
                "inc",
                1.0,
            )
            .unwrap(),
        )
        .unwrap();

    let mut kernel = DynMetropolisKernel::new(updates);
    let mut rng = SeedSource::new(123).rng_for(ChainId(0));

    let (stats, measurement_count) = run_typed(
        &mut rng,
        &mut kernel,
        TypedCounterMeasurement { count: 0 },
        SimulationParams {
            max_steps: 7,
            steps_per_cycle: 3,
            cycles_per_check: 1,
        },
    )
    .unwrap();

    assert_eq!(value.get(), 7);
    assert_eq!(stats.steps_done, 7);
    assert_eq!(stats.cycles_done, 3);
    assert_eq!(measurement_count, 3);
}

#[test]
fn run_rejects_zero_steps_per_cycle() {
    let mut updates = DynUpdateSet::new();
    updates
        .add(
            UpdateEntry::new(
                IncrementUpdate {
                    value: Rc::new(Cell::new(0)),
                },
                "inc",
                1.0,
            )
            .unwrap(),
        )
        .unwrap();

    let mut measurements = DynMeasurementSet::new();
    let mut kernel = DynMetropolisKernel::new(updates);
    let mut rng = SeedSource::new(123).rng_for(ChainId(0));

    let err = run(
        &mut rng,
        &mut kernel,
        &mut measurements,
        SimulationParams {
            max_steps: 10,
            steps_per_cycle: 0,
            cycles_per_check: 1,
        },
    )
    .unwrap_err();

    assert_eq!(
        err.to_string(),
        "invalid argument: steps_per_cycle must be > 0"
    );
}

#[derive(Clone)]
struct CountingUpdate {
    accepted: Rc<Cell<i32>>,
    rejected: Rc<Cell<i32>>,
    probability: f64,
}

impl Update for CountingUpdate {
    fn attempt(&mut self, _rng: &mut dyn rand::RngCore) -> f64 {
        self.probability
    }

    fn accept(&mut self) {
        self.accepted.set(self.accepted.get() + 1);
    }

    fn reject(&mut self) {
        self.rejected.set(self.rejected.get() + 1);
    }
}

#[test]
fn run_typed_works_with_static_single_update_set() {
    let accepted = Rc::new(Cell::new(0));
    let rejected = Rc::new(Cell::new(0));
    let updates = SingleUpdateSet::new(CountingUpdate {
        accepted: Rc::clone(&accepted),
        rejected: Rc::clone(&rejected),
        probability: 1.0,
    });
    let mut kernel = StaticMetropolisKernel::new(updates);
    let mut rng = SeedSource::new(123).rng_for(ChainId(0));

    let (stats, measurement_count) = run_typed(
        &mut rng,
        &mut kernel,
        TypedCounterMeasurement { count: 0 },
        SimulationParams {
            max_steps: 8,
            steps_per_cycle: 4,
            cycles_per_check: 1,
        },
    )
    .unwrap();

    assert_eq!(accepted.get(), 8);
    assert_eq!(rejected.get(), 0);
    assert_eq!(stats.steps_done, 8);
    assert_eq!(measurement_count, 2);
    assert_eq!(kernel.updates().stats()[0].nprops, 8);
    assert_eq!(kernel.updates().stats()[0].naccs, 8);
    assert_eq!(kernel.updates().stats()[0].nimps, 0);
}

#[test]
fn static_single_update_set_tracks_rejections_and_impossible_updates() {
    let accepted = Rc::new(Cell::new(0));
    let rejected = Rc::new(Cell::new(0));
    let mut rng = SeedSource::new(123).rng_for(ChainId(0));

    let mut rejected_set = SingleUpdateSet::new(CountingUpdate {
        accepted: Rc::clone(&accepted),
        rejected: Rc::clone(&rejected),
        probability: 0.0,
    });
    let rejected_outcome = rejected_set.select_and_step(&mut rng).unwrap();
    assert!(!rejected_outcome.accepted);
    assert!(!rejected_outcome.impossible);
    assert_eq!(accepted.get(), 0);
    assert_eq!(rejected.get(), 1);
    assert_eq!(rejected_set.stats()[0].nprops, 1);
    assert_eq!(rejected_set.stats()[0].naccs, 0);
    assert_eq!(rejected_set.stats()[0].nimps, 0);

    let mut impossible_set = SingleUpdateSet::new(CountingUpdate {
        accepted: Rc::clone(&accepted),
        rejected: Rc::clone(&rejected),
        probability: -1.0,
    });
    let impossible_outcome = impossible_set.select_and_step(&mut rng).unwrap();
    assert!(!impossible_outcome.accepted);
    assert!(impossible_outcome.impossible);
    assert_eq!(accepted.get(), 0);
    assert_eq!(rejected.get(), 2);
    assert_eq!(impossible_set.stats()[0].nprops, 1);
    assert_eq!(impossible_set.stats()[0].naccs, 0);
    assert_eq!(impossible_set.stats()[0].nimps, 1);
}

#[test]
fn two_update_set_runs_two_static_update_types() {
    let first_accepted = Rc::new(Cell::new(0));
    let first_rejected = Rc::new(Cell::new(0));
    let second_accepted = Rc::new(Cell::new(0));
    let second_rejected = Rc::new(Cell::new(0));

    let updates = TwoUpdateSet::new(
        CountingUpdate {
            accepted: Rc::clone(&first_accepted),
            rejected: Rc::clone(&first_rejected),
            probability: 1.0,
        },
        0.0,
        CountingUpdate {
            accepted: Rc::clone(&second_accepted),
            rejected: Rc::clone(&second_rejected),
            probability: 1.0,
        },
        1.0,
    )
    .unwrap();
    let mut kernel = StaticMetropolisKernel::new(updates);
    let mut rng = SeedSource::new(123).rng_for(ChainId(0));

    let (stats, measurement_count) = run_typed(
        &mut rng,
        &mut kernel,
        TypedCounterMeasurement { count: 0 },
        SimulationParams {
            max_steps: 6,
            steps_per_cycle: 2,
            cycles_per_check: 1,
        },
    )
    .unwrap();

    assert_eq!(stats.steps_done, 6);
    assert_eq!(measurement_count, 3);
    assert_eq!(first_accepted.get(), 0);
    assert_eq!(second_accepted.get(), 6);
    assert_eq!(kernel.updates().stats()[0].nprops, 0);
    assert_eq!(kernel.updates().stats()[1].nprops, 6);
    assert_eq!(kernel.updates().stats()[1].naccs, 6);
}

#[test]
fn two_update_set_inverse_pair_sets_ratios() {
    let first = CountingUpdate {
        accepted: Rc::new(Cell::new(0)),
        rejected: Rc::new(Cell::new(0)),
        probability: 0.5,
    };
    let second = CountingUpdate {
        accepted: Rc::new(Cell::new(0)),
        rejected: Rc::new(Cell::new(0)),
        probability: 0.5,
    };

    let updates = TwoUpdateSet::inverse_pair(first, 2.0, second, 4.0).unwrap();

    assert_eq!(updates.weights(), [2.0, 4.0]);
    assert_eq!(updates.ratios(), [2.0, 0.5]);
}

#[test]
fn two_update_set_validates_weights() {
    let first = CountingUpdate {
        accepted: Rc::new(Cell::new(0)),
        rejected: Rc::new(Cell::new(0)),
        probability: 1.0,
    };
    let second = first.clone();

    assert!(TwoUpdateSet::new(first.clone(), 0.0, second.clone(), 0.0).is_err());
    assert!(TwoUpdateSet::new(first.clone(), -1.0, second.clone(), 1.0).is_err());
    assert!(TwoUpdateSet::inverse_pair(first.clone(), 0.0, second.clone(), 1.0).is_err());
    assert!(TwoUpdateSet::inverse_pair(first, 0.0, second, 0.0).is_err());
}

#[derive(Clone)]
struct RandomWalkUpdate {
    position: Rc<Cell<i32>>,
    pending_delta: i32,
}

impl RandomWalkUpdate {
    fn new(position: Rc<Cell<i32>>) -> Self {
        Self {
            position,
            pending_delta: 0,
        }
    }
}

impl Update for RandomWalkUpdate {
    fn attempt(&mut self, rng: &mut dyn rand::RngCore) -> f64 {
        self.pending_delta = if rng.next_u64() & 1 == 0 { 1 } else { -1 };
        1.0
    }

    fn accept(&mut self) {
        self.position.set(self.position.get() + self.pending_delta);
    }
}

#[test]
fn update_attempt_can_draw_from_chain_rng() {
    fn run_walk(seed: u64) -> i32 {
        let position = Rc::new(Cell::new(0));
        let updates = SingleUpdateSet::new(RandomWalkUpdate::new(Rc::clone(&position)));
        let mut kernel = StaticMetropolisKernel::new(updates);
        let mut rng = SeedSource::new(seed).rng_for(ChainId(0));

        run_typed(
            &mut rng,
            &mut kernel,
            TypedCounterMeasurement { count: 0 },
            SimulationParams {
                max_steps: 64,
                steps_per_cycle: 8,
                cycles_per_check: 1,
            },
        )
        .unwrap();

        position.get()
    }

    assert_eq!(run_walk(123), run_walk(123));
}
