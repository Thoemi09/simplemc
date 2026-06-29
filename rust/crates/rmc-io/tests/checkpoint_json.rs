use rand_core::RngCore;
use rmc_core::{
    mc::{
        run_typed, Measurement, MetropolisKernel, SimulationParams, SingleUpdateSet, Update,
        UpdateSet,
    },
    random::{ChainId, DefaultRng, Rng, SeedSource},
};
use rmc_io::{
    from_json_str, load_json, load_payload_json, save_json, save_json_atomic, save_payload_json,
    to_json_string, Checkpoint, IoError, CHECKPOINT_VERSION,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Payload {
    step: u64,
    state: i64,
}

#[test]
fn checkpoint_round_trips_through_json_string_and_file() {
    let checkpoint = Checkpoint::new(Payload { step: 7, state: -3 });
    let encoded = to_json_string(&checkpoint).unwrap();
    let decoded: Checkpoint<Payload> = from_json_str(&encoded).unwrap();

    assert_eq!(decoded, checkpoint);
    assert_eq!(decoded.version, CHECKPOINT_VERSION);
    assert_eq!(decoded.payload().state, -3);

    let path =
        std::env::temp_dir().join(format!("rmc-checkpoint-json-{}.json", std::process::id()));
    save_json(&path, &checkpoint).unwrap();
    let loaded: Checkpoint<Payload> = load_json(&path).unwrap();
    assert_eq!(loaded, checkpoint);

    let payload_path = std::env::temp_dir().join(format!(
        "rmc-checkpoint-payload-{}.json",
        std::process::id()
    ));
    save_payload_json(&payload_path, checkpoint.payload()).unwrap();
    let payload: Payload = load_payload_json(&payload_path).unwrap();
    assert_eq!(payload, *checkpoint.payload());

    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(payload_path);
}

#[test]
fn load_rejects_unsupported_checkpoint_version() {
    let err =
        from_json_str::<Payload>(r#"{"version":999,"payload":{"step":1,"state":2}}"#).unwrap_err();

    assert!(matches!(
        err,
        IoError::UnsupportedVersion {
            expected: CHECKPOINT_VERSION,
            found: 999
        }
    ));
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct RandomWalkUpdate;

impl Update<i64> for RandomWalkUpdate {
    fn attempt<R: Rng + ?Sized>(&mut self, state: &mut i64, rng: &mut R) -> f64 {
        if rng.gen_bool(0.5) {
            *state += 1;
        } else {
            *state -= 1;
        }
        1.0
    }

    fn accept(&mut self, _state: &mut i64) {}
}

#[derive(Clone, Debug, Default)]
struct LastStateValue(i64);

#[derive(Serialize, Deserialize)]
struct RestartPayload {
    state: i64,
    rng: DefaultRng,
    kernel: MetropolisKernel<SingleUpdateSet<RandomWalkUpdate>>,
}

#[test]
fn checkpointed_run_resumes_same_trajectory_as_uninterrupted_run() {
    let seed = SeedSource::new(0x7eed);
    let full_params = SimulationParams {
        max_steps: 40,
        steps_per_cycle: 4,
        cycles_per_check: 1,
    };
    let first_params = SimulationParams {
        max_steps: 13,
        steps_per_cycle: 4,
        cycles_per_check: 1,
    };
    let second_params = SimulationParams {
        max_steps: 27,
        steps_per_cycle: 4,
        cycles_per_check: 1,
    };

    let mut full_rng = seed.rng_for(ChainId(0));
    let mut full_kernel = MetropolisKernel::new(SingleUpdateSet::new(RandomWalkUpdate));
    let (full_state, _full_stats, full_last) = run_typed(
        0,
        &mut full_rng,
        &mut full_kernel,
        LastStateValue::default(),
        full_params,
    )
    .unwrap();

    let mut split_rng = seed.rng_for(ChainId(0));
    let mut split_kernel = MetropolisKernel::new(SingleUpdateSet::new(RandomWalkUpdate));
    let (split_state, _first_stats, _first_last) = run_typed(
        0,
        &mut split_rng,
        &mut split_kernel,
        LastStateValue::default(),
        first_params,
    )
    .unwrap();

    let checkpoint = Checkpoint::new(RestartPayload {
        state: split_state,
        rng: split_rng,
        kernel: split_kernel,
    });
    let path = std::env::temp_dir().join(format!("rmc-restart-{}.json", std::process::id()));
    save_json_atomic(&path, &checkpoint).unwrap();

    let restored: Checkpoint<RestartPayload> = load_json(&path).unwrap();
    let RestartPayload {
        state,
        mut rng,
        mut kernel,
    } = restored.into_payload();
    let (resumed_state, _second_stats, resumed_last) = run_typed(
        state,
        &mut rng,
        &mut kernel,
        LastStateValue::default(),
        second_params,
    )
    .unwrap();

    assert_eq!(resumed_state, full_state);
    assert_eq!(resumed_last, full_last);
    assert_eq!(rng.next_u64(), full_rng.next_u64());
    assert_eq!(kernel.updates().stats(), full_kernel.updates().stats());

    let _ = std::fs::remove_file(path);
}

impl Measurement<i64> for LastStateValue {
    type Output = i64;

    fn measure(&mut self, state: &i64) {
        self.0 = *state;
    }

    fn finish(self) -> Self::Output {
        self.0
    }
}
