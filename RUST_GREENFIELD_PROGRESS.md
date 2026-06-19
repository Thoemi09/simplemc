# Greenfield Rust progress

This document tracks implementation progress after switching to the greenfield design in
`RUST_GREENFIELD_PLAN.md`.

## Current layout

The active crate still lives under `rust/`, but it has been renamed to `rmc-core`.

- `rust/Cargo.toml`: `rmc-core` crate.
- `rust/src/error.rs`: `RmcError` and crate-local `Result`.
- `rust/src/merge.rs`: `Merge` trait.
- `rust/src/random/`: analytical sample/PDF helpers plus `SeedSource` using `rand_xoshiro`.
- `rust/src/scalar.rs`: minimal `Scalar` / `SampleType`.
- `rust/src/mc/`: MC traits, dynamic fallback wrappers/sets, and update/measurement metadata.
- `rust/src/mc/run.rs`: sequential prototype run loop.
- `rust/src/mc/parallel.rs`: rayon-backed multi-chain run loops.
- `rust/src/mc/kernel.rs`: dynamic, static, and stateful Metropolis kernels.
- `rust/examples/random_walk.rs`: first executable prototype example.
- `rust/examples/ising_2d.rs`: first physical Monte Carlo example.
- `rust/tests/`: native Rust tests using closed-form checks and behavior tests, not C++ fixtures.

## Completed

### Greenfield pivot

- Renamed the crate package from `simplemc` to `rmc-core`.
- Removed active C++ parity fixture artifacts:
  - hand-ported `splitmix64`
  - hand-ported `xoshiro256`
  - `seed_rng` / `std::seed_seq` parity implementation
  - C++ RNG fixture JSON
  - C++ fixture dumper
  - RNG parity tests
- Replaced public RNG API with:
  - `SeedSource`
  - `ChainId`
  - `DefaultRng = rand_xoshiro::Xoshiro256PlusPlus`
- Rewrote random helper tests to use closed-form analytical checks instead of C++ fixture values.

### Core traits

- Added `Merge`.
- Renamed the public error type from `SimplemcError` to `RmcError`.
- Updated `Measurement` to use typed ownership:
  - associated `Output`
  - `finish(self) -> Self::Output`
- Updated `Update` so `attempt` receives the active chain RNG.
- Added first stateful traits:
  - `StatefulUpdate<State>`
  - `StatefulMeasurement<State>`
  - `StatefulKernel<State, R>`
  - `SteppingStatefulUpdateSet<State, R>`
- Added initial `StepOutcome`, `UpdateStats`, and `UpdateSet` trait skeletons.

### Dynamic fallback path

- Renamed boxed wrappers to make their role explicit:
  - `DynMeasurement`
  - `DynUpdate`
  - `DynMeasurementSet`
  - `DynUpdateSet`
- Removed public `Any` downcast APIs from dynamic wrappers and sets.
- Dynamic measurements are restricted to `Measurement<Output = ()>`, because typed result retrieval
  belongs on concrete measurement values in the greenfield design.

### Sequential prototype

- Added `SimulationParams`, `SimulationStats`, and `SimulationCtx`.
- Added `run` and `run_with_callbacks`.
- Added `run_typed` and `run_typed_with_callbacks`, which return typed measurement output through
  `Measurement::finish(self)`.
- Added `DynMetropolisKernel` over `DynUpdateSet`.
- Added weighted dynamic update selection using `rand::distributions::WeightedIndex`.
- Added `SimulationParams::validate`; `steps_per_cycle = 0` is rejected to avoid non-terminating
  runs.
- Added prototype end-to-end tests:
  - a one-dimensional always-accepted update runs for the requested number of steps
  - active measurements run once per cycle
  - callbacks can stop the run loop
  - typed measurements return output by ownership
  - invalid run parameters fail fast

### First static hot-path slice

- Added `SteppingUpdateSet<R>` as the trait used by kernels that can select and execute updates.
- Added `SingleUpdateSet<U>` for one concrete update type with static dispatch.
- Added `SingleStatefulUpdateSet<U>` for one concrete stateful update type with static dispatch.
- Added `TwoUpdateSet<A, B>` for two concrete update types with static dispatch and weighted
  selection.
- Added `StaticMetropolisKernel<S>` generic over any `SteppingUpdateSet`.
- Added `StatefulMetropolisKernel<S>` generic over any `SteppingStatefulUpdateSet`.
- Kept `DynUpdateSet` compatible by implementing `SteppingUpdateSet` for it.
- Added tests covering:
  - typed run with `StaticMetropolisKernel<SingleUpdateSet<_>>`
  - typed run with `StaticMetropolisKernel<TwoUpdateSet<_, _>>`
  - stateful typed run with owned state returned to the caller
  - stateful parallel run with independent chain states and merged outputs
  - accepted update stats
  - rejected update stats
  - impossible update stats
  - inverse-pair detailed-balance ratios
  - static update weight validation

### First parallel run slice

- Added `rayon`.
- Added `ParallelConfig`.
- Added `run_parallel`.
- Added `run_parallel_in_pool` so tests and callers can choose a specific rayon thread pool.
- Added `run_stateful_parallel` and `run_stateful_parallel_in_pool`.
- `run_parallel`:
  - validates `chains > 0`
  - derives one RNG per chain from `SeedSource`
  - builds independent `(kernel, measurement)` pairs through a user closure
  - runs each chain with `run_typed`
  - merges `SimulationStats` and `Measurement::Output` through `Merge`
- `run_stateful_parallel`:
  - builds independent `(state, kernel, measurement)` triples
  - runs each chain with `run_stateful_typed`
  - returns merged `SimulationStats` and typed measurement output
- Added `Merge` impls for common scalar outputs used by prototype tests.
- Added tests for:
  - merging independent chain outputs
  - rejecting zero chains
  - reproducibility across one-thread and four-thread rayon pools for a stochastic update

### RNG-backed update proposals

- Changed `Update::attempt` to receive the active chain RNG as `&mut dyn rand::RngCore`.
- Updated dynamic wrappers, `UpdateEntry`, `DynUpdateSet`, `SingleUpdateSet`, and `TwoUpdateSet` to
  pass the same framework-managed RNG into update attempts.
- Added a test proving updates can draw proposal randomness from the chain RNG while remaining
  reproducible for the same seed.

### First example

- Added `rust/examples/random_walk.rs`.
- The example demonstrates:
  - static update dispatch via `StaticMetropolisKernel<SingleUpdateSet<_>>`
  - typed measurement output via `run_typed`
  - rayon-backed multi-chain execution via `run_parallel`
  - update proposal randomness through the framework-managed chain RNG

### First physical example

- Added `rust/examples/ising_2d.rs`.
- The example implements a compact 16x16 periodic 2D Ising model with single-spin Metropolis
  updates.
- It demonstrates:
  - a realistic update that proposes a random spin index and computes an acceptance probability from
    the state
  - a measurement that samples magnetization once per cycle
  - typed result reduction across rayon chains through `Merge`
  - sequential and parallel execution with the same engine API
- The example intentionally uses `Arc<Mutex<IsingLattice>>` to share state between update and
  measurement values in its original version. The current version uses the stateful API instead:
  each chain owns its `IsingLattice`, updates receive `&mut IsingLattice`, and measurements receive
  `&IsingLattice`.

### First state-owned engine slice

- Added `run_stateful_typed` and `run_stateful_typed_with_callbacks`.
- Added `run_stateful_parallel` and `run_stateful_parallel_in_pool`.
- `run_stateful_typed` returns `(State, SimulationStats, M::Output)` so callers can inspect or later
  checkpoint the final chain state.
- Ported `rust/examples/ising_2d.rs` away from `Arc<Mutex<IsingLattice>>`.
- Added `Merge for i64`, useful for signed observables and stateful prototype tests.

## Verification

Current verification command set:

```text
cd rust
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
```

After the RNG-backed proposal and example pass, the command set passed with 39 Rust tests green. The
example was also run manually:

```text
cd rust
cargo run --example random_walk
```

It printed:

```text
single chain: steps=1000, cycles=100, final_position=40
parallel: chains=8, steps=8000, cycles=800, mean_final_position=-2.250
```

After the `RmcError` rename, the same verification command set passed again with 39 Rust tests
green.

After adding the 2D Ising example, the verification command set passed again with 39 Rust tests
green. The example was also run manually:

```text
cd rust
cargo run --example ising_2d
```

It printed:

```text
single chain: steps=20000, samples=79, mean_m=0.658, mean_abs_m=0.658
parallel: chains=8, steps=160000, samples=632, mean_m=0.738, mean_abs_m=0.738
```

After adding the first state-owned engine slice and porting the Ising example to it, the
verification command set passed with 41 Rust tests green.

## Problems encountered

### Pivot invalidated parity code

The faithful-port RNGs and C++ fixtures were correct for `RUST_PORT_PLAN.md`, but they directly
contradicted `RUST_GREENFIELD_PLAN.md`.

Resolution: removed them from the active crate and switched to `rand_xoshiro`.

### Package rename changed test imports

Renaming the package to `rmc-core` changed the crate import path to `rmc_core`.

Resolution: updated integration tests to import `rmc_core`.

### Measurement outputs changed the dynamic-wrapper design

The greenfield `Measurement` trait returns typed results through `finish(self)`. A heterogeneous
boxed measurement set cannot expose arbitrary typed outputs without reintroducing downcasting.

Resolution: dynamic measurements are a side-effect-only fallback with `Output = ()`. Typed
measurements should stay concrete and be returned by ownership in future run APIs.

### New ecosystem RNG dependencies required fetching

Adding `rand` and `rand_xoshiro` required Cargo to fetch new crates.

Impact: low after dependencies are cached and `Cargo.lock` is updated.

### Rayon dependency required fetching

Adding `run_parallel` introduced `rayon` and its crossbeam dependencies.

Impact: low after dependencies are cached and `Cargo.lock` is updated.

### Temporary dynamic Metropolis kernel is not the final hot path

`DynMetropolisKernel` uses `DynUpdateSet` and boxed updates. This is acceptable for a prototype but
does not satisfy the greenfield goal of monomorphized hot-path update dispatch.

Resolution: mark it as the dynamic fallback. The next engine iteration should add a concrete typed
update-set path and keep the dynamic kernel for runtime/config-driven workflows.

### Kernel generic bound was too loose for the trait definition

The first `DynMetropolisKernel` impl used `R: Rng + ?Sized`, but `Kernel<R>` still had the implicit
`R: Sized` bound.

Resolution: kept the simpler sized RNG bound for the prototype.

### Public re-export missed `run_with_callbacks`

The function existed in `mc::run` but was not re-exported through `mc::mod`.

Resolution: exported it and added callback-stop coverage.

### `steps_per_cycle = 0` could make runs non-terminating

The initial run loop incremented steps only inside `0..steps_per_cycle`. With `steps_per_cycle = 0`,
the loop could continue cycling without making progress.

Resolution: added `SimulationParams::validate` and reject zero `steps_per_cycle` before preparing
the kernel.

### Static dispatch is still limited to one or two update types

`SingleUpdateSet<U>` and `TwoUpdateSet<A, B>` prove the static hot-path shape, but real simulations
may need more update types with ergonomic weighted selection.

Resolution: keep `SingleUpdateSet` and `TwoUpdateSet` as minimal static baselines. Next options are
a tuple/HList implementation, a user enum-based update set, or a small derive macro later.

### Parallel reduction needed explicit accumulator type

The first `run_parallel` reduction used `None` without enough type context for Rust to infer the
merged output type.

Resolution: annotated the merged accumulator as `Option<(SimulationStats, M::Output)>`.

### Reproducibility across thread counts needs configurable pools

The default `run_parallel` uses rayon's global pool, which is fine for users but awkward for testing
specific thread counts.

Resolution: added `run_parallel_in_pool`, then tested the same seeded stochastic simulation on
one-thread and four-thread pools.

### Update proposals originally could not use the chain RNG

The initial `Update::attempt() -> f64` API allowed deterministic acceptance probabilities, but it
did not let an update draw a proposal from the framework-managed per-chain RNG. Real Monte Carlo
updates commonly need that proposal randomness, and putting private RNGs inside updates would weaken
the reproducibility model.

Resolution: changed `Update::attempt` to `attempt(&mut self, rng: &mut dyn rand::RngCore) -> f64`
and threaded that RNG through dynamic and static update sets.

Current tradeoff: this keeps the dynamic fallback object-safe but means proposal RNG calls go
through `dyn RngCore`. If profiling shows that cost matters on the typed hot path, the next design
step is a generic `Update<R>` path for static kernels while retaining the object-safe dynamic path.

### Parallel verification caused Cargo build-lock contention

Running Cargo commands concurrently made later commands wait on Cargo's build-directory lock. This
was seen with `cargo test`/`cargo clippy`, and again with `cargo clippy`/`cargo run --example
ising_2d`.

Impact: no code issue; it only made verification output noisier. Future verification can run these
commands sequentially when clean output is more important than wall-clock time.

### Error rename caused formatting drift

Renaming `SimplemcError` to `RmcError` shortened a couple of expressions enough that
`cargo fmt --check` failed.

Resolution: ran `cargo fmt`, then reran `cargo fmt --check`, `cargo test`, and
`cargo clippy --all-targets -- -D warnings` successfully.

### Realistic state sharing was initially awkward

The Ising example needed both the update and the measurement to access the same lattice state.
Because the initial engine API did not model simulation state as a first-class value passed to
updates and measurements, the first version of the example used `Arc<Mutex<IsingLattice>>`.

Impact: that was acceptable for proving the public API, but not ideal for the hot path. It added
locking overhead even though each chain is single-threaded internally.

Resolution: added the first stateful run/kernel/update/measurement path and ported the Ising
example to it. The shared-owner pattern remains available through the older non-stateful traits, but
physical hot-path examples can now avoid per-step locking.

Remaining limitation: the stateful static path currently has only `SingleStatefulUpdateSet<U>`.
Weighted multi-update stateful composition still needs the same kind of ergonomic work as the
non-stateful static path.

### Stateful tests exposed missing signed merge support

The first stateful parallel test used a signed counter output. `Merge` had unsigned integer and
`f64` impls, but not `i64`.

Resolution: added `Merge for i64`.

### Trait method use needed explicit trait import in tests

The stateful test called `stats()` on `SingleStatefulUpdateSet`, but `stats` is provided by the
`UpdateSet` trait.

Resolution: imported `UpdateSet` in the test module.

## Deferred / next

- Split `rust/` into the planned workspace shape:
  - `crates/rmc-core`
  - later `rmc-stats`, `rmc-grids`, `rmc-numeric`, `rmc-io`, `rmc-mpi`, façade `rmc`
- Add serde bounds where needed by checkpointing.
- Add property tests for `SeedSource` reproducibility across thread counts.
- Implement multi-update monomorphized update sets for both non-stateful and stateful hot loops.
- Keep `DynUpdateSet` as explicit flexible fallback, not the default kernel path.
- Factor shared loop mechanics between dynamic and typed run APIs if duplication grows.
- Add richer `Merge` impls for accumulator/result types as they are introduced.
- Decide whether the stateful API should become the default engine API before adding `rmc-stats`.
