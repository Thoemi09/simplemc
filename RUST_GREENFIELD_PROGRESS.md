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
- `rust/src/mc/kernel.rs`: dynamic boxed and static state-generic Metropolis kernels.
- `rust/examples/random_walk.rs`: first executable prototype example.
- `rust/examples/ising_2d.rs`: first physical Monte Carlo example.
- `rust/benches/hot_path.rs`: first Criterion hot-path benchmark.
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
- Updated `Measurement<State>` to use typed ownership:
  - associated `Output`
  - `finish(self) -> Self::Output`
- Updated `Update<State>` so `attempt` receives the chain state and active chain RNG.
- Collapsed the separate stateless/stateful trait families:
  - stateless updates are `Update<()>`
  - stateless measurements are `Measurement<()>`
  - stateful simulations use the same traits with a real `State`
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

- Added `SteppingUpdateSet<State, R>` as the trait used by kernels that can select and execute
  updates.
- Added `SingleUpdateSet<U>` for one concrete update type with static dispatch.
- Added `TwoUpdateSet<A, B>` for two concrete update types with static dispatch and weighted
  selection.
- Added `WeightedUpdateSet<U>` for any number of weighted updates of one concrete type. Heterogeneous
  simulations can use an enum as `U` to avoid boxed dynamic dispatch.
- Added `MetropolisKernel<S>` generic over any `SteppingUpdateSet<State, R>`.
- Kept `DynUpdateSet` compatible by implementing `SteppingUpdateSet<(), R>` for it.
- Added tests covering:
  - typed run with `MetropolisKernel<SingleUpdateSet<_>>`
  - typed run with `MetropolisKernel<TwoUpdateSet<_, _>>`
  - typed run with `MetropolisKernel<WeightedUpdateSet<_>>`
  - typed run with owned non-unit state returned to the caller
  - parallel run with independent chain states and merged outputs
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
- `run_parallel`:
  - validates `chains > 0`
  - derives one RNG per chain from `SeedSource`
  - builds independent `(state, kernel, measurement)` triples through a user closure
  - runs each chain with `run_typed`
  - merges `SimulationStats` and `Measurement::Output` through `Merge`
- Added `Merge` impls for common scalar outputs used by prototype tests.
- Added tests for:
  - merging independent chain outputs
  - rejecting zero chains
  - reproducibility across one-thread and four-thread rayon pools for a stochastic update

### RNG-backed update proposals

- Changed `Update::attempt` to receive the chain state and active chain RNG:
  `attempt<R: rand::Rng + ?Sized>(&mut self, state: &mut State, rng: &mut R) -> f64`.
- Updated dynamic wrappers, `UpdateEntry`, `DynUpdateSet`, `SingleUpdateSet`, and `TwoUpdateSet` to
  pass the same framework-managed RNG into update attempts.
- Added a test proving updates can draw proposal randomness from the chain RNG while remaining
  reproducible for the same seed.

### First example

- Added `rust/examples/random_walk.rs`.
- The example demonstrates:
  - static update dispatch via `MetropolisKernel<SingleUpdateSet<_>>`
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
  measurement values in its original version. The current version uses the state-generic API
  instead: each chain owns its `IsingLattice`, updates receive `&mut IsingLattice`, and measurements
  receive `&IsingLattice`.

### Unified state-generic engine slice

- Collapsed the separate stateless and stateful APIs into one state-generic path.
- `run_typed` returns `(State, SimulationStats, M::Output)` so callers can inspect or later
  checkpoint the final chain state. Stateless callers pass `()`.
- `run_parallel` now builds `(state, kernel, measurement)` triples for every chain.
- Ported `rust/examples/ising_2d.rs` away from `Arc<Mutex<IsingLattice>>`.
- Ported `rust/examples/random_walk.rs` away from `Arc<AtomicI64>`.
- Added `Merge for i64`, useful for signed observables and state-owned prototype tests.

### Stronger validation slice

- Added a seed-diffusion test over 512 chain seeds:
  - checks uniqueness
  - checks adjacent-chain seed hamming-distance distribution
- Added `rust/tests/ising_validation.rs`, a deterministic 2x2 Ising validation at infinite
  temperature. It compares sampled signed and absolute magnetization against exact enumeration.
- Added deterministic transform-sampler midpoint moment checks.
- Added a fixed-seed `uniform_index` bucket-balance check.
- Added `uniform_index` helper using `rand::Rng::gen_range`, so examples no longer demonstrate
  modulo-biased index selection.
- Documented that scalar `Merge` impls are for sum-like quantities, not already-normalized means.
- Documented that accepted-certain Metropolis moves do not consume an acceptance RNG draw.

### First benchmark harness

- Added Criterion as a dev dependency.
- Added `rust/benches/hot_path.rs`.
- The benchmark compares:
  - `static_single_update_10k_steps`
  - `dyn_single_update_10k_steps`
- A short local run with `cargo bench --bench hot_path -- --sample-size 10` produced:
  - static path: roughly `21.7 us` per 10k-step run
  - dynamic boxed path: roughly `210 us` per 10k-step run
- Criterion reported that `gnuplot` was unavailable and used the plotters backend instead.

### Rustdoc pass

- Added crate-level docs describing the current core scope.
- Added docs for random seeding and `SeedSource`.
- Added docs for simulation params, stats, context, run entry points, parallel config, kernels, and
  update sets.

### Self-energy DMC source assessment

- Inspected `/home/samox/studies/phd/repos/self-energy-dmc/src` read-only to judge whether the
  current greenfield engine can host a DiagMC Fröhlich-polaron prototype.
- Found that the core simulation maps well to the current Rust API:
  - the C++ `diagram` becomes the owned chain `State`
  - each update becomes an `Update<Diagram>`
  - the update schedule can use `WeightedUpdateSet<PolaronUpdate>`, where `PolaronUpdate` is an enum
    wrapping `ChangeTau`, `ChangeInternalTau`, `AddPhonon`, `RemovePhonon`,
    `RescaleDiagram`, `ChangeInternalQModulus`, `ChangeInternalQDirection`, and `ChangeTopology`
  - the histogram/self-energy measurement can use the typed `Measurement<Diagram>` path
  - independent-chain parallel execution can use `run_parallel`
- Current conclusion: a prototype port is feasible now; a full feature-parity replacement still
  needs additional support crates/modules for statistics, grids/interpolation, checkpoint/config
  I/O, FFT, and possibly MPI.
- Important design point for the first prototype: the C++ state uses `std::list<vertex>` plus stored
  iterators. The Rust port should not copy that layout directly; it needs a stable-ID/slab/vector
  representation for vertices and topology mutations.
- The inspected `self-energy-dmc` checkout already had an unrelated dirty file
  (`src/diagram.hpp`). No files in that repository were modified.

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

After collapsing the stateless/stateful trait families and adding validation tests, the verification
command set passed with 43 Rust tests green.

After adding the benchmark harness, additional validation tests, and rustdoc pass, the verification
command set passed with 45 Rust tests green.

After adding `WeightedUpdateSet<U>`, the verification command set passed with 49 Rust tests green.
The same state was checked with:

```text
cd rust
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
```

Benchmark smoke command:

```text
cd rust
cargo bench --bench hot_path -- --sample-size 10
```

Short-run result:

```text
static_single_update_10k_steps: ~21.7 us
dyn_single_update_10k_steps:    ~210 us
```

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

### Static dispatch was initially limited to one or two update types

`SingleUpdateSet<U>` and `TwoUpdateSet<A, B>` prove the static hot-path shape, but real simulations
may need more update types with ergonomic weighted selection.

Resolution: added `WeightedUpdateSet<U>`. It stores many weighted updates of one concrete type, and
users can make that type an enum to combine heterogeneous update implementations without
`Box<dyn Update>`. `SingleUpdateSet` and `TwoUpdateSet` remain useful for the smallest cases and for
true two-type inverse-pair baselines.

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

Resolution: changed `Update::attempt` to the state-generic, RNG-generic form
`attempt<R: rand::Rng + ?Sized>(&mut self, state: &mut State, rng: &mut R) -> f64` and threaded that
RNG through dynamic and static update sets.

Current tradeoff: this makes `Update<State>` object-unsafe, which is intentional for the static
hot path. The dynamic fallback therefore uses an object-safe bridge specialized to `Update<()>`.

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

Resolution: collapsed the first stateful experiment into the default state-generic API and ported
the Ising and random-walk examples to owned state. The shared-owner pattern remains possible in user
code, but physical hot-path examples no longer require per-step locking.

Remaining limitation: `WeightedUpdateSet<U>` uses a `Vec`, so enum-backed heterogeneous updates pay
one enum match after weighted selection. A macro-generated concrete-field update set can still be
added later if benchmarks show that this matters.

### State-owned tests exposed missing signed merge support

The first state-owned parallel test used a signed counter output. `Merge` had unsigned integer and
`f64` impls, but not `i64`.

Resolution: added `Merge for i64`.

### Trait method use needed explicit trait import in tests

The state-owned test called `stats()` on `SingleUpdateSet`, but `stats` is provided by the
`UpdateSet` trait.

Resolution: imported `UpdateSet` in the test module.

### Infinite-temperature Ising validation was initially periodic

The first 2x2 Ising validation attempted to use an always-accepted beta=0 single-spin flip and
measure every four flips. That chain is periodic on the four-spin hypercube, so it sampled only one
parity sector and produced the wrong exact absolute magnetization.

Resolution: made the validation chain lazy by accepting proposed flips with probability `0.5`.
The stationary distribution remains uniform, but self-loops break the parity artifact.

### Criterion added a larger dev-dependency tree

Adding the benchmark harness pulled in Criterion, plotters, clap, serde, and related transitive
dependencies.

Impact: this only affects dev/bench builds, but `Cargo.lock` changed and initial benchmark/test
builds fetch and compile more crates.

### Self-energy DMC port will need a deliberate state representation

The C++ self-energy code stores diagram vertices in a linked list and keeps iterators in auxiliary
storage. That works naturally in C++ but does not translate cleanly to safe Rust, especially when
updates add/remove vertices and measurements inspect topology.

Resolution for the prototype: start with `Vertex` and `Diagram` as owned Rust state and choose a
stable-index representation before porting update semantics. This is the main design risk for the
first DiagMC pass.

## Deferred / next

- Split `rust/` into the planned workspace shape:
  - `crates/rmc-core`
  - later `rmc-stats`, `rmc-grids`, `rmc-numeric`, `rmc-io`, `rmc-mpi`, façade `rmc`
- Add serde bounds where needed by checkpointing.
- Add property tests for `SeedSource` reproducibility across thread counts.
- Keep `DynUpdateSet` as explicit flexible fallback, not the default kernel path.
- Add richer `Merge` impls for accumulator/result types as they are introduced.
- Continue expanding analytical/property validation, including more sampler moment checks and a
  stronger finite-size Ising benchmark.
- Decide the dyn-measurement result-sink design.
- Benchmark `WeightedUpdateSet<Enum>` against a hand-written concrete-field update set if update
  dispatch overhead shows up in real simulations.
- Start the self-energy DMC prototype in Rust:
  - port `vertex` and the core `diagram` state first
  - add diagram sanity/invariant tests before update ports
  - implement `ChangeTau` as the first minimal update
  - add `PolaronUpdate` plus `WeightedUpdateSet<PolaronUpdate>`
  - port `AddPhonon` / `RemovePhonon` as the first nontrivial inverse pair
  - add a minimal typed histogram measurement after the update path is validated
