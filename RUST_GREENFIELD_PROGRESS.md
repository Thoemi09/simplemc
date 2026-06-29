# Greenfield Rust progress

Tracks implementation against `RUST_GREENFIELD_PLAN.md`. See `RUST_GREENFIELD_REVIEW_JUN19.md` for
the last external review (its High/Medium findings are resolved; see below).

## Steering decisions (2026-06-23)

- **Scope:** build a **general framework first** (plan's phase order stands). `self-energy-dmc` is
  treated as one future *user*, not the roadmap driver — see the appendix assessment.
- **Testing:** **hybrid** — hand-written analytical/deterministic tests are primary; `proptest` is
  used selectively where randomized input adds coverage (Merge laws, parallel==sequential, seed
  reproducibility/separation).
- **serde:** **opt-in `serde` feature, not a `Scalar`/state trait bound.** Implemented for core
  configuration/metadata, reproducible RNG state, static update containers, and scalar accumulators.
  Dynamic boxed wrappers remain intentionally non-serializable.
- **Workspace:** `rust/` is now a Cargo workspace with `rmc-core`, `rmc-stats`, `rmc-grids`,
  `rmc-numeric`, `rmc-io`, and the `rmc` façade. New subsystems should continue as sibling crates.

## Current state

Roughly **Phases 0–4 mostly complete** (parallel core shipped, reproducible across thread counts,
and scalar/vector stats implemented), **Phase 5 (grids)** substantially implemented, plus the first
**Phase 6 (numeric)** interpolation MVP and **Phase 7 (io)** JSON checkpoint/restart MVP. `rust/` is
now a workspace. CI exists for the Rust workspace. Opt-in serde support is implemented. Selective
`proptest` coverage is in place.

Layout:
- `crates/rmc-core/` — engine crate. `error.rs` (`RmcError` / `Result`), `merge.rs` (`Merge`),
  `scalar.rs` (`Scalar`, `SampleType`), `random/` (sample/PDF helpers, `uniform_index`,
  `SeedSource`/`ChainId`, `DefaultRng = Xoshiro256PlusPlus`, public `Rng` re-export), and `mc/`
  state-generic `Update`/`Measurement`/`Kernel`/`RunCallbacks` traits; static update sets
  (`Single`/`Two`/`WeightedUpdateSet`) and dyn fallback (`DynUpdateSet`/wrappers); `MetropolisKernel`;
  sequential `run`/`run_typed` and rayon `run_parallel`/`run_parallel_in_pool`.
- `crates/rmc-stats/` — `Accumulator`/`Mean`/`Variance` traits; `ScalarMoments` (Welford +
  Chan-Golub-LeVeque merge), `WeightedScalarMoments`, `ScalarCovariance`, and fixed-lag
  `ScalarAutocorrelation`; `ScalarBatchMeans` and `ScalarJackknife` resampling helpers;
  `VectorMoments` and `VectorCovariance` over `nalgebra::DVector`/`DMatrix`; all `Merge`, with
  opt-in serde derives.
- `crates/rmc-grids/` — grid slice: `Grid1d`, `LinearGrid`, `PowerGrid`, `SymmetricPowerGrid`,
  `CustomGrid`, `AxisGrid`, `NdGrid<G, const N: usize>` including mixed-axis
  `NdGrid<AxisGrid, N>`, safe point/bin queries, point/center/width/volume iterators, row-major
  flat/N-D indexing helpers, integer/grid subrange helpers, opt-in serde derives/manual serde where
  const-generic arrays need it.
- `crates/rmc-numeric/` — first numeric slice: checked owned 1-D `LinearInterpolation<G>` and
  const-generic N-D `LinearInterpolationNd<G, N>` over `rmc-grids`, including mixed-axis
  `LinearInterpolationMixed<N> = LinearInterpolationNd<AxisGrid, N>`, with opt-in serde derives.
- `crates/rmc-io/` — checkpoint/restart MVP: versioned `Checkpoint<T>` envelope, JSON string/bytes
  encode/decode, JSON file save/load, atomic JSON save, payload convenience helpers, and version
  rejection errors.
- `crates/rmc/` — façade crate. Re-exports `rmc-core`; re-exports `rmc-stats` as `rmc::stats`
  behind the default `stats` feature; re-exports `rmc-grids` as `rmc::grids` behind the default
  `grids` feature; re-exports `rmc-numeric` as `rmc::numeric` behind the opt-in `numeric` feature;
  re-exports `rmc-io` as `rmc::io` behind the opt-in `io` feature. Placeholder feature flags remain
  for later `mpi` and `hdf5`.
- `crates/rmc/examples/` — `random_walk.rs`, `ising_2d.rs` (owned per-chain state, no `Arc<Mutex>`).
- `crates/rmc-core/benches/hot_path.rs` — static vs dyn dispatch.
- `crates/*/tests/` — native analytical/deterministic tests by crate ownership.
- `crates/rmc-core/tests/properties.rs` and `crates/rmc-stats/tests/merge_properties.rs` —
  selective property tests for merge laws, seed reproducibility/separation, and parallel reduction.
- `.github/workflows/rust.yml` — GitHub Actions CI for fmt, feature-aware clippy, locked tests,
  all-features tests, lean façade feature check, example builds, and an MSRV test job on Rust 1.75.

## What works

- **Core traits:** `Merge`; state-generic `Update<State>`/`Measurement<State>` (typed output via
  `finish(self)`, no `Any`/downcast); `Kernel<State, R>`. Stateless = `State = ()`.
- **Hot path is monomorphized:** `Update::attempt<R: Rng>` is generic, so RNG draws inline on the
  static path (review finding #2). Static sets dispatch without a vtable; `DynUpdateSet` bridges
  through an object-safe `UpdateObject` for the runtime-flexible path (no per-step allocation after
  review finding #1 fix). Bench: ~21.7 µs static vs ~210 µs dyn per 10k steps.
- **Run loop:** one shared `drive` fn; `run_typed` returns `(State, SimulationStats, M::Output)`;
  callbacks can stop the run; `SimulationParams::validate` rejects `steps_per_cycle = 0` and documents
  partial final cycles.
- **Parallel:** `run_parallel` derives one RNG per chain from `SeedSource`, runs independent chains,
  reduces via `Merge` in deterministic chain order (reproducible across thread counts — verified on
  1- and 4-thread pools). MPI backend (Phase 8) must preserve this ordering.
- **Stats:** scalar moments / weighted moments / covariance / fixed-lag autocorrelation / scalar
  batch means / scalar jackknife / vector moments / vector covariance live in `rmc-stats`, all
  mergeable and usable as typed `run_parallel` output.
- **Grids:** 1D linear, power, symmetric-power, custom, tagged `AxisGrid`, and const-generic N-D
  grids live in `rmc-grids`; mixed-axis product grids use `NdGrid<AxisGrid, N>`. Constructors
  validate inputs, point/bin APIs are bounds checked, endpoint binning is defined, iterators expose
  grid points/bin centers/bin widths/volumes, row-major flat/N-D index helpers are available, and
  integer/grid subrange helpers mirror the C++ centering semantics.
- **Numeric:** checked 1-D and N-D linear interpolation lives in `rmc-numeric`; constructors reject
  value/grid size mismatches, evaluation reports out-of-domain points, and affine analytical tests
  cover nonuniform and mixed-axis grids.
- **IO:** versioned JSON checkpoints live in `rmc-io`; callers choose the serializable payload, so
  state, RNG, kernels/update sets, measurements, and metadata can be checkpointed without adding IO
  bounds to core traits. Atomic JSON save is available for checkpoint files.
- **Serde:** `serde` features exist on `rmc-core`, `rmc-stats`, `rmc-grids`, `rmc-numeric`, and the
  `rmc` façade. Round-trip tests cover `SeedSource`, `DefaultRng`, run/parallel config, update
  stats/outcomes, static update sets/kernels, scalar/vector accumulators, 1D grids, tagged axes,
  N-D grids, and interpolation objects. `rmc-io` checkpoint tests cover JSON envelope round-trips
  and version rejection.

## Validation status

- Seed diffusion over 512 chain seeds (uniqueness + adjacent hamming-distance range).
- Deterministic 2×2 Ising at infinite temperature vs exact enumeration (signed + |m|); lazy chain to
  avoid the parity artifact.
- Finite-size 8×8 2D Ising validation around the exact square-lattice critical point
  `β_c = 0.5 ln(1 + sqrt(2))`: post-burn-in Binder cumulant lands in the expected critical window
  and orders between high-temperature and low-temperature behavior.
- Transform-sampler midpoint moment checks; fixed-seed `uniform_index` bucket balance.
- Stats: closed-form scalar and vector mean/variance/covariance/correlation/autocorrelation,
  batch-means standard error, jackknife delete-one estimates/error, merge-equals-single-pass where
  mathematically appropriate, independent-chain autocorrelation/batch merge semantics, edge cases.
- Grids: closed-form 1D linear/power/symmetric-power/custom/tagged-axis points, homogeneous and
  mixed-axis N-D shapes/points/bin centers/bin volumes, row-major flat/N-D index round-trips,
  increasing and decreasing ranges, midpoint/end-point bin indexing, exact-size iterators, invalid
  input errors, integer/grid subrange boundary cases, and serde round-trips.
- Numeric: 1-D linear interpolation reproduces affine functions on nonuniform grids; bilinear and
  trilinear interpolation reproduce affine planes/hyperplanes on uniform and mixed-axis grids; size
  mismatch and out-of-domain errors are covered; serde round-trips preserve interpolation objects.
- IO: JSON checkpoints round-trip through strings and files; unsupported checkpoint versions are
  rejected; a split random-walk run restored from checkpoint reaches the same final state, RNG
  stream position, measurement output, and update statistics as an uninterrupted run.
- Property tests: exact scalar `Merge` laws; approximate merge laws and single-pass equivalence for
  scalar moments, weighted moments, and covariance; seeded parallel run equals a manual sequential
  chain reduction; `SeedSource` is reproducible and separates adjacent randomized chain IDs; derived
  RNG stream prefixes are distinct and have balanced bits / adjacent Hamming distance across many
  chains.
- **Gaps (next):** broader finite-size scaling / longer physics validation beyond the lightweight
  Binder smoke test.

## Verification

```text
cd rust
cargo fmt --all --check
cargo test --locked # latest: 124 tests green across the workspace
cargo test --locked --all-features # latest: 147 tests green, including serde/checkpoint round-trips
cargo test --locked -p rmc --no-default-features
cargo test --locked -p rmc --features numeric
cargo test --locked -p rmc --features io
cargo clippy --all-targets --all-features -- -D warnings
cargo build --locked -p rmc --examples
cargo bench --bench hot_path -- --sample-size 10   # smoke
```

Examples run manually (`cargo run --example random_walk` / `ising_2d`); Ising at β≈0.44 gives
physically plausible |m| ≈ 0.7 near criticality.

CI note: stable-side checks were verified locally with Rust 1.96.0. The Rust 1.75 MSRV job is defined
in GitHub Actions but has not been run locally.

## Next (prioritized)

1. **Continue numeric** (Phase 6): polynomial/spline interpolation or quadrature, depending on which
   MVP user workflow is next.
2. **Decide checkpoint binary format** (Phase 7 polish): add `bincode`/`postcard` only after choosing
   format compatibility expectations.
3. **Grid polish if needed:** tighten docs/examples and adjust APIs only if numeric work exposes
   friction.
4. **Finish stats polish** (Phase 4): decide whether explicit block abstractions should sit above
   scalar batch means, and add any missing property tests/docs before moving on.
5. **Broaden validation:** finite-size scaling / longer Ising physics checks beyond the lightweight
   Binder smoke test.
6. Then per plan: mpi (8), hdf5 (9).
7. Decide the dyn-measurement result-sink design (review finding #4, deliberately deferred).

## Notable lessons (non-obvious only)

- **Generic `attempt<R: Rng>` makes `Update` object-unsafe** — intentional, to inline RNG on the
  static hot path. The dyn path bridges via an object-safe `UpdateObject` (valid through rand's
  blanket `impl<R: RngCore + ?Sized> Rng for R`).
- **`Merge for f64` = addition** is for sum-like quantities, not already-normalized means — documented
  footgun. Examples divide at the end.
- **Accepted-certain Metropolis moves don't consume an acceptance draw** (`p >= 1.0` short-circuit),
  so RNG stream position depends on the acceptance branch — documented for stream reasoning.
- **2×2 Ising at β=0 with deterministic flips is periodic** on the hypercube and samples one parity
  sector; the validation chain accepts flips w.p. 0.5 to break the artifact.
- Run Cargo commands **sequentially** during verification — concurrent invocations contend on the
  build-directory lock.

## Appendix — `self-energy-dmc` future-user assessment

A read-only inspection of `self-energy-dmc/src` (no files modified there; it already had an unrelated
dirty `src/diagram.hpp`). **Not the roadmap driver** — captured here to validate the framework's
generality and inform later subsystem design.

- The DiagMC Fröhlich-polaron simulation maps onto the current API: C++ `diagram` → owned chain
  `State`; each update → `Update<Diagram>`; schedule → `WeightedUpdateSet<PolaronUpdate>` with an enum
  over `ChangeTau`, `ChangeInternalTau`, `AddPhonon`, `RemovePhonon`, `RescaleDiagram`,
  `ChangeInternalQModulus`, `ChangeInternalQDirection`, `ChangeTopology`; self-energy histogram →
  typed `Measurement<Diagram>`; parallelism → `run_parallel`.
- A prototype is feasible now; a full feature-parity replacement still needs stats, grids/interp,
  checkpoint/config IO, FFT, possibly MPI.
- **Design risk:** the C++ state uses `std::list<vertex>` + stored iterators. Do **not** copy that
  layout — choose a stable-ID/slab/vector representation for vertices and topology before porting
  update semantics. Suggested first steps if/when this is taken up: port `Vertex`/`Diagram` with
  invariant tests → `ChangeTau` → the `AddPhonon`/`RemovePhonon` inverse pair → a typed histogram.
