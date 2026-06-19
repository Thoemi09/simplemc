# simplemc → Rust port plan

A staged plan to port `simplemc` (C++23 Monte Carlo library) to Rust. The plan favors
**idiomatic Rust over mechanical translation**: as long as observable behavior and numerical
results are preserved, modules, file layout, and APIs are free to be reshaped to fit the Rust
ecosystem. Work proceeds **one subsystem end-to-end at a time** (code + tests + examples + parity
checks) before the next is started. **Core simulation functionality ships first; MPI and HDF5 are
deferred to the end.**

---

## 1. Guiding principles

1. **Behavior parity, not source parity.** The acceptance bar is "produces the same numbers / same
   observable behavior as the C++ library", verified by a parity harness (§4). Internal structure is
   redesigned freely.
2. **Embrace Rust idioms.** C++ concepts → traits; type erasure → `Box<dyn Trait>` + `Any`; ADL
   customization points → traits; `std::variant` → enums; exceptions → `Result`; `if constexpr`
   capability detection → optional traits; `fmt`/`ranges`/`nlohmann_json` → std formatting /
   iterators / `serde`.
3. **Ship the core ASAP.** The minimal runnable product is `foundation + random + mc` — enough to
   drive a real Monte Carlo simulation. That is the `0.1.0` milestone (Phase 3). Everything else is
   additive.
4. **One subsystem, fully done.** A subsystem is not "done" until its code compiles warning-clean,
   its ported tests pass, its examples run, and its parity fixtures match. No half-ported modules
   left dangling while moving on.
5. **MPI and HDF5 last.** Both pull in C libraries and are the highest-friction integrations. The
   core is designed so they slot in additively without reworking earlier subsystems.

---

## 2. Source inventory (what we are porting)

| Subsystem | C++ LOC (hdrs) | Role | Rust dep of note |
|---|---|---|---|
| `utils` | ~1.0k | concepts, error, timer, formatting, nd-indexing, ranges | `thiserror`, `num-complex` |
| `random` | ~1.1k | xoshiro256 (+,++,**), splitmix64, seeding, PDFs/samplers | `rand` (`RngCore`/`SeedableRng`) |
| `accs` | ~6.3k | mean/var/covar/autocorr/batch/block/jackknife/multivalue accumulators | `nalgebra`, `num-complex` |
| `grids` | ~2.2k | linear/power/symmetric-power/custom 1-D grids, N-D grids | — |
| `numeric` | ~4.0k | polynomials, interpolation, quadrature, Bravais lattices, linear map | `nalgebra` |
| `mc` | ~3.2k | updates, measurements, named sets, kernels, run loop, params/stats/ctx, callbacks | — |
| `serialize` | ~1.3k | hierarchical store: JSON / HDF5 / runtime-variant backends | `serde`, `serde_json`, (`hdf5`) |
| `mpi` | ~2.9k | communicator RAII, collectives, datatypes | `mpi` (rsmpi) |

Tests: **644 cases across 69 files** (googletest). These become the port oracle.

---

## 3. Target architecture

### 3.1 Crate layout

A **single published crate** `simplemc` with internal modules per subsystem and Cargo **features**
for the optional, C-dependent backends. This keeps consumption trivial (`cargo add simplemc`) while
letting MPI/HDF5 be opt-in.

```
simplemc/
  Cargo.toml          # [features] default=["serde-json"]; serde-json; hdf5; mpi
  src/
    lib.rs            # re-exports, crate docs
    error.rs          # SimplemcError (thiserror), Result alias
    scalar.rs         # Scalar trait (f64, Complex64); SampleType (scalar | nalgebra vector)
    random/           # splitmix64, xoshiro256, seed, samples
    mc/               # update, measurement, named_set, kernel, run, params, stats, ctx, callbacks
    accs/             # mean, var, covar, autocorr, batch, block, jackknife, multivalue
    grids/            # linear, power, symmetric_power, custom, nd, traits
    numeric/          # polynomials, interpolation, quadrature, lattice, linear_map
    serialize/        # store trait, json backend, (hdf5 backend behind feature), variant→enum
    mpi/              # behind "mpi" feature: communicator, collectives, datatype
  tests/              # integration + parity tests
  fixtures/           # committed reference vectors generated from the C++ build
  examples/           # ported tutorials
  benches/            # criterion benchmarks for hot paths (RNG, accumulators, run loop)
```

> If MPI/HDF5 build complexity becomes disruptive to core CI, split them into sibling crates
> (`simplemc-mpi`, `simplemc-hdf5`) re-exported by the façade. Defer that decision to Phase 8.

### 3.2 Dependency choices

| Concern | C++ | Rust | Notes |
|---|---|---|---|
| Formatting | `fmt` | std `format!` / `Display` | drop entirely |
| Ranges | `range-v3` / `std::ranges` | native iterators | drop entirely; nicer in Rust |
| JSON | `nlohmann_json` | `serde` + `serde_json` | derive-based; idiomatic |
| Linear algebra | `Eigen3` | `nalgebra` (`DVector`/`DMatrix`, `SymmetricEigen`) | `num-complex` for complex |
| Complex | `std::complex<double>` | `num_complex::Complex64` | |
| RNG framework | `<random>` engines | `rand` (`RngCore`/`SeedableRng`) | our engines plug into `rand` |
| Errors | exceptions | `Result<_, SimplemcError>` (`thiserror`) | API change, embraced |
| Type erasure | virtual interface + `unique_ptr` + RTTI | `Box<dyn Trait>` + `Any` + `dyn-clone` | |
| Ordered maps | hand-rolled name→index | `indexmap::IndexMap` | preserves insertion order |
| Tests | googletest | `#[test]`, `approx`, `rstest`, `proptest` | typed/parametrized → `rstest` |
| MPI | raw `mpi.h` wrappers | `mpi` (rsmpi) + `Equivalence` derive | Phase 8 |
| HDF5 | HighFive | `hdf5` / `hdf5-metno` crate | Phase 9 |

### 3.3 Concept → trait mappings (the heart of the redesign)

The C++ concept hierarchy is essentially a set of Rust traits already. Direct mappings:

**Scalars / samples** (`utils/concepts.hpp`, `accs/concepts.hpp`):
```rust
pub trait Scalar: Copy + /* arithmetic, conj, ... */ {}   // impl for f64, Complex64
pub trait SampleType { type Value: Scalar; fn size(&self) -> usize; }
// impl for f64, Complex64 (size 1) and nalgebra DVector<f64>/DVector<Complex64>
```

**Accumulators** (`accs/concepts.hpp`) → supertrait chain:
```rust
pub trait Accumulator {
    type Sample: SampleType;
    type Value: Scalar;
    fn count(&self) -> u64;
    fn size(&self) -> usize;
    fn accumulate(&mut self, x: &Self::Sample);   // C++ `acc << x`
}
pub trait MeanAccumulator:       Accumulator { fn mean(&self) -> Self::Sample; }
pub trait VarianceAccumulator:   MeanAccumulator { fn variance(&self) -> /* sample */; }
pub trait CovarianceAccumulator: VarianceAccumulator { fn covariance(&self) -> DMatrix<Self::Value>; }
pub trait BatchAccumulator:      CovarianceAccumulator { fn batches(&self) -> &[impl MeanAccumulator]; }
```
(Optionally also `impl Shl` for the `<<` ergonomics, but the method is the canonical API.)

**MC contracts** (`mc/concepts.hpp`):
```rust
pub trait Measurement { fn measure(&mut self); }
pub trait Update {
    fn attempt(&mut self) -> f64;
    fn accept(&mut self);
    fn reject(&mut self) {}            // default no-op (matches C++ optional reject)
}
pub trait Kernel {
    fn step<R: Rng + ?Sized>(&mut self, rng: &mut R);
    fn prepare(&mut self) {}           // default no-op (matches optional prepare())
}
pub trait RunCallbacks {
    fn on_step(&self, _ctx: &SimulationCtx) {}
    fn on_cycle(&self, _ctx: &SimulationCtx) {}
    fn on_checkpoint(&self, _ctx: &SimulationCtx) {}
    fn stop_when(&self, _ctx: &SimulationCtx) -> bool { false }
}
```

**Grids** (`grids/concepts.hpp`):
```rust
pub trait GridCommon { type Value; type Index; const DIM: usize;
    fn size(&self)->i64; fn first(&self)->Self::Value; fn last(&self)->Self::Value;
    fn at(&self, i: Self::Index)->Self::Value; fn index(&self, x: Self::Value)->Self::Index;
    fn volume(&self, i: Self::Index)->f64; fn center(&self, i: Self::Index)->Self::Value; }
pub trait Grid1d: GridCommon /* Value=f64, Index=i64 */ + Index<usize> { }
pub trait GridNd<const N: usize>: GridCommon { fn shape(&self)->[i64; N]; /* grids() */ }
```

**Serialization** (`serialize/concepts.hpp`) → split into (a) `serde` for leaf values and (b) an
object-safe `Store` trait for the navigable hierarchical document, so the runtime backend choice
(JSON now, HDF5 later) survives:
```rust
pub trait Store {
    fn sub(&self, key: &str) -> Self where Self: Sized;     // C++ `s[key]`
    fn has(&self, key: &str) -> bool;
    fn save_at<T: Serialize>(&mut self, key: &str, v: &T) -> Result<()>;
    fn load_at<T: DeserializeOwned>(&self, key: &str, v: &mut T) -> Result<()>;
    fn try_load_at<T: DeserializeOwned>(&self, key: &str, v: &mut T) -> Result<bool>;
}
```
The C++ `variant_serializer` (compile-time backend pack visited at runtime) becomes a plain enum:
```rust
pub enum Backend { Json(JsonStore), #[cfg(feature="hdf5")] Hdf5(Hdf5Store) }  // match, no std::visit
```

**MPI collect** (`mc/concepts.hpp`) → optional trait, wired in Phase 8:
```rust
#[cfg(feature="mpi")]
pub trait MpiCollect { fn mpi_collect(&self, comm: &impl Communicator) -> Self; }
```

### 3.4 The one genuinely non-mechanical spot: type erasure + optional capabilities

`basic_measurement` / `basic_update` erase a user type and, via `if constexpr`, *conditionally*
forward serialization and MPI-collect only if the user type supports them. Rust has no stable
specialization, so this is the single design decision that needs care.

**Recommended approach** (stable, explicit, no nightly):

- Core erased object carries only the always-present capability + identity:
  ```rust
  trait MeasurementObject: DynClone {            // dyn-clone for the C++ clone()
      fn measure(&mut self);
      fn as_any(&self) -> &dyn Any;              // C++ get<T>() via RTTI → downcast_ref
      fn as_any_mut(&mut self) -> &mut dyn Any;
  }
  ```
  `Measurement::get::<T>()` → `obj.as_any().downcast_ref::<T>()`.
- Optional capabilities (serialize, mpi-collect) are captured **at construction time** through the
  generic constructor's trait bounds, not probed at runtime:
  ```rust
  impl Measurement {
      pub fn new<M: Measurement + Clone + 'static>(m: M, name: impl Into<String>) -> Self { /* core only */ }
      pub fn new_serializable<M: Measurement + Clone + Serialize + DeserializeOwned + 'static>(...) -> Self { /* + store fns */ }
  }
  ```
  i.e. capabilities are explicit at the registration call. This trades a little ergonomics for
  zero nightly features and full clarity. (Alternative: autoref-specialization trick to auto-detect
  capability in a single `new`; documented as a fallback, not the default.)

This keeps Phase 3 (mc core) shippable with no serialization, and Phases 7–8 add the
`new_serializable` / MPI paths additively.

### 3.5 Ownership redesigns

- `metropolis_kernel` holds a raw `update_set*` with a "must outlive" contract. In Rust, **the
  kernel owns the `UpdateSet`** (moved in at construction); `run` borrows the kernel; update
  statistics are read back through kernel accessors afterward. No raw pointers, no lifetimes leaking
  into the public API.
- `measurement_set` / `update_set` derive from a shared `named_set`. → generic `NamedSet<T>` backed
  by `IndexMap<String, T>` (insertion order preserved for reports and serialization). `active_`
  cache stays as a `Vec<usize>` (or recomputed lazily).
- Validation (`validate_simulation_params`, non-empty names) → returns `Result`, not panic. `run`
  returns `Result<(), SimplemcError>`.

---

## 4. Parity & verification strategy (set up before any porting)

This is the backbone that makes "behavior parity, not source parity" enforceable.

1. **Reference dumper in C++.** A small harness in the existing C++ tree emits deterministic
   fixtures as JSON: raw RNG streams (per engine, per seed), sampler outputs, accumulator results
   for fixed input sequences, polynomial/interpolation/quadrature values, grid mappings, and a full
   small end-to-end MC run trace (seed → steps → final stats).
2. **Commit fixtures** into `simplemc/fixtures/`. Rust CI then needs **no C++ toolchain**.
3. **Rust assertions:**
   - RNG / integer math / closed-form: **bit-exact** (`assert_eq!`).
   - Floating-point reductions (means, variances, covariances, quadrature): `approx`
     `assert_relative_eq!` with a documented tolerance (start `1e-12` relative; loosen only with
     justification tied to summation order).
4. **Test translation.** Port the 644 gtest cases to `#[test]`; typed/parametrized tests
   (`TYPED_TEST`, `TEST_P`) → `rstest` cases; invariants → `proptest` where natural.
5. **Determinism guard.** A dedicated test runs a fixed-seed end-to-end simulation and compares the
   final `SimulationStats` + a measurement digest against the C++ fixture.

Exit criterion shared by every phase: *the subsystem's fixtures match within tolerance and all its
ported tests pass.*

---

## 5. Phased roadmap

Strict ordering. Each phase is a subsystem taken **end-to-end**. `★` marks the shippable core.

### Phase 0 — Scaffolding & parity harness
- Cargo crate, feature flags, CI (fmt, clippy `-D warnings`, test matrix), `rustfmt`/`clippy` config.
- Pick and pin deps (§3.2). Spike `nalgebra` symmetric eigensolver and `rand` integration.
- Build the C++ reference dumper; commit the first fixtures (RNG streams).
- **Exit:** empty crate builds in CI; fixture pipeline works; design docs for §3.3–3.5 agreed.
- **Est:** 2–4 days.

### Phase 1 — `foundation` (utils + scalar/linalg core)
- `SimplemcError` (thiserror) + `Result`. `Scalar` trait (f64, Complex64). `SampleType`
  (scalar + `nalgebra` vector). Timer, nd-indexing, to-string/Display helpers. Drop `fmt`,
  `ranges` (replace with iterators).
- **Exit:** foundation tests pass; downstream traits compile against it.
- **Est:** 2–4 days.

### Phase 2 — `random`
- `splitmix64`, `xoshiro256<{plus, plusplus, starstar}>` (Rust: three types or one enum-param
  generic), seeding, `samples.rs` PDFs/inverse-samplers. Implement `RngCore` + `SeedableRng` so the
  whole `rand` distribution ecosystem works on our engines.
- **Parity:** bit-exact raw streams vs C++ fixtures for every engine/seed. Port the exact
  `[0,1)` transform used by the Metropolis acceptance (do **not** rely on `rand`'s `Uniform` for the
  parity-critical path — `std::uniform_real_distribution` is not bit-identical).
- **Exit:** bit-exact RNG parity; samplers within tolerance; tests + example run.
- **Est:** 2–4 days.

### Phase 3 ★ — `mc` core (CORE SHIP MILESTONE → `0.1.0`)
- Traits: `Measurement`, `Update`, `Kernel`, `RunCallbacks` (§3.3). Type-erased `Measurement` /
  `Update` wrappers (§3.4, core capability only). `NamedSet<T>` + `MeasurementSet` / `UpdateSet`.
  `MetropolisKernel` (owns `UpdateSet`, §3.5). `SimulationParams` / `SimulationStats` /
  `SimulationCtx`. `run()` loop → `Result`. Default callbacks.
- Serialization / MPI hooks are **stubbed via the additive constructors** — not wired yet.
- **Parity:** fixed-seed end-to-end run (a hand-written Ising-like update + counter measurement)
  reproduces the C++ trace's final stats and acceptance counts exactly.
- **Exit:** can drive a real simulation from Rust; tests + an end-to-end example run; **publish
  `0.1.0`** (or tag internal milestone). This is the "core shipped" point.
- **Est:** ~1 week.

### Phase 4 — `accs`
- Accumulator trait chain (§3.3): mean, variance, covariance, autocorrelation, batch, block,
  jackknife, multivalue. Scalar and `nalgebra`-vector samples. Use numerically stable online
  algorithms matching the C++ (Welford / batched), verified against fixtures.
- **Parity:** accumulator outputs for fixed input sequences within `1e-12`; jackknife/autocorr
  against C++ fixtures.
- **Exit:** all accumulator tests pass; example composing an accumulator into an MC measurement runs.
- **Est:** 1–1.5 weeks (largest numeric surface).

### Phase 5 — `grids`
- `GridCommon`, `Grid1d`, `GridNd<N>` traits. `linear_grid`, `power_grid`,
  `symmetric_power_grid`, `custom_grid`, `nd_grid`. Iterator impls replace the C++ grid iterators.
- **Parity:** `at`/`index`/`volume`/`center` mappings exact vs fixtures across grid types.
- **Exit:** grid tests + example run.
- **Est:** ~1 week.

### Phase 6 — `numeric`
- `nalgebra`-backed: orthogonal polynomials (Chebyshev/Hermite/Laguerre/Legendre), interpolation
  (linear/cubic-spline/polynomial), quadrature, Bravais lattices (1-D/2-D/3-D), linear map. Replaces
  Eigen usage; verify eigensolver-dependent paths against `SymmetricEigen`.
- **Parity:** polynomial/interp/quadrature/lattice values within tolerance; eigen-based results
  cross-checked.
- **Exit:** numeric tests + example run.
- **Est:** 1.5–2 weeks (nalgebra↔Eigen API gaps, eigensolver verification).

### Phase 7 — `serialize` (JSON) → adds checkpoint/restart
- `Store` trait + `JsonStore` (serde_json). `Backend` enum (variant replacement). Wire the
  `new_serializable` paths in `mc` (§3.4): checkpoint save/load + input-config save/load. Make all
  library types `Serialize`/`Deserialize` where the C++ had serialization hooks.
- **Parity:** round-trip a checkpoint; restart-from-checkpoint reproduces a continued run;
  byte/structure compatibility with C++ JSON where feasible (document any intentional schema change).
- **Exit:** checkpoint + input-config tests pass; restart example runs.
- **Est:** 3–5 days.

### Phase 8 — `mpi` (feature `mpi`, deferred)
- rsmpi (`mpi` crate). Communicator handling (rsmpi owns RAII), collectives map to rsmpi built-ins
  (broadcast/reduce/gather/scatter/all_reduce/all_gather*/scatterv). Custom datatypes →
  `#[derive(Equivalence)]`. `MpiCollect` trait (§3.3) + wire the MPI paths in `mc` measurement/update
  sets. Multi-process test harness replacing `gtest-mpi-listener` (run under `mpirun` in CI).
- **Parity:** reductions across ranks match C++; the MPI hello-world + a reduced-measurement example
  match expected aggregates.
- **Exit:** MPI tests pass under `mpirun -n N`; examples run.
- **Est:** 1.5–2.5 weeks (highest integration risk; rsmpi ergonomics + CI MPI setup).

### Phase 9 — `hdf5` (feature `hdf5`, deferred)
- `Hdf5Store: Store` via the `hdf5`/`hdf5-metno` crate; add the `Hdf5` arm to `Backend`. (HighFive
  is a C++ convenience layer with no Rust analogue — use the `hdf5` crate directly.)
- **Parity:** HDF5 checkpoint round-trips; cross-read a C++-written file where feasible.
- **Exit:** HDF5 tests pass with the feature enabled; build documented (needs `libhdf5`).
- **Est:** 3–5 days.

---

## 6. Timeline summary

| Milestone | Phases | Cumulative est. |
|---|---|---|
| **Core shipped (`0.1.0`)** | 0–3 | **~2.5–4 weeks** |
| Full single-process library (stats, grids, numerics, checkpointing) | 0–7 | ~6–8 weeks |
| Full parity incl. MPI + HDF5 | 0–9 | **~8–11 weeks** |

Estimates assume one experienced engineer pairing with agentic assistants (Claude Code / Codex).
Agents accelerate the mechanical bulk (trait scaffolding, RNG/accumulator/test translation, serde
wiring, examples); the human-paced bottlenecks are the §3.4 erasure design, MPI integration,
nalgebra↔Eigen gaps, and floating-point parity debugging.

---

## 7. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Type-erasure capability dispatch (§3.4) needs nightly/specialization | Med | Explicit-bound constructors (stable); autoref-spec only as fallback |
| RNG / sampler not bit-exact (esp. `uniform_real_distribution`) | Med | Port the exact transform; bit-exact fixtures; don't lean on `rand`'s `Uniform` on the parity path |
| Floating-point drift in reductions | Med | Match C++ summation order; tolerance-based asserts with documented bounds |
| `nalgebra` eigensolver/API differences vs Eigen | Med | Spike in Phase 0; cross-check eigen paths against fixtures |
| rsmpi maturity / CI MPI setup | Med-High | Deferred to Phase 8; isolate behind feature; containerized `mpirun` CI |
| HDF5 file-format cross-compatibility | Low | Treat as optional; document any schema divergence |
| Scope creep from "redesign freely" | Med | Parity harness is the hard gate; redesign only where it improves Rust ergonomics without changing behavior |

---

## 8. Definition of done (whole port)

- All nine phases complete; every subsystem's ported tests pass and fixtures match within tolerance.
- A fixed-seed end-to-end simulation (incl. checkpoint/restart, and an MPI reduction run) reproduces
  the C++ reference traces.
- `cargo build` (default features), `cargo build --features hdf5,mpi`, `cargo test`, `cargo clippy
  -D warnings`, and `cargo doc` all succeed in CI.
- README + rustdoc cover the trait-based API and the feature flags; tutorials ported as examples.
