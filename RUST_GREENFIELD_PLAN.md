# Greenfield Rust Monte Carlo framework — design plan

A design for a **new, from-scratch** Monte Carlo framework in Rust, covering the same scientific
scope as `simplemc` (RNG, statistical accumulators, grids, numerics, an MC driver, checkpointing,
and parallel execution) but **without any obligation to match the C++ implementation's source,
APIs, or bit-level outputs.** Freed from parity, the design optimizes for idiomatic Rust,
correctness against analytical truth, and Rust's strongest advantage for this domain: **fearless
parallelism.**

Working title: **`rmc`** (placeholder — rename freely). This plan keeps `simplemc`'s sequencing
philosophy — *ship the core first, do one subsystem end-to-end before the next, MPI and HDF5 last* —
but changes the *content* of the core: parallel execution and reproducible per-chain RNG move into
the core, not the tail.

> See `RUST_PORT_PLAN.md` for the faithful-port alternative. The two share a domain feature list but
> differ fundamentally in constraints and architecture.

---

## 1. Design philosophy

1. **No parity constraint.** Correctness is judged against **analytical truth** (known moments,
   known Ising critical temperature, closed-form integrals, exact polynomial reconstruction) and
   **property-based invariants** — not against the C++ outputs. There is no reference dumper, no
   bit-exact fixtures, no 644-test translation.
2. **Parallelism is a core, day-one capability.** Independent Markov chains are embarrassingly
   parallel and are *the* common MC parallelism pattern. The `run` API, RNG seeding, and result
   reduction are designed for multi-chain execution from the start. Threads/`rayon` are the default
   backend; MPI is an *optional* HPC-cluster backend that reuses the same reduction abstraction.
3. **Static dispatch on the hot path.** Updates run every MC step; a per-step vtable call is wasted
   overhead for cheap updates. Update sets are **monomorphized**. `dyn` is reserved for the cold
   path (measurements, run per cycle) and for an explicit opt-in dynamic-config path.
4. **No type-erasure result retrieval.** No `Any`/`downcast`/`typeid`. Results flow back through
   ownership: `run` *returns* the (typed) measurements/accumulators, or measurements write into a
   caller-owned sink.
5. **serde everywhere; HDF5 only for output data.** Checkpoints are serde blobs (compact binary +
   optional JSON). HDF5 is reserved for large scientific *output datasets*, behind a feature — not
   for framework plumbing. No bespoke navigable serializer abstraction.
6. **Reuse the ecosystem.** Use `rand` + `rand_xoshiro` for RNG (no hand-rolled engines),
   `nalgebra` for linear algebra, `rayon` for parallelism, `thiserror` for errors.
7. **Lean core, batteries opt-in.** A user who only needs the engine should not compile `nalgebra`.

---

## 2. Architecture

### 2.1 Workspace layout

A small Cargo **workspace** with a lean core and opt-in batteries/backends (the from-scratch payoff:
the engine crate stays dependency-light).

```
rmc/                       (workspace)
  crates/
    rmc-core/              # traits, run loop, RNG seeding, parallel execution, Merge
                           #   deps: rand, rand_xoshiro, rayon, thiserror, serde
    rmc-stats/             # accumulators (mean/var/covar/autocorr/batch/block/jackknife)
                           #   deps: rmc-core, nalgebra, num-complex
    rmc-grids/             # 1-D + N-D grids
    rmc-numeric/           # polynomials, interpolation, quadrature, lattices  (nalgebra)
    rmc-io/                # serde checkpointing; hdf5 output behind feature
    rmc-mpi/               # optional MPI chain-distribution + reduction backend (rsmpi)
    rmc/                   # façade: re-exports, feature flags, prelude
  examples/                # Ising, harmonic oscillator, etc.
  benches/                 # criterion: RNG, accumulators, run loop, parallel scaling
```

Features on the `rmc` façade: `default = ["stats", "grids", "numeric", "io"]`, plus `mpi`, `hdf5`.

### 2.2 Dependencies

| Concern | Crate | Notes |
|---|---|---|
| RNG | `rand`, `rand_xoshiro` (or `rand_pcg`) | off-the-shelf; no hand-rolled engines |
| Parallelism | `rayon` | data-parallel independent chains |
| Linear algebra | `nalgebra` (+ `num-complex`) | vectors, covariance matrices, `SymmetricEigen` |
| Errors | `thiserror` | `Result<_, RmcError>` |
| Serialization | `serde` + `bincode`/`postcard` + `serde_json` | binary checkpoints + human-readable |
| Ordered maps | `indexmap` | named sets preserve insertion order |
| Output data (opt) | `hdf5` / `hdf5-metno` | large measurement datasets only |
| MPI (opt) | `mpi` (rsmpi) | cluster backend; `#[derive(Equivalence)]` |
| Tests | `proptest`, `approx`, `rstest` | property + analytical + parametrized |
| Benches | `criterion` | parallel scaling, hot-path dispatch |

---

## 3. Core trait design

### 3.1 Scalars & samples
```rust
pub trait Scalar: Copy + /* arithmetic, conj */ + Serialize + DeserializeOwned {}  // f64, Complex64
pub trait SampleType { type Value: Scalar; fn size(&self) -> usize; }              // scalar | DVector
```

### 3.2 MC contracts
```rust
pub trait Update {
    fn attempt(&mut self) -> f64;     // acceptance probability
    fn accept(&mut self);
    fn reject(&mut self) {}           // default no-op
}
pub trait Measurement {
    type Output;                      // typed result; no downcast needed
    fn measure(&mut self);
    fn finish(self) -> Self::Output;  // results flow back by ownership
}
pub trait Kernel {
    fn step<R: Rng + ?Sized>(&mut self, rng: &mut R);
    fn prepare(&mut self) {}
}
pub trait RunCallbacks {
    fn on_step(&self, _: &Ctx) {}
    fn on_cycle(&self, _: &Ctx) {}
    fn on_checkpoint(&self, _: &Ctx) {}
    fn stop_when(&self, _: &Ctx) -> bool { false }
}
```

### 3.3 The unifying abstraction: `Merge`
The single most important trait. It expresses "combine results from independent chains" and is the
**one reduction primitive used by both the thread backend (now) and the MPI backend (later).**
```rust
/// Associative, commutative combination of partial results from independent chains/ranks.
pub trait Merge {
    fn merge(self, other: Self) -> Self;
}
```
Every accumulator implements `Merge` (two mean accumulators combine into one; covariance likewise).
Thread-parallel reduction is `rayon`'s `reduce(merge)`; the MPI backend performs the same `merge`
across ranks. Designing accumulators to be mergeable from day one is what makes both parallelism
backends fall out cleanly.

### 3.4 Update sets — monomorphized hot path
The kernel is generic over a concrete update set so per-step dispatch is **static**:
```rust
pub trait UpdateSet {
    fn select_and_step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> StepOutcome;  // weighted pick + attempt/accept/reject
    fn stats(&self) -> &[UpdateStats];
}
```
- **Default (fast):** a typed set built from concrete update types (tuple/HList, or a user-defined
  enum, optionally via a small derive macro), fully monomorphized — no vtable per step.
- **Opt-in (flexible):** `Vec<Box<dyn Update>>` for runtime/config-driven composition, accepting the
  dispatch cost. Explicit, not the default.

Measurements run per cycle (cold), so a `Vec<Box<dyn Measurement>>` set is fine there; static sets
are also offered for when results need to come back typed.

### 3.5 Accumulator hierarchy (`rmc-stats`)
```rust
pub trait Accumulator: Merge { type Sample: SampleType; type Value: Scalar;
    fn count(&self)->u64; fn size(&self)->usize; fn accumulate(&mut self, x:&Self::Sample); }
pub trait MeanAccumulator:       Accumulator { fn mean(&self)->Self::Sample; }
pub trait VarianceAccumulator:   MeanAccumulator { fn variance(&self)->/*sample*/; }
pub trait CovarianceAccumulator: VarianceAccumulator { fn covariance(&self)->DMatrix<Self::Value>; }
pub trait BatchAccumulator:      CovarianceAccumulator { fn batches(&self)->&[impl MeanAccumulator]; }
```
Online, numerically stable algorithms (Welford / batched). `Merge` lets per-chain accumulators be
reduced into a global one.

### 3.6 Grids (`rmc-grids`)
`GridCommon` / `Grid1d` / `GridNd<const N: usize>` with const-generic dimension; iterator impls
instead of bespoke iterator classes. (Same shape as the port's grid traits.)

---

## 4. Parallel execution model (the core differentiator)

### 4.1 Reproducible per-chain RNG
A master seed deterministically derives a **well-separated, non-overlapping** stream per chain, so
results are **reproducible regardless of thread scheduling**:
```rust
pub struct SeedSource { master: u64 }
impl SeedSource {
    pub fn rng_for(&self, chain: u64) -> Xoshiro256PlusPlus {
        // derive via splitmix-hashed (master, chain) seed, or xoshiro jump() per chain index
    }
}
```
Property test: distinct `chain` indices yield statistically independent streams; same master + same
chain count yields identical aggregate results across runs and thread counts.

### 4.2 Multi-chain run
```rust
pub fn run_parallel<S, K, M>(cfg: ParallelConfig, build: impl Fn(ChainId) -> (S, K, M) + Sync) -> M::Output
where K: Kernel, M: Measurement + Merge { /* rayon: per-chain run, then reduce via Merge */ }
```
- Each chain: own state `S`, kernel `K`, measurement set `M`, and `SeedSource::rng_for(chain)`.
- After all chains finish, partial results are reduced with `Merge` (rayon `reduce`).
- The MPI backend (Phase 8) provides the same signature with cross-rank `Merge` — **identical user
  code, different backend.**

This is the architecture's payoff: parallelism is a property of the framework, not an afterthought,
and the thread/MPI split is a single swappable reduction step.

---

## 5. Serialization & I/O (`rmc-io`)

- **serde derive** on all simulation state. Checkpoint = `bincode`/`postcard` (compact) or
  `serde_json` (human-readable); chosen by the caller. No custom navigable store trait.
- **Restart**: deserialize state + RNG (the chosen RNG crate types are `Serialize`); continue.
- **HDF5 (feature `hdf5`)**: for large scientific *output* arrays only (measurement time series,
  histograms), not framework checkpoints. Keeps the binary scientific format decoupled from the
  engine.

---

## 6. Validation strategy (no reference oracle)

Correctness is established intrinsically:

- **Analytical truth:** RNG/sampler moments vs closed forms; Ising 2-D vs known `T_c`; quadrature of
  known integrals; interpolation exact on polynomials up to its order; autocorrelation of an AR(1)
  process vs known integrated autocorrelation time.
- **Property tests (`proptest`):** mean of constants == constant; variance ≥ 0; covariance symmetric
  & PSD; grid `index(at(i)) == i`; serialization round-trips; **`Merge` associativity/commutativity**;
  **parallel result == sequential result** for the same seeds.
- **Reproducibility guard:** identical master seed + chain count ⇒ identical aggregate, independent
  of thread count / scheduling.
- **Statistical sanity:** light chi-square/KS on samplers (the `rand` engines are already
  battle-tested, so this is a thin layer).
- **Benchmarks (`criterion`):** confirm the monomorphized hot path has no per-step vtable cost; track
  parallel scaling.

Exit criterion per subsystem: analytical + property tests green, examples run, benches recorded.

---

## 7. Phased roadmap

Core-first; one subsystem end-to-end; MPI and HDF5 last. **Parallel execution is part of the core.**

### Phase 0 — Workspace & validation tooling
Workspace, crate skeletons, CI (`fmt`, `clippy -D warnings`, test matrix), `proptest`/`criterion`
setup. **Spike the reproducible per-chain RNG + rayon reduction** (the riskiest novel piece) before
committing to the API. **Est: 2–3 days.**

### Phase 1 — `rmc-core` foundation
`RmcError`/`Result`, `Scalar`/`SampleType`, RNG integration (`rand_xoshiro` + `SeedSource`), `Merge`
trait. Reproducibility property tests. **Est: 3–4 days.**

### Phase 2 — `rmc-core` MC engine (sequential)
`Update`/`Measurement`/`Kernel`/`RunCallbacks` traits; monomorphized `UpdateSet` (+ dyn fallback);
`MetropolisKernel` (owns its update set); sequential `run` loop → `Result`; `SimulationParams`/
`Stats`/`Ctx`. Analytical end-to-end test (single-chain Ising). **Est: ~1 week.**

### Phase 3 ★ — Parallel execution (CORE SHIP → `0.1.0`)
`run_parallel` over rayon; per-chain reproducible RNG; cross-chain reduction via `Merge`. Property
test: parallel == sequential for matched seeds; reproducibility across thread counts. **This is the
shippable core: a parallel MC engine.** **Est: 3–5 days.**

### Phase 4 — `rmc-stats`
Accumulator trait hierarchy (mean/var/covar/autocorr/batch/block/jackknife/multivalue), each `Merge`.
Scalar + nalgebra-vector samples; stable online algorithms. Analytical + property validation.
**Est: 1–1.5 weeks.**

### Phase 5 — `rmc-grids`
Linear/power/symmetric-power/custom 1-D grids, N-D grids (const generics), iterator impls.
**Est: ~1 week.**

### Phase 6 — `rmc-numeric`
Orthogonal polynomials, interpolation, quadrature, Bravais lattices, linear map (nalgebra;
`SymmetricEigen` where needed). **Est: 1.5–2 weeks.**

### Phase 7 — `rmc-io`
serde checkpoint save/restore (bincode/json); RNG-state round-trip; restart-continues-run test.
**Est: 3–5 days.**

### Phase 8 — `rmc-mpi` (optional backend)
rsmpi backend implementing the **same `run_parallel`/`Merge` contract across ranks**; datatypes via
`#[derive(Equivalence)]`; multi-process CI under `mpirun`. User code unchanged from the thread
backend. **Est: 1–2 weeks.**

### Phase 9 — `rmc-io` HDF5 output (optional)
`hdf5`-crate output for large measurement datasets, behind the `hdf5` feature. **Est: 3–5 days.**

---

## 8. Timeline summary

| Milestone | Phases | Cumulative est. |
|---|---|---|
| **Parallel core shipped (`0.1.0`)** | 0–3 | **~2.5–3.5 weeks** |
| Full single-node library (stats, grids, numerics, checkpointing) | 0–7 | ~6–8 weeks |
| Full incl. MPI + HDF5 | 0–9 | **~8–10 weeks** |

Roughly comparable to the port, but the work is redistributed: the parity harness, hand-rolled RNG,
and 644-test translation are gone (saving time), while a proper parallel core and analytical
validation suite are added (spending some of it). The result is a *better* framework for the same
budget, because effort goes into design rather than reproduction.

---

## 9. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Per-chain RNG streams overlap / non-reproducible | Med | Spike in Phase 0; well-separated seeds (xoshiro `jump()` or splitmix-hashed); independence + reproducibility property tests |
| Monomorphized update-set ergonomics (macro complexity) | Med | Ship `dyn` path first; add typed set + optional derive macro incrementally |
| `Merge` abstraction leaks between thread & MPI backends | Med | Constrain to associative+commutative combine; test parallel == sequential before MPI work |
| nalgebra eigensolver/API gaps vs Eigen | Med | Phase 0 spike; validate eigen paths against analytical results |
| rsmpi maturity / MPI CI setup | Med-High | Deferred, feature-gated; containerized `mpirun` CI; same contract as thread backend limits surface |
| HDF5 build complexity (`libhdf5`) | Low | Optional feature; output-only scope |
| Over-design from greenfield freedom | Med | Analytical/property tests + "ship core at Phase 3" are the discipline; YAGNI on generality |

---

## 10. Definition of done

- Phases 0–9 complete; every subsystem's analytical + property tests pass, examples run, benches
  recorded.
- A parallel, fixed-seed Ising simulation is **reproducible across thread counts** and reproduces
  known physics (`T_c`); the **MPI backend yields the same aggregate as the thread backend**.
- `cargo build`/`test`/`clippy -D warnings`/`doc` green for default and `--features mpi,hdf5`.
- Benchmarks confirm no per-step vtable cost on the default update path and show parallel scaling.
- Rustdoc + README document the trait API, the parallel model, and feature flags; tutorials shipped
  as examples.

---

## Appendix — key divergences from the faithful port

| Aspect | Port (`RUST_PORT_PLAN.md`) | Greenfield (this doc) |
|---|---|---|
| Correctness oracle | bit/tolerance parity vs C++ fixtures | analytical truth + property tests |
| RNG | hand-port xoshiro/splitmix for bit-parity | off-the-shelf `rand_xoshiro` |
| Parallelism | MPI only, last phase | thread/rayon in core (Phase 3), MPI optional backend |
| Hot-path dispatch | type-erased (vtable per step, as in C++) | monomorphized update set; `dyn` cold path only |
| Result retrieval | `Any`/`get::<T>()` downcast | typed return by ownership; no downcast |
| Serialization | bespoke navigable `Store` + variant backends | serde everywhere; HDF5 = output data only |
| Test suite | translate 644 gtest cases | native property/analytical tests |
| Crate shape | single crate + features | lean-core workspace + batteries/backends |
