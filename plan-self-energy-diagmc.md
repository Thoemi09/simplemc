# Plan: Port self-energy Diagrammatic Monte Carlo to the Rust framework

## 0. Purpose and audience

This document is an **executable implementation plan** for porting the C++ self-energy
Diagrammatic Monte Carlo (DiagMC) code in `self-energy-dmc/` to the greenfield Rust Monte Carlo
framework in `rust/`. The goal is two-fold:

1. Produce a working Rust polaron DiagMC that reproduces the physics of the C++ version.
2. **Stress-test the Rust framework** (`rmc-core`, `rmc-stats`, `rmc-grids`, `rmc-numeric`,
   `rmc-io`) against a real, non-trivial application, surfacing API gaps.

The executor is assumed to be a competent Rust programmer of *medium* domain knowledge. Therefore
this plan ports **formulas verbatim** from the C++ source (with file/function references) rather
than re-deriving the physics. When in doubt, the C++ source is the ground truth.

### Decisions already made (do not revisit)

- **FFT/Dyson stage**: included. Add the `rustfft` crate as a workspace dependency and port
  `fourier_transform.hpp`.
- **Location**: a new binary crate `rust/crates/rmc-polaron` (a workspace member), with library
  modules + a `main.rs` runner + integration tests.
- **Validation bar**: physics validation within statistical error bars (assert E and Z against
  reference values after a fixed-seed run), plus per-step diagram sanity invariants.
- **Feature breadth**: include `run_parallel` (rayon) reduction and a JSON checkpoint round-trip.
- **Performance parity is a goal** (see `RUST_GREENFIELD_PLAN.md`). The diagram must support
  **O(1) structural updates** matching the C++ `std::list` + vector-of-pointers design. The data
  model is therefore a **`slotmap` arena + intrusive doubly-linked list**, *not* a sorted `Vec`.
  See §4 — this supersedes the earlier `arc_id` + `rebuild_links` sketch.

### Current implementation state (2026-06-30)

A first pass of `rmc-polaron` already exists (~1.8k LOC: `diagram.rs`, `updates/`, `measurement.rs`,
`fourier.rs`, `sanity.rs`, `config.rs`, `app.rs`, `main.rs`, plus `tests/`). It was written against
the **superseded** sorted-`Vec<Vertex>` + `arc_id` + `rebuild_links()` model, whose
`rebuild_links()` is **O(n²) per structural update** — the opposite of the C++ O(1) goal. The
updates and measurement currently address vertices by **positional `usize` index** and rely on
ordering (`len-2`, `0..len-1`, `i±1`, ranges `vertex1..vertex2`) and `uniform_index(rng, len)` for
random pick. **Phase 1 below is now a migration task** to the slotmap model, with §4.4 mapping each
positional idiom to its key-based replacement. Phases 2–5 are largely in place and need only the
mechanical index→key substitution plus re-validation.

### The physics in one paragraph (context only)

We compute the imaginary-time self-energy `Σ(p, τ)`, ground-state energy `E(p)`, and quasiparticle
weight `Z(p)` of the **Fröhlich polaron** (one electron coupled to a dispersionless phonon bath,
`ω = 1`, `m = 1`). A "diagram" is an electron world-line on `[0, τ]` dressed by phonon arcs. The MC
samples diagram configurations (order = number of phonon arcs, vertex times, momenta) weighted by
the diagram value. Order 0 is a special **normalization sector** (the "fake function") used to fix
the absolute scale. The reference result for `α = 1, p = 0` is `E ≈ -1.013` and `Z ≈ 0.59`.

---

## 1. Source-of-truth map (C++ → what it does)

Read these before writing code. Paths are under `self-energy-dmc/src/`.

| C++ file | Role | Port target (Rust) |
|---|---|---|
| `diagram.hpp` | Diagram state: vertices, momenta, dispersion, propagator, `set_to_0th_order`, `add_phonon`, `get_p_mean`, `exact_estimator`, constants (`MASS=1`, `OMEGA=1`, `P0=sqrt(2)`) | `diagram.rs` |
| `vertex.hpp` | Per-vertex data: `tau_`, `p_out_`, `q_`, `phonons_above_`, `link_` | `Vertex` in `diagram.rs` |
| `updates/add_phonon.hpp`, `remove_phonon.hpp` | Order ±1 (paired, inverse) incl. the order-0 "fake" sector | `updates/phonon.rs` |
| `updates/change_tau.hpp` | Move the final time `τ` (the measured time) | `updates/mod.rs` |
| `updates/change_internal_tau.hpp` | Move an internal vertex time | `updates/mod.rs` |
| `updates/rescale_diagram.hpp` | Rescale all times by a common factor | `updates/mod.rs` |
| `updates/change_internal_q_modulus.hpp` | Change a phonon momentum magnitude | `updates/mod.rs` |
| `updates/change_internal_q_direction.hpp` | Rotate a phonon momentum | `updates/mod.rs` |
| `updates/change_topology.hpp` | Swap two adjacent vertices' arc connectivity | `updates/mod.rs` |
| `measurements.hpp` (`histogram`) | Batched accumulators; reweighted estimators; self-consistent `E` re-estimation; jackknife of Σ/E/Z | `measurement.rs` |
| `sanity_checker.hpp` | Debug invariants: momentum conservation, time ordering, links, `phonons_above` | `sanity.rs` |
| `utils.hpp` | `draw_from_exp`, `spherical_to_cartesian`, `theta/phi_from_cartesian`, `random_from`, `norm0` | reuse `rmc-core::random` + small helpers in `diagram.rs` |
| `fourier_transform.hpp` | Σ(τ)→Σ(iω) via FFT, Dyson G=1/(1/G0−Σ), back-transform | `fourier.rs` |
| `main.cpp` | Wiring: build updates, run, write results, FFT | `main.rs` + `app.rs` |
| `run_properties.hpp` | Output metadata JSON | part of `app.rs` output |
| `runs/input.json` | Reference run parameters | default config in `config.rs` |

The C++ `simplemc::*` symbols map to Rust framework symbols as in §2.

---

## 2. Framework API the port will use (verified against `rust/`)

This section is the contract between the port and `rmc-core`. All items below exist today.

### Traits (`rmc_core::mc`)
- `Update<State>`: `fn attempt<R: Rng + ?Sized>(&mut self, state: &mut State, rng: &mut R) -> f64`
  (return `<0` ⇒ impossible/rejected, `>=1` ⇒ always accept, else accept w.p. value);
  `fn accept(&mut self, state: &mut State)`; `fn reject(&mut self, _: &mut State) {}` (default no-op).
  **Important**: in this framework `attempt` returns the acceptance *probability* directly. The C++
  `attempt_impl` already returns a Metropolis ratio R; return that R unchanged. The framework treats
  `R >= 1` as certain-accept and `R < 0` as impossible — this matches C++ `0.` meaning reject and
  positive ratios meaning "accept w.p. min(1,R)". So **return exactly the C++ ratio**, and return a
  negative number (e.g. `-1.0`) where you want an explicit impossible. Where C++ returns `0.` to
  reject, returning `0.0` is fine (accepted with probability 0 ⇒ rejected). Prefer `0.0` to mirror
  C++ unless you specifically want it counted as "impossible".
- `Measurement<State>`: `type Output; fn measure(&mut self, &State); fn finish(self) -> Output`.
- `Merge`: `fn merge(self, other: Self) -> Self` (sum-like reduction for parallel chains).
- `Kernel<State, R>`, `SteppingUpdateSet<State, R>`, `UpdateSet`.

### Update sets and kernel
- Heterogeneous updates (we have 8) must be a **single enum** `PolaronUpdate` that implements
  `Update<Diagram>` by dispatching to the variant, stored in
  `WeightedUpdateSet<PolaronUpdate>` (see `sets.rs`). This keeps dispatch monomorphized (no
  `Box<dyn Update>`), which is the framework's intended pattern for heterogeneous static sets
  (see the doc comment on `WeightedUpdateSet`).
- The add/remove pair is an **inverse pair**: build the set with explicit `WeightedUpdate::with_ratio`
  entries. The C++ already bakes the `(2n-1)/n` selection-ratio factors **into the returned ratio**
  inside `add_phonon`/`remove_phonon` (see `algo_ratio` in both files). Therefore set the
  framework-level `ratio = 1.0` for every entry (including add/remove) and let the update return the
  full ratio itself. Use `WeightedUpdate::new(update, weight)` (ratio defaults to 1.0) for all eight.
- Kernel: `MetropolisKernel::new(update_set)` implements `Kernel<Diagram, R>`.

### Run loop
- Single chain returning a typed output:
  `run_typed(state, &mut rng, &mut kernel, measurement, params) -> Result<(State, SimulationStats, M::Output)>`.
- With callbacks (for checkpoint/convergence/log): `run_typed_with_callbacks(..., &mut callbacks)`
  where `callbacks: RunCallbacks<SimulationCtx>` (hooks: `on_step`, `on_cycle`, `on_checkpoint`,
  `stop_when`). `SimulationParams { max_steps, steps_per_cycle, cycles_per_check }`.
- Parallel: `run_parallel(ParallelConfig { chains, seed, params }, build)` where
  `build: Fn(ChainId) -> (State, K, M)` and `M::Output: Merge`. Reduction is order-preserving and
  reproducible.

### RNG and sampling (`rmc_core::random`)
- `SeedSource::new(master)`, `seed.rng_for(ChainId(i))`, `ChainId(u64)`, `DefaultRng = Xoshiro256++`.
- `uniform_index(rng, len)` ⇒ uniform pick; apply it to the diagram's `storage: Vec<VKey>` to
  replace C++ `random_from(vertex_storage_, ...)`: `storage[uniform_index(rng, storage.len())]` (§4).
- `rng.gen::<f64>()` ⇒ uniform `[0,1)` (replaces C++ `uniDist_(rng)`).
- `exponential_sample_bounded(r, lambda, a, b)` and `safe_exponential_sample(r, lambda, a, b)`
  ⇒ replace C++ `draw_from_exp`. **Check the sign convention**: C++ `draw_from_exp(begin,end,lambda,r)
  = begin - 1/lambda * ln(1 - r*(1 - exp(-lambda*(end-begin))))`. This equals
  `exponential_sample_bounded(r, lambda, begin, end)` **only for `lambda > 0`**. The C++ code calls
  `draw_from_exp` with possibly-negative `lambda` (e.g. `change_internal_tau`). For negative/zero
  lambda use `safe_exponential_sample(r, lambda, a, b)` which handles all signs. **Action**: write a
  unit test asserting `safe_exponential_sample` reproduces C++ `draw_from_exp` for a few
  `(lambda, a, b, r)` triples with `lambda>0`, then audit each call site's lambda sign and pick the
  matching helper (see §5 per-update notes).
- For Gaussian draws (`rescale_diagram`, `change_internal_q_modulus`) the C++ uses
  `std::normal_distribution`. The framework has only `normal_pdf`, not a sampler. **Action**: add
  `rand_distr` to the crate deps and use `rand_distr::Normal`, OR implement Box–Muller locally in
  `diagram.rs` (`fn sample_normal(rng, mu, sigma) -> f64`). Prefer `rand_distr::Normal` (well-tested).

### Vectors
- `nalgebra` is a workspace dependency. Use `nalgebra::Vector3<f64>` for momenta (replaces
  `Eigen::Vector3d`). Methods used: `.norm()`, `.norm_squared()` (= Eigen `squaredNorm`), `.dot()`,
  `.normalize()`, scalar `* / + -`, `Vector3::zeros()`, `Vector3::new(x,y,z)`.

### Stats (`rmc_core`'s `rmc-stats`, re-exported via `rmc::stats` when using the facade; here depend
on `rmc-stats` directly)
- Available: `ScalarMoments`, `ScalarBatchMeans`, `ScalarBlockMeans` (+ `.jackknife() ->
  ScalarJackknife`), `VectorMoments`, all `Merge`.
- **Gap**: there is no *ratio* jackknife over two correlated batched series (the C++
  `jackknife_propagate(num, den, fn)`). We need it for Σ = exact/zeroth, E = energy/zeroth,
  Z = f(A/zeroth). **Action**: implement a local `BatchedRatio` accumulator + delete-one jackknife in
  `measurement.rs` (spec in §6). Record this as a framework finding: "rmc-stats should grow a paired
  ratio jackknife."

### IO (`rmc-io`)
- `Checkpoint::new(payload)`, `save_json/load_json`, `save_payload_json/load_payload_json`,
  `to_json_string`, `from_json_str`. Payload must be `Serialize + DeserializeOwned`.
- The `serde` feature of `rmc-core` must be enabled to serialize framework types; our own state
  (`Diagram`, measurement accumulators) derives `serde` directly.

---

## 3. Crate scaffold (Phase 0)

Create `rust/crates/rmc-polaron/`:

```
rmc-polaron/
  Cargo.toml
  src/
    lib.rs          # pub mod re-exports
    diagram.rs      # Diagram, Vertex, dispersion/propagator, p_mean, exact_estimator, helpers
    sanity.rs       # debug invariants (ported sanity_checker)
    updates/
      mod.rs        # PolaronUpdate enum + Update impl dispatch + the 6 single updates
      phonon.rs     # AddPhonon, RemovePhonon (incl. order-0 fake sector)
    measurement.rs  # PolaronMeasurement, BatchedRatio, estimators, self-consistent E
    fourier.rs      # FFT/Dyson post-processing
    config.rs       # RunConfig (serde), defaults matching runs/input.json
    app.rs          # build_chain, run wiring, output writing, checkpoint callbacks
  src/main.rs       # CLI entry (parse config path, run, write outputs, FFT)
  tests/
    diagram_ops.rs       # add/remove round-trip, sanity invariants
    sampling.rs          # draw_from_exp parity, spherical<->cartesian round-trip
    physics_validation.rs# E and Z within error bars (the acceptance gate)
```

`Cargo.toml`:
```toml
[package]
name = "rmc-polaron"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[dependencies]
rmc-core = { workspace = true, features = ["serde"] }
rmc-stats.workspace = true
rmc-grids.workspace = true
rmc-numeric.workspace = true
rmc-io.workspace = true
nalgebra = { workspace = true, features = ["serde-serialize"] }
slotmap = { version = "1", features = ["serde"] }
rand.workspace = true
rand_distr = "0.4"
rustfft = "6"
num-complex.workspace = true
serde = { workspace = true }
serde_json.workspace = true

[dev-dependencies]
proptest.workspace = true
```

Add `rand_distr = "0.4"`, `rustfft = "6"`, and `slotmap = { version = "1", features = ["serde"] }`
(and `serde-serialize` for nalgebra) to the `[workspace.dependencies]` section of `rust/Cargo.toml`,
and add `"crates/rmc-polaron"` to `[workspace].members`.

**Phase 0 done when**: `cargo build -p rmc-polaron` compiles an empty lib and `cargo metadata`
shows the new member.

---

## 4. Diagram data model (Phase 1 — slotmap migration)

This is the highest-risk part of the port **and** the performance-parity vehicle. C++ achieves fast
updates with two cooperating structures: a `std::list<vertex>` (stable node identity + **ordered**
O(1) splice/erase + `std::next`/`std::prev` neighbour access) and a parallel `vertex_storage_`
(vector of iterators for O(1) uniform random pick). The Rust port must reproduce **both**
properties to match C++ performance.

**Chosen model: a `slotmap` arena + an intrusive doubly-linked list, plus a key-storage vector for
random pick.** `slotmap` supplies the stable-handle node pool (generational keys, O(1) insert/remove/
get, contiguous slot storage — cache behaviour equal-or-better than `std::list`'s scattered heap
nodes). The intrusive `prev`/`next` keys supply the ordering that `slotmap` itself does not provide.
This is the faithful structural analog of the C++ design; complexity is the same class (O(1)
structural updates), with only a negligible per-`get` generation check as overhead.

> This **supersedes** the sorted-`Vec<Vertex>` + `arc_id` + `rebuild_links()` model in the first
> draft (and in the current partial implementation), whose `rebuild_links()` is O(n²) per update.

Add `slotmap = { version = "1", features = ["serde"] }` to `[workspace.dependencies]` and to
`rmc-polaron`'s `Cargo.toml`. The `serde` feature is required so checkpoints still serialize
(slotmap keys and maps serialize under it).

### 4.1 Types

```rust
use nalgebra::Vector3;
use slotmap::{new_key_type, SlotMap, Key};

new_key_type! { pub struct VKey; }   // generational vertex handle; VKey::null() = "no vertex"

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Vertex {
    pub tau: f64,
    pub p_out: Vector3<f64>,  // electron-line momentum leaving this vertex to the right
    pub q: Vector3<f64>,      // phonon momentum of this vertex's arc
    pub phonons_above: usize, // arcs spanning over this vertex, EXCLUDING its own arc
    pub link: VKey,           // phonon-partner vertex (VKey::null() at order 0 / unset)
    pub prev: VKey,           // previous vertex in tau order (null for head)
    pub next: VKey,           // next vertex in tau order (null for tail)
    pub storage_idx: usize,   // this vertex's slot in Diagram::storage (mirrors C++ storage_link_)
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Diagram {
    pub arena: SlotMap<VKey, Vertex>, // node pool; never iterated for order — walk prev/next instead
    pub head: VKey,                   // first vertex (tau = 0)
    pub tail: VKey,                   // last vertex (tau = measured τ)
    pub storage: Vec<VKey>,           // O(1) uniform random pick (replaces C++ vertex_storage_)
    pub order: usize,                 // number of phonon arcs
    pub alpha: f64,
    pub mu: f64,
    pub momentum: f64,                // |p| of the external electron, along +x
    pub max_tau: f64,
    pub start_tau: f64,
    pub min_order: usize,
    pub max_order: usize,
}

impl Diagram {
    pub const MASS: f64 = 1.0;
    pub const OMEGA: f64 = 1.0;
    pub const DELTA_TAU_LIMIT: f64 = 10.0 * f64::EPSILON;
    pub fn p0() -> f64 { (2.0 * Self::MASS * Self::OMEGA).sqrt() } // C++ P0 = sqrt(2)
}
```

Helper accessors keep call sites readable: `fn v(&self, k: VKey) -> &Vertex { &self.arena[k] }`,
`fn vm(&mut self, k) -> &mut Vertex`, `fn next(&self, k) -> VKey { self.arena[k].next }`,
`fn prev(&self, k) -> VKey`, `fn is_tail(&self, k) -> bool { k == self.tail }`.

### 4.2 Key translation rules (memorize)

The C++ iterator idioms map to **keys + intrusive links**, not to positional indices. Let `k` be a
vertex key.

| C++ | Rust |
|---|---|
| `vertices_.begin()` | `self.head` |
| `vertices_.back()` / `std::prev(end())` | `self.tail` |
| `vertices_.end()` (one past) | `VKey::null()` (never index it) |
| `std::next(it)` | `self.arena[k].next` (`== null` past the tail) |
| `std::prev(it)` | `self.arena[k].prev` (`== null` before the head) |
| 2nd-last vertex (`std::prev(end(), 2)`) | `self.arena[self.tail].prev` |
| `v.link_->link_` | `self.arena[self.arena[k].link].link` (must equal `k`) |
| `v.is_incoming()` (`link.tau < tau`) | `self.arena[self.arena[k].link].tau < self.arena[k].tau` |
| `v.is_outgoing()` | partner tau `>` own tau |
| `random_from(vertex_storage_, …)` | `self.storage[uniform_index(rng, self.storage.len())]` |
| iterate vertices in order | `let mut k = self.head; while !k.is_null() { …; k = self.arena[k].next; }` |
| range `[v1, v2)` (`std::for_each(v1, v2, …)`) | walk `k = v1; while k != v2 { …; k = next(k); }` |

**Critical**: where C++ compares against `vertices_.end()` to mean "the proposed vertex sits beyond
the last vertex," use `next(k).is_null()` / a `VKey::null()` partner. The acceptance gains an extra
external-tail dispersion term in exactly those branches (see `add_phonon::higher_order_impl` and
`remove_phonon::attempt_higher_order_impl`) — port them with the null-key test.

### 4.3 Structural mutators — O(1), matching C++ (cover with `tests/diagram_ops.rs`)

Every mutator maintains four invariants: tau-ordering of the `prev`/`next` chain, `head`/`tail`
correctness, phonon `link` symmetry, and the `storage`↔`storage_idx` correspondence. No global
rebuild — each edit touches only O(1) neighbours, like the C++ list splice.

- **`set_to_0th_order(&mut self)`**: clear arena/storage; insert two vertices at `tau=0` and
  `tau=start_tau`, both `p_out=(momentum,0,0)`, `q=0`, `link=null`; chain them
  (`head.next=tail`, `tail.prev=head`, ends `null`); push both keys to `storage` recording each
  `storage_idx`; `order=0`. Mirror `diagram.hpp::set_to_0th_order`.
- **`push_storage(&mut self, k)` / `swap_remove_storage(&mut self, k)`**: O(1) storage maintenance.
  Removal mirrors C++ `iter_swap(...storage_link_, last); pop_back()`: move the last key into the
  removed slot, fix that moved vertex's `storage_idx`, `pop`. This is the parity-critical detail —
  do **not** linear-scan `storage` on removal.
- **`splice_after(&mut self, left: VKey, v: Vertex) -> VKey`**: insert `v` into the arena, wire it
  between `left` and `left.next` (fix four `prev`/`next` fields), update `tail` if it became last,
  push to storage. O(1).
- **`unlink(&mut self, k)`**: detach `k` from the `prev`/`next` chain, fix `head`/`tail`, swap-remove
  from storage, `arena.remove(k)`. O(1).
- **`insert_arc(&mut self, tau1, tau2, q)`** and **`remove_arc(&mut self, a: VKey, b: VKey)`**: the
  add/remove-phonon structural cores. Insert: find the two splice points by walking `next` from the
  relevant vertex, `splice_after` both new vertices, set their mutual `link`, then apply the C++
  `phonons_above` `++` deltas and subtract `q` from `p_out` over the spanned range. Remove: reverse.
  Port the `phonons_above` bookkeeping **literally** from C++ (it is physics, not connectivity), then
  verify with `sanity::check_phonons_above`.

Because identity is now stable (generational keys, not shifting indices), `link`/`prev`/`next` are
maintained directly and never invalidated by other edits — this is what the `arc_id`+`rebuild_links`
sketch was working around, now unnecessary. A captured `VKey` from `attempt` stays valid into
`accept` (no structural change happens in between).

### 4.4 Migrating the existing implementation

The current `diagram.rs` + `updates/` + `measurement.rs` use positional `usize` indices. Migrate
mechanically:

1. Replace `Vec<Vertex>` with the arena/keys above; delete `arc_id`, `next_arc_id`, `rebuild_links`,
   `NO_LINK`.
2. Change update structs to stage `VKey` instead of `usize` (the `vertex1_`/`vertex2_` analogs).
3. Replace each positional idiom per the §4.2 table: `len-2` → `prev(tail)`; `i+1`/`i-1` →
   `next(k)`/`prev(k)`; `0..len-1` and `vertex1..vertex2` → `prev`/`next` walks;
   `uniform_index(rng, len)` over the Vec → index into `storage`.
4. `measurement.rs` / `exact_estimator` iterate via the head→tail walk.
5. Re-run `tests/diagram_ops.rs`, `tests/updates.rs`, `tests/measurement.rs`, then the physics gate.

Keep `sanity::check_sanity` enabled in debug throughout the migration — it catches a mis-wired
`prev`/`next`/`link`/`storage_idx` immediately.

### 4.5 Physics methods (port verbatim from `diagram.hpp`)

- `dispersion(p) = p.norm_squared()/(2*MASS) - mu`
- `bare_dispersion(p) = p.norm_squared()/(2*MASS)`
- `bare_propagator(p, tau) = exp(-dispersion(p)*tau)`
- `momentum_vec() -> Vector3` = `Vector3::new(momentum, 0, 0)`; `tau()` = `vertices.last().tau`;
  `momentum_out()` = `vertices.last().p_out`.
- `get_p_mean(begin, end, addition)` (static, over a vertex range) and the `(tau1, tau2, begin)`
  overload — port both from `diagram.hpp` exactly; they compute the time-averaged electron momentum
  over an interval and are used by add/remove and the q-updates.
- `exact_estimator(&self, t0) -> f64` — port `diagram.hpp::exact_estimator` verbatim. It reweights
  the measured `τ` to the bin center `t0`. Note `v.link_->link_` self-reference and
  `std::next(v.link_->link_)`; since `arena[arena[k].link].link == k`, `std::next(v.link_->link_)` is
  just `arena[k].next`. So the dispersion sum term is `Σ_k dispersion(p_out_k) * (tau_{next(k)} -
  tau_k)` walking `head` up to (but excluding) `tail`. The second sum adds `Σ OMEGA * (tau_k -
  tau_link(k))` for outgoing arcs (positive delta), over all vertices. Reproduce both sums and the
  prefactor `(t0/tau)^(2*(order-1)) * exp(-lambda*…)`, `lambda = t0/tau - 1`.
- Free fn `norm0(max_tau, energy) = (1 - exp(-energy*max_tau))/energy` (from `diagram.hpp`).

### 4.6 Diagram factory (optional but recommended for tests)

Port `diagram_factory.hpp` as `Diagram::from_arcs(order, tau, &[(Vector3, …)])` building an explicit
order-`n` diagram with evenly spaced vertex times. Used by `tests/diagram_ops.rs` to construct known
configurations and check `exact_estimator`, `get_p_mean`, and sanity invariants.

**Phase 1 done when**: `tests/diagram_ops.rs` passes: build order-0/1/2 diagrams; insert and remove
an arc and assert the diagram returns to the prior state (taus, momenta, links, `phonons_above`);
`sanity::check_sanity` passes on all of them.

---

## 5. Updates (Phase 2)

### 5.1 The enum and dispatch

```rust
pub enum PolaronUpdate {
    ChangeTau(ChangeTau),
    ChangeInternalTau(ChangeInternalTau),
    AddPhonon(AddPhonon),
    RemovePhonon(RemovePhonon),
    RescaleDiagram(RescaleDiagram),
    ChangeQModulus(ChangeQModulus),
    ChangeQDirection(ChangeQDirection),
    ChangeTopology(ChangeTopology),
}

impl Update<Diagram> for PolaronUpdate {
    fn attempt<R: rand::Rng + ?Sized>(&mut self, d: &mut Diagram, rng: &mut R) -> f64 {
        match self { /* dispatch to inner.attempt */ }
    }
    fn accept(&mut self, d: &mut Diagram) { match self { /* dispatch */ } }
    // reject: default no-op is correct because every C++ attempt_impl is non-mutating
    // (it only stages tau_/q_/vertex indices into the update struct). Verify this while porting:
    // NONE of the attempt_impls mutate `diag_`; all mutation is in accept_impl. Keep it that way.
}
```

Each concrete update is a small struct holding the staged proposal (the C++ members like `tau_prime_`,
`q_`, `vertex1_`, `vertex2_`). **In Rust store vertex selections as `usize` indices, not references**,
captured during `attempt` and consumed in `accept`. Because `attempt` does not mutate the diagram and
no structural change happens between `attempt` and `accept` of the *same* update, indices stay valid.

`attempt` signature note: the framework passes `state: &mut Diagram` even though C++ `attempt_impl`
treats the diagram as const. Take `&mut` but only read it; do all writes in `accept`.

### 5.2 Per-update porting notes (return the C++ ratio directly)

For each update, port `attempt_impl` → `attempt` (returns the ratio) and `accept_impl` → `accept`.
Drop the `update_advice` wrapper (sanity checks): instead, gate `sanity::check_sanity(d)` behind
`debug_assert!`-style calls invoked from the kernel-side wrapper or inside `accept` under
`#[cfg(debug_assertions)]`. Specific gotchas:

- **`change_tau`** (`change_tau.hpp`): `lambda = dispersion(p_out of 2nd-last vertex) + (order!=0 ?
  OMEGA : 0)`. Draw `tau' = draw_from_exp(tau_{2nd_last}, max_tau, lambda, r)`. **lambda may be any
  sign** (dispersion can be negative since `mu<0` makes it ≥ +1.1 here, but stay general): use
  `safe_exponential_sample`. Returns ratio `1.0`. `accept`: set `tail` vertex `tau = tau'`.
  The "2nd-last vertex" is `prev(tail)`.
- **`change_internal_tau`** (`change_internal_tau.hpp`): early-return `0.0` if `order <= 1`. Pick a
  random key `k` from `storage` that is neither `head` nor `tail` (loop). `tau_next = (next(k) ==
  tail ? … )` — re-read: C++ uses `std::next(vertex_) == end()` to switch the upper bound to
  `max_tau`. Since `k` is never `tail` here, the upper bound is `max_tau` iff `next(k).is_null()`
  (cannot happen for non-tail) else `tau_{next(k)}`; in practice `tau_next = tau_{next(k)}`. Follow
  the C++ test literally. `lambda = bare_dispersion(p_{prev(k)}) - bare_dispersion(p_k) +
  (is_incoming(k)? +OMEGA : -OMEGA)`. Use `safe_exponential_sample` (lambda any sign). Reject (`0.0`)
  if `tau'` not finite. `accept`: set vertex `k` `tau = tau'` then — because
  `tau` changed, re-sort is **not** needed since the draw is bounded within `(tau_prev, tau_next)`
  so ordering is preserved; assert it.
- **`add_phonon`** / **`remove_phonon`** (`phonon.rs`): the most involved. See §5.3.
- **`rescale_diagram`** (`rescale_diagram.hpp`): `0.0` if `order==0`. Compute `E` via the loop
  (sum over propagators of `delta_s_i*(E_i + OMEGA*phonon_count)`, `+(-mu)` initial). Draw
  `tau' ~ Normal(2n/E, sqrt(2n)/E)` with `n = order-1` (use `rand_distr::Normal`). Reject if
  `tau'<0 || tau'>max_tau || !finite`. Acceptance = `exp(2n*ln(tau'/τ) - E*(tau'-τ) + ((E*tau'-2n)^2
  - (E*τ-2n)^2)/(4n))`; reject if non-finite. `accept`: multiply every vertex `tau` by
  `tau'/τ`.
- **`change_internal_q_modulus`** (`change_internal_q_modulus.hpp`): `0.0` if `order==0`. Pick random
  vertex `i`, partner `j = link(i)`; order them so `tau_i < tau_j`. `q0 = get_p_mean(i,j,q_i).dot(
  q_i.normalized())`; `s = sqrt(MASS/(tau_j - tau_i))`. Draw `q' ~ Normal(q0, s)`; reject if `q'<0 ||
  !finite`. Returns `1.0`. `accept`: set new vector `new_q = q_i.normalize()*q'`; for vertices in
  `[i, j)` do `p_out += q_i; p_out -= new_q`; set `q_i = q_j = new_q`.
- **`change_internal_q_direction`** (`change_internal_q_direction.hpp`): `0.0` if `order==0`. Same
  vertex/partner selection. Compute `p_mean = get_p_mean(i,j,q_i)`, `A = (tau_j-tau_i)*p_mean.norm()*
  q_i.norm()/MASS`. Draw `phi = 2π·r`; `theta = acos(ln(-2·r2·sinh(A) + exp(A))/A)`. Build rotation
  aligning +z to `p_mean` (use `nalgebra::Rotation3` from Euler `Z(phi_base)·Y(theta_base)` where
  `theta_base = theta_from_cartesian(p_mean)`, `phi_base = phi_from_cartesian(p_mean)`). `q' = rotM *
  spherical_to_cartesian(|q_i|, theta, phi)`. Reject if `q'` has NaN. Returns `1.0`. `accept`: as in
  q_modulus but with the rotated `q'`.
- **`change_topology`** (`change_topology.hpp`): `0.0` if `order==0`. Pick random key `i` from
  `storage`; if `next(i) == tail`-past (i.e. `next(i).is_null()`) use `i = prev(i)`; `j = next(i)`.
  `c1 = is_outgoing(i)?1:-1`, `c2 = is_outgoing(j)?1:-1`. If `link(i) == j` it's a no-op ⇒ `0.0`.
  `p' = p_out_i + c1*q_i - c2*q_j`. Reducibility guard: if `i` outgoing and `j` incoming and
  `link(i)!=j`, reject if `phonons_above_i==1 || phonons_above_j==1`.
  Acceptance = `exp(-(tau_j-tau_i)*(bare_dispersion(p') - bare_dispersion(p_out_i) - OMEGA*(c1-c2)))`.
  `accept`: `update_phonons_above` (port the two ±1 branches), set `p_out_i = p'`, and if `link(i)!=j`
  swap the arc connectivity exactly as C++ (`std::swap(q_i, q_j); std::swap(link_i, link_j);
  std::swap(link_i->link_, link_j->link_)`): swap the two vertices' `q` and `link` keys, then repair
  the partners' back-pointers so `arena[arena[k].link].link == k` holds for both. Verify against
  `sanity::check_sanity` afterwards.

Helpers needed in `diagram.rs`: `spherical_to_cartesian(r, theta, phi)`,
`theta_from_cartesian(v)`, `phi_from_cartesian(v)` — port from `utils.hpp`.

### 5.3 Add/remove phonon (the inverse pair)

Port `add_phonon.hpp` and `remove_phonon.hpp` faithfully, including the **order-0 "fake" sector**:

- `AddPhonon::attempt`: if `order==0` → `from_zero_fake_impl` (draw `q` via `draw_new_q`, return the
  order-0 ratio). Else `higher_order_impl`: return `0.0` if `order>=max_order`; pick a vertex `i` with
  `i+1 != n` (i.e. not the last — `std::next(vertex1_) != end()`); draw `tau1 ∈ (tau_i, tau_{i+1})`;
  reject if too close to neighbors (`DELTA_TAU_LIMIT`); draw `q`; draw `tau2` from a bounded exp with
  `lambda = OMEGA*(1+|q|/P0)^2` over `(tau1, max_tau)` (lambda>0 ⇒ `exponential_sample_bounded`);
  compute `p_mean = get_p_mean(tau1, tau2, i)`; apply the closeness checks; return the full ratio
  including `algo_ratio = (2*order-1)/order`.
- `draw_new_q` (`add_phonon.hpp`): `theta = acos(1 - 2r)`, `q = P0/r2 - P0`,
  `spherical_to_cartesian(q, theta, 2π·r3)`. Three uniforms.
- `AddPhonon::accept`: order-0 branch sets `q` on the two existing endpoint vertices and links them
  (set `head.link = tail`, `tail.link = head`), `order=1`. Higher-order branch calls `insert_arc`
  (splice two vertices via `splice_after`, set mutual `link`), updates `phonons_above` per the C++
  `++` logic, subtracts `q` from `p_out` on the spanned range (walk `next`), `order += 1`.
- `RemovePhonon::attempt`: if `order==min_order` → `0.0`; if `order==1` → `attempt_to_zero_fake_impl`;
  else `attempt_higher_order_impl` (pick vertex not the start and whose partner isn't the start;
  order the pair; **reducibility guard** using `phonons_above` exactly as C++; compute `delta_t`,
  `p_mean = get_p_mean(i,j,q)` on the *proposed* diagram; return ratio with `algo_ratio =
  (order-1)/(2*order-3)`).
- `RemovePhonon::accept`: order-1 branch restores the endpoints to bare (`p_out += q`, `q=0`, unlink),
  `order=0`. Else remove the two vertices, fix `p_out`/`phonons_above` on the spanned range, `order
  -= 1`.

**Detailed-balance check (do this!)**: After porting, run a *short* fixed-seed simulation and inspect
add/remove acceptance counts plus the order histogram. Use this as a qualitative guard: a sign error
in any ratio shows up as the order distribution collapsing to 0 or exploding to `max_order`.

**Phase 2 done when**: a 1e6-step single-chain run with all 8 updates completes with
`sanity::check_sanity` passing every step (debug build), no panics, and a non-degenerate order
histogram.

---

## 6. Measurement and estimators (Phase 3)

Port `measurements.hpp::histogram` as `PolaronMeasurement: Measurement<Diagram>`.

### 6.1 BatchedRatio accumulator (the missing framework piece)

Implement a serde-derivable accumulator that supports the correlated num/den jackknife the C++ gets
from `batch_acc` + `jackknife_propagate`:

```rust
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct BatchedSum {
    n_batches: usize,
    batch_sums: Vec<f64>, // len n_batches
    batch_counts: Vec<u64>,
    next: usize,          // round-robin cursor (global sample index % n_batches)
}
```
- `push(&mut self, x)`: `let b = self.next % n_batches; batch_sums[b]+=x; batch_counts[b]+=1;
  self.next += 1;`
- `Merge`: element-wise add `batch_sums` and `batch_counts` (require equal `n_batches`).
- For a multi-bin series (the histogram Σ has 2000 bins), use `Vec<BatchedSum>` (one per bin), all
  fed on the **same global sample schedule** as the scalar `zeroth` so batch `b` corresponds to the
  same samples across series. **Implementation rule**: drive all accumulators from a single
  per-measurement sample counter so numerator-bin `b` and denominator `zeroth` batch `b` are aligned.

Delete-one jackknife of a ratio functional `f(num, den)` over `NB` batches:
```
batch_mean_num[b] = batch_sums_num[b] / batch_counts[b]   // skip empty batches
batch_mean_den[b] = batch_sums_den[b] / batch_counts[b]
loo_num[b] = mean over b' != b of batch_mean_num[b']
loo_den[b] = mean over b' != b of batch_mean_den[b']
theta[b]   = f(loo_num[b], loo_den[b])
mean   = (1/NB) Σ theta[b]
stderr = sqrt((NB-1)/NB * Σ (theta[b]-mean)^2)
```
This reproduces `jackknife_propagate`. Cross-check on a toy case in a unit test (e.g. ratio of two
constant series gives that constant with ~0 stderr).

### 6.2 Accumulators and `measure`

Hold (mirroring C++):
- `zeroth: BatchedSum` (scalar) — 1 if `order==0` else 0.
- `exact: Vec<BatchedSum>` (per bin, `num_bins`) — `exact_estimator(d, t0)` at the bin index, else 0.
- `hist: Vec<BatchedSum>` (per bin) — diagnostic count (1.0 at the bin), optional for Σ but cheap.
- `energy: BatchedSum` (scalar) — `-exp((E_est - mu)*τ)` (0 at order 0).
- `a: BatchedSum` (scalar) — `τ*exp((E_est - mu)*τ)` (0 at order 0).
- grid: `rmc_grids::LinearGrid::new(0.0, max_tau, num_bins + 1)`. Bin index via `grid.bin_index(τ)`;
  bin center via `grid.bin_center(index)`.
- self-consistency fields: `energy_estimate` (init `-1.0168`), `energy_estimates: Vec<f64>`,
  `self_consistent_count`, `self_consistent_period` (init `1000`), `period_multiplier` (`1.5`),
  `self_consistent_periods: Vec<usize>`.

`measure(&mut self, d)` (port `histogram::measure`):
```
let idx = grid.bin_index(d.tau());        // guard None (shouldn't happen: 0 < τ < max_tau)
if d.order == 0 {
    zeroth.push(1.0); exact[idx].push(0.0); energy.push(0.0); a.push(0.0);
} else {
    let e = ((energy_estimate - d.mu) * d.tau()).exp();
    let t0 = grid.bin_center(idx).unwrap();
    zeroth.push(0.0);
    hist[idx].push(1.0);
    exact[idx].push(exact_estimator(d, t0));
    energy.push(-e);
    a.push(d.tau() * e);
}
self_consistent_count += 1;
if self_consistent_count > self_consistent_period {
    let new = d.bare_dispersion(d.momentum_out()) + self.jackknife_energy_mean(d);
    energy_estimates.push(new); energy_estimate = new;       // reevaluate_energy_estimate
    // reset energy & a accumulators (fresh BatchedSum of same shape)
    self_consistent_count = 0;
    self_consistent_period = (self_consistent_period as f64 * period_multiplier) as usize;
    self_consistent_periods.push(self_consistent_period);
}
```
Note `bare_dispersion(p=0)=0`, so `new = jackknife_energy_mean`. Keep the term for general `p`.

### 6.3 Output type and Merge

`PolaronMeasurement::Output = PolaronStats` holding **all batched accumulators + the grid + final
`energy_estimate`** (clone them in `finish`). `PolaronStats: Merge` adds the accumulators batch-wise
(and sums any sample counters). The final Σ/E/Z jackknife is computed **after** the parallel merge,
in `app.rs`, from the merged `PolaronStats`. This matches the C++ MPI flow (`mpi_collect` the
accumulators, then jackknife once on rank 0).

### 6.4 Final estimators (port `jackknife_selfenergy/energy/quasiparticle_weight`)

With `E = dispersion(p)`, `n0 = norm0(max_tau, E)`, `binsize = grid step`:
- **Σ(τ_bin)**: per bin, `f(exact, zeroth) = exact * n0 / (zeroth * binsize)`; jackknife with the
  per-bin `exact` numerator and scalar `zeroth` denominator. Output `{tau: bin_centers, mean[],
  stderr[]}`.
- **E**: `f(energy, zeroth) = energy * n0 / zeroth` (scalar). Mean is the polaron energy (≈ -1.013).
- **Z**: `f(a, zeroth) = 1/(1 + a*n0/zeroth)` (scalar) (≈ 0.59).

Also expose `get_exact()` (the C++ `get_exact`, Σ(τ) on bin centers used as FFT input):
`exact_mean[i]/binsize * n0 / zeroth_mean`.

**Phase 3 done when**: `tests/physics_validation.rs` (below) passes.

---

## 7. App wiring, parallel, checkpoint (Phase 4)

### 7.1 Config (`config.rs`)

`RunConfig` (serde) mirroring `runs/input.json`: `alpha, mu, momentum, max_tau, start_tau, min_order,
max_order, num_bins, energy_estimate, initial_self_consistent_period, period_multiplier, fft_beta,
seed, chains, max_steps, steps_per_cycle, cycles_per_check`. Provide `Default` equal to the reference
`runs/input.json` (alpha=1, mu=-1.1, momentum=0, max_tau=30, num_bins=2000, beta=100,
energy_estimate=-1.0168). `main.rs` reads a JSON path from argv (default `input.json`), or writes a
default config when invoked as `rmc-polaron def` (mirror `main.cpp::gen_default_config`).

### 7.2 build_chain and run

```rust
fn build_chain(cfg: &RunConfig) -> impl Fn(ChainId)
    -> (Diagram, MetropolisKernel<WeightedUpdateSet<PolaronUpdate>>, PolaronMeasurement)
```
Construct `Diagram` at order 0 from `cfg`; build the 8-entry `WeightedUpdateSet` (all weight 1.0,
ratio 1.0); build the measurement from `cfg`. Single-chain path: `run_typed`. Parallel path:
`run_parallel(ParallelConfig { chains: cfg.chains, seed: SeedSource::new(cfg.seed), params }, build)`,
then compute Σ/E/Z from the merged `PolaronStats`.

`SimulationParams { max_steps, steps_per_cycle: 5, cycles_per_check }` (C++ used 5 steps/cycle).
There is no separate warm-up entry point in the framework; emulate warm-up by either (a) running a
short throwaway `run_typed` and discarding its measurement, or (b) adding a `warmup_steps` field and
skipping measurement for the first cycles via a `RunCallbacks` flag. **Recommended**: do (a) — run
`warmup_steps` with a no-op measurement (a `Measurement<Diagram>` whose `measure` does nothing),
reusing the final `Diagram` as the starting state of the measured run. Note `run_typed` consumes and
returns `state`, so thread the returned `Diagram` into the measured run.

### 7.3 Callbacks: checkpoint + convergence

Implement `PolaronCallbacks: RunCallbacks<SimulationCtx>`:
- `on_checkpoint`: serialize a checkpoint payload `{ diagram, rng_seed_info, measurement_snapshot,
  steps_done }` via `rmc_io::save_payload_json` (atomic variant). Because `run_typed` owns the RNG
  and state, the simplest robust checkpoint is **at end of run** (write final state + accumulators);
  for mid-run checkpoints the callback only has `SimulationCtx` (counts), not `&State`. **Therefore**:
  do end-of-run checkpointing of `(final Diagram, PolaronStats)` in `app.rs`, and use `on_checkpoint`
  only for logging E/Z progress (port `checkpoint_supplier::checkpoint_callback`'s table output).
  Record as a framework finding: "callbacks lack `&State`/measurement access; mid-run checkpoint of
  full chain state isn't expressible." (The C++ `checkpoint_callback` *did* have access to the
  manager.)
- `stop_when`: optional convergence (port `convergence_callback`: stop when the running stddev of the
  last 1000 E estimates `< 1e-4`). Keep simple or omit behind a config flag.

### 7.4 Checkpoint round-trip test

`tests/` (or in `app.rs` under `#[cfg(test)]`): run N steps, write JSON checkpoint of
`(Diagram, PolaronStats)`, load it back, assert structural equality (taus, momenta, order, batch
sums). This exercises `rmc-io` + serde on our types. **Phase 4 done when** this passes and a parallel
run reduces to the same E (within error) as the single-chain run.

---

## 8. Fourier/Dyson stage (Phase 5)

Port `fourier_transform.hpp` to `fourier.rs` using `rustfft`.

- Inputs: the Σ(τ) on bin centers (`PolaronMeasurement::get_exact()`), the linear grid, `alpha`,
  `mu`, `momentum`, `beta` (= `fft_beta`).
- Pipeline (port the C++ `fourier_analysis` exactly):
  1. Build dense time grid `[min_time, beta]` with `num_frequencies = 2^16` points;
     `delta_t = (beta - min_time)/(N-1)`.
  2. Subtract the analytic first-order tail: `S' = Σ - S1` with `S1_t = alpha*sqrt(MASS)/(sqrt(π)*
     sqrt(τ)) * exp(-(OMEGA-mu)*τ)` (`initialize_S1_t`). Interpolate `S'` to the dense grid with a
     **cubic spline** — use `rmc_numeric::CubicSplineInterpolation::natural(grid, values)` then
     `.evaluate(x)` per dense point (extrapolate to 0 beyond `grid.last()`), replacing the C++
     `tk::spline`. Add back the analytic dense `S1_t`.
  3. Forward FFT (`rustfft` `FftPlanner::plan_fft_forward(N)`): apply to
     `(S_t - S1_t)*delta_t`, add analytic `S1_w = alpha/sqrt(1 - mu + iω)` (`initialize_S1_w`).
     Frequencies via `get_omega(j, delta_t, N)` (port the index-folding formula exactly).
  4. Dyson: `G0_w = 1/(dispersion(p) + iω)`; `G_w = 1/(1/G0_w - S_w)`.
  5. Inverse FFT of `(G_w - G0_w)/beta`, add `G0_t = bare_propagator(p, τ)` (`initialize_G0_t`).
     `rustfft` inverse is **unnormalized**; the C++ matches FFTW (also unnormalized) and divides by
     `beta` on input — keep the exact same normalization as C++ (do **not** add an extra `1/N`).
     Verify against the C++ `runs/fft.dat` / `fft_dense.dat` if numerically reproducible.
  6. Coarse-grain `G_t`, `Re/Im G_w` back to the bin grid (cubic spline), assemble the output JSON
     identical in shape to C++ (`{tau, G0_t, S_higher_t, G_w:{real,imag}, G_t}`).

`num_complex::Complex<f64>` interops with `rustfft::num_complex` (same crate). Write `out_fft.json`.

**Phase 5 done when**: the FFT runs without NaNs on a real Σ(τ) from a measured run, `G_t(0+) ≈ 1`,
and `G_t` decays like `exp(-E·τ)` with the measured `E` at large τ (sanity assertions in a test).
If exact bit-parity with the C++ FFTW output is required, compare to `self-energy-dmc/runs/fft.dat`.

---

## 9. Validation (the acceptance gate) — `tests/physics_validation.rs`

This is the success criterion. Use a fixed seed and enough steps for tight error bars.

1. **E and Z**: run `run_parallel` with `chains = num_cpus` (or 8), `max_steps` per chain ≈ 2e7
   (tune so the test runs in < ~60 s in release; mark `#[ignore]` a longer high-precision variant).
   Compute E and Z from the merged stats. Assert:
   - `E_mean` within `max(3*E_stderr, 0.01)` of **-1.013** (Fröhlich α=1, p=0 reference).
   - `Z_mean` within `max(3*Z_stderr, 0.02)` of **≈0.59** (looser; Z is noisier).
   - `E_stderr` is finite and small (`< 0.02`), `zeroth_mean > 0` (normalization sector visited).
   Document the reference values and tolerance rationale in the test.
2. **Order distribution**: record the order histogram and guard only against clearly degenerate
   behavior, such as persistent collapse to order 0 or runaway growth toward `max_order`.
3. **Sanity invariants**: a debug-build run asserts `sanity::check_sanity` every step over a short
   chain (already exercised in Phase 2; keep a small instance here).
4. **Σ(τ) shape**: assert `Σ(τ)` is finite, and `Σ(τ→0)` is positive / monotone-ish (loose).

Run all of: `cargo test -p rmc-polaron` (release recommended for the physics test:
`cargo test -p rmc-polaron --release`). Also run `cargo clippy -p rmc-polaron -- -D warnings` and
`cargo fmt`.

---

## 10. Execution order (milestones) and checklists

1. **Phase 0 – scaffold**: crate + workspace wiring compiles. ✔ when `cargo build -p rmc-polaron`.
2. **Phase 1 – diagram (slotmap migration)**: arena + intrusive list + key-storage; O(1) mutators;
   physics methods; sanity; factory. Migrate the existing positional-index code per §4.4.
   ✔ `tests/diagram_ops.rs` (add/remove round-trip, sanity, `exact_estimator` on a known diagram)
   and a criterion bench confirming O(1) (not O(n²)) per structural update.
3. **Phase 2 – updates**: enum + 8 updates; `draw_from_exp` parity test; short run with per-step
   sanity. ✔ 1e6-step debug run, no panics, sane order histogram.
4. **Phase 3 – measurement**: `BatchedRatio`/jackknife, estimators, self-consistent E. ✔ ratio
   jackknife unit test + a single-chain E estimate in the right ballpark.
5. **Phase 4 – app/parallel/checkpoint**: config, build_chain, warm-up, `run_parallel`, checkpoint
   round-trip. ✔ checkpoint test + parallel≈single E.
6. **Phase 5 – FFT/Dyson**: `fourier.rs`, `out_fft.json`. ✔ FFT sanity test.
7. **Validation**: `tests/physics_validation.rs` green; clippy/fmt clean.

After each phase, run `cargo test -p rmc-polaron` and fix before proceeding. Do **not** start a later
phase before the earlier phase's ✔ holds — the diagram/update layers are where bugs hide and they are
cheap to localize with the phase tests.

---

## 11. Risks, gotchas, and framework findings to report

- **Key/link translation** (§4.2) is the #1 bug source. Use the `VKey::null()` end-sentinel and
  keep `sanity::check_sanity` on in debug; it catches mis-wired `prev`/`next`/`link`/`storage_idx`.
- **Performance parity is a goal**, so the diagram is a `slotmap` arena + intrusive doubly-linked
  list + key-storage vector for O(1) updates (§4), matching the C++ `std::list` + `vertex_storage_`.
  *Framework finding*: this is application-owned state — confirm nothing in `rmc-core` forces a
  representation that defeats O(1) updates (it does not: state is fully caller-owned and threaded by
  value, which is exactly what makes a slotmap arena usable here).
- **`draw_from_exp` sign convention** (§2): audit every call's `lambda` sign; use
  `safe_exponential_sample` for the possibly-negative ones (`change_tau`, `change_internal_tau`) and
  `exponential_sample_bounded` for the strictly-positive ones (`add_phonon` tau2).
- **Attempt must not mutate** the diagram (the framework can reject after `attempt`). Verified true of
  all C++ `attempt_impl`s; preserve it. Default `reject` no-op is then correct.
- **Acceptance semantics**: framework `attempt` returns probability with `>=1`⇒certain, `<0`⇒
  impossible. Return the raw C++ ratio; this is already a Metropolis ratio. Don't pre-clamp.
- **Gaussian sampler missing**: add `rand_distr::Normal` (§2).
- **No ratio jackknife in `rmc-stats`**: implemented locally (§6.1) — *finding*: upstream a paired
  ratio/delete-one jackknife.
- **Callbacks lack `&State`/measurement access** (§7.3): mid-run full-state checkpointing isn't
  expressible; we checkpoint at end of run — *finding*: `RunCallbacks` may need state access or a
  measurement handle for parity with the C++ checkpoint design.
- **No warm-up phase concept**: emulated via a discarded `run_typed` (§7.2) — *finding*: a first-class
  warm-up / measurement-gating mechanism would help.
- **FFT normalization**: `rustfft` inverse is unnormalized like FFTW; match the C++ `/beta` exactly,
  add no extra `1/N` (§8).
- **Determinism**: framework RNG (Xoshiro256++) differs from C++; do **not** expect bit-identical
  trajectories. Validate physics within error bars, not stream equality.

---

## 12. Quick reference — reproduction targets

- Reference run params: `self-energy-dmc/runs/input.json` (α=1, μ=-1.1, p=0, max_tau=30,
  num_bins=2000, beta=100, energy_estimate=-1.0168).
- Physics targets (α=1, p=0): **E ≈ -1.013**, **Z ≈ 0.59**.
- Reference C++ outputs to diff against if desired: `runs/run_properties.json`, `runs/fft.dat`,
  `runs/fft_dense.dat`, `runs/results/`, `runs/thread_0`.
