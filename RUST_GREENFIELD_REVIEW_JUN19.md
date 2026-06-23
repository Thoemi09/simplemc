  Verdict

  This is a solid, genuinely idiomatic Phase 2–3 prototype. It builds clean, 41 tests pass, clippy --all-targets -D warnings is clean, both examples run, and the Ising example at β=0.44 produces physically
  plausible magnetization (~0.71 |m| near criticality). The core greenfield ideas from the plan are actually realized: Merge-based reduction, per-chain SeedSource seeding, no Any/downcast (typed results via
  finish()), static and dyn update-set paths, Result-based errors. Good work.

  But two of the headline design goals are only partially met, and there's one real hot-path bug.

  What's done well

  - Deterministic reduction order. into_par_iter().map().collect() preserves chain order, so the left-fold merge in parallel.rs:103-110 is order-deterministic — reproducibility across thread counts doesn't
  even depend on Merge being commutative. That's a stronger guarantee than the plan asked for. (Note for Phase 8: the MPI backend must preserve this ordering to keep it.)
  - Inverse-pair detailed-balance ratios (sets.rs:177-205) faithfully reproduce the C++ semantics, with proper validation.
  - Clean error handling, sensible validation paths, the Ising example avoids Arc<Mutex> by owning chain state.

  Findings, by severity

  High

  1. ✅ FIXED — DynUpdateSet reallocates a Vec on every MC step. select_and_step calls self.refresh_stats() (sets.rs:255), and refresh_stats does self.stats = …collect() (sets.rs:304-315) — a heap allocation + full
  iteration over all updates per step. The per-update counters already live on UpdateEntry; the parallel stats vec is redundant. The static paths update stats[idx] in place (step_update, sets.rs:563-577) and
  don't do this. Fix is to update self.stats[idx] incrementally and rebuild only on add. This directly undercuts the dyn path's performance — the thing the design wanted to be "the slow-but-flexible path,"
  not "the allocates-every-step path."
     Resolution: select_and_step now writes self.stats[idx] in place from the stepped entry's counters (no allocation, no full iteration). refresh_stats() is retained only for add() and the bulk
     reset_run_counters()/accumulate_counters() paths (also resynced so stats() stays consistent). Per-step heap allocation eliminated.

  2. ✅ FIXED — The static path still pays dynamic dispatch on every RNG draw. Update::attempt(&mut self, rng: &mut dyn RngCore) (traits.rs:20) takes dyn RngCore everywhere, including SingleUpdateSet/TwoUpdateSet. So
  while update dispatch is monomorphized, every rng.next_u64() inside an update is a non-inlinable vtable call. For cheap updates (Ising spin flip = one draw + a few ops), that indirection is now the dominant
  per-step cost — which is exactly the overhead the greenfield design cited as its advantage over the port. It's dyn presumably to keep Update object-safe for the boxed path. Worth resolving: a generic
  attempt<R: Rng> for the static trait (with the dyn wrapper bridging separately), so the static hot path actually inlines the RNG. I'd benchmark this before committing, but it's the single most important
  design call here.
     Resolution: `Update<State>::attempt` is now generic — `fn attempt<R: Rng + ?Sized>(&mut self, state: &mut State, rng: &mut R) -> f64` (traits.rs). On the static `SingleUpdateSet`/`TwoUpdateSet` paths the
     concrete RNG is monomorphized, so `rng.next_u64()`/`rng.gen()` inline. This makes `Update` object-unsafe, so the boxed `DynUpdateSet` path bridges through the object-safe `UpdateObject` trait (wrappers.rs), which instantiates
     attempt with R = dyn RngCore (valid via rand's blanket impl<R: RngCore + ?Sized> Rng for R). DynUpdate/UpdateEntry no longer implement Update; they expose inherent attempt/accept/reject forwarders for the
     boxed path (called only with concrete sized RNGs). The Ising example's uniform_index helper was also made generic so the draw inlines. Criterion benchmark added: short local run measured roughly 21.7 us for
     `static_single_update_10k_steps` and 210 us for `dyn_single_update_10k_steps`.

  Medium

  3. ✅ FIXED — Stateless/stateful trait duplication. Update/StatefulUpdate, Measurement/StatefulMeasurement, Kernel/StatefulKernel, three …UpdateSet traits, and run/run_typed/run_stateful_typed each ×_with_callbacks
  ×parallel. The stateful family is clearly the better design (Ising uses it, no shared-mutable-state); the stateless family pushes users straight into Arc<AtomicI64> (see random_walk.rs:11-36) — the pattern
  greenfield set out to avoid. Stateless is just stateful with State = (). Collapsing the families would remove a lot of surface and the run-loop is copy-pasted verbatim 3× (run.rs:90-120, 155-185, 222-252).
     Resolution: the API is now state-generic. `Update<State>`, `Measurement<State>`, `Kernel<State, R>`, and `SteppingUpdateSet<State, R>` are the only core traits; stateless callers use `State = ()`. `run_typed`
     returns `(State, SimulationStats, Output)`, `run_parallel` builds `(state, kernel, measurement)` triples, and the run loop is shared through one internal `drive` function. The random-walk and Ising examples
     both use owned chain state.

  4. ⚠️ DELIBERATE FOR NOW — The dyn measurement path can't return results. DynMeasurement/MeasurementEntry are hardwired to Output = () (wrappers.rs:46-54, entries.rs:55-63), so DynMeasurementSet collects nothing — results only
  escape via side effects/shared state. The typed path returns M::Output cleanly; the dyn path has no result-extraction story. Worth a deliberate design answer (e.g. a result sink) rather than leaving it
  implicit.
     Current answer: the boxed dyn measurement path is explicitly documented as side-effecting/runtime-flexible only. Typed results are returned through `run_typed`. A result-sink design is still possible later,
     but no downcast-based retrieval was reintroduced.

  5. ✅ DOCUMENTED — max_steps allows a partial final cycle. The inner loop breaks mid-cycle (run.rs:91-103) and then still calls measure/on_cycle, so a measurement can be taken after a short or zero-step cycle when
  max_steps isn't a multiple of steps_per_cycle. The C++ guaranteed step counts were a multiple of cycles_per_check × steps_per_cycle. Either align or document.
     Resolution: `SimulationParams::validate` now documents partial final cycles and tells callers how to choose aligned `max_steps` / `cycles_per_check` values. We kept partial final cycles as allowed
     greenfield behavior.

  6. 🟡 PARTIALLY FIXED — Validation is lighter than the plan claims. seed_source_separates_chain_streams only asserts != for two chains (random_seed_source.rs:16-26) — far weaker than the "statistically independent streams"
  property the plan's risk register flagged as the linchpin. And there's no analytical/physics validation test yet (e.g. Ising T_c, or sampler moments) — the "validate against analytical truth" pillar is only
  represented by closed-form helper checks in random_samples.rs. The construction (splitmix64-seeded xoshiro) is sound, but the evidence is thin.
     Added: seed diffusion validation over 512 chain seeds (uniqueness + adjacent hamming-distance range), deterministic transform-sampler midpoint moment checks, fixed-seed `uniform_index` bucket-balance check,
     and a deterministic 2x2 Ising infinite-temperature test against exact magnetization enumeration. Still TODO: broader property tests and a stronger finite-size Ising validation suite.

  Low

  - ✅ DOCUMENTED — Merge for f64 = addition (merge.rs:27) is a footgun: correct for merging sums (the examples divide at the end), wrong if someone merges per-chain means. Consider dropping the bare-scalar impl or
  documenting the "merge sums, not means" contract.
  - ✅ DOCUMENTED — Acceptance divergence (sets.rs:246, :569): u < p with a p >= 1.0 short-circuit vs C++ p >= u always drawing. Fine for greenfield, but RNG consumption now depends on the acceptance branch — worth a doc
  note since it affects stream reasoning.
  - ✅ FIXED — Modulo bias in uniform_index (ising_2d.rs:118-120) — negligible at 256/2⁶⁴, but the library should eventually expose an unbiased index sampler so examples don't model the biased pattern.
    Resolution: `rmc_core::random::uniform_index` now delegates to `Rng::gen_range`, and examples/tests use it.
  - 🟡 PARTIALLY FIXED — Sparse rustdoc on public items; SimulationCtx is materialized every step and relies on inlining/DCE to be free under NoopCallbacks.
    Added crate/module docs plus docs for seeding, run params/stats/context, run entry points, parallel config, kernels, and update sets. More exhaustive rustdoc/examples can wait until the API settles further.
  - ✅ FIRST PASS — Static multi-update ergonomics beyond one/two updates.
    Added `WeightedUpdateSet<U>` for arbitrary counts of one concrete update type. Heterogeneous simulations can use an enum as `U`, preserving monomorphized storage and avoiding boxed `dyn Update`. A macro-generated concrete-field set remains a possible future optimization if profiling justifies it.

  Against the plan

  On track for Phases 0–3. The parallel core ships and is reproducible. Missing vs the stated DoD: stronger RNG-independence + analytical/physics tests, rustdoc, and the stats/grids/numeric subsystems
  (correctly deferred). The two things I'd fix before building on top of this are #1 (hot-path allocation) and a decision on #2 (RNG dispatch), because both touch the per-step path that everything else will
  sit on — and #2 is the design's whole performance thesis.
