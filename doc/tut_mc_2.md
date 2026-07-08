@page tut_mc_2 MC Tutorial 2: Checkpointing and input-config files

[TOC]

In this tutorial, we extend the minimal MC integration from @ref tut_mc_1 with production-style
file I/O:

- an **input-config file** that is bootstrapped with default values on the first run and lets the
  user configure the simulation — including the integration domain \f$ [a, b] \f$ — without
  recompiling,
- **checkpoint files** written during and after the simulation, and
- an optional **continuation**: a `load_checkpoint` flag in the input-config file restores the
  previous run and keeps going.

The simulation itself (uniform-resampling update, integral measurement, simplemc::metropolis_kernel,
free simplemc::run) is unchanged, so this page only walks through the differences to @ref tut_mc_1.

@section tut_mc_2_channels The two file channels

The @ref simplemc-serialize sublibrary distinguishes two serialization channels, and the mc types
implement both:

- The **persistent-state channel** (`simplemc_save` / `simplemc_load`) captures everything needed
  to continue a run: the RNG state, the update counters, the measurement accumulators, and the
  cumulative simulation statistics. Loads are *strict*: a missing key throws a
  simplemc::simplemc_exception, because an incomplete checkpoint is corrupt.
- The **input-config channel** (`simplemc_save_input_config` / `simplemc_load_input_config`)
  covers only the user-facing knobs (e.g. the fields of simplemc::simulation_params). Loads are
  *tolerant*: missing keys leave the destination untouched, so a partial config file is valid
  input.

Both channels write through the same serializer API (here the JSON backend via
`<simplemc/serialize/json.hpp>`), and the documents stay *open*: extra keys are just more
`save_at` / `try_load_at` calls next to the standard ones. Also note that the save hooks take the
serializer handle **by value** — serializers are cheap handles into a shared document — so rvalue
sub-handles like `s["params"]` bind directly.

@section tut_mc_2_workflow The workflow

The tutorial program implements the classic three-invocation workflow of a production MC code:

1. **Bootstrap**: no input-config file exists yet → write one with the default parameters and
   exit, so the user can inspect and edit it.
2. **Fresh run**: the input-config file exists → read it, run the simulation, and write a
   checkpoint every `checkpoint_after_steps` steps as well as at the very end.
3. **Continuation**: the user sets `"load_checkpoint": true` in the config file → the program
   restores the previous run from the checkpoint and continues it.

@section tut_mc_2_details Step-by-step guide

@subsection tut_mc_2_serializable Making the user types serializable

A type opts into a channel by providing the corresponding ADL hooks (here as `friend` functions).
The MC configuration provides both pairs — but note *what* each channel covers:

```cpp
// MC configuration shared between the update and the measurement:
// - x: the current sample (chain state, checkpointed),
// - [a, b]: the integration domain (set via the input-config file; also checkpointed, since it
//   must not change between a run and its continuation -- a loaded checkpoint overrides the
//   config file).
struct mc_config {
    double x = 1.0;
    double a = 0.0;
    double b = 2.0;

    // checkpoint channel: the full snapshot (strict load)
    template <simplemc::serializer S>
    friend void simplemc_save(S s, const mc_config& c) {
        s.save_at("x", c.x);
        s.save_at("a", c.a);
        s.save_at("b", c.b);
    }
    template <simplemc::serializer S>
    friend void simplemc_load(const S& s, mc_config& c) {
        s.load_at("x", c.x);
        s.load_at("a", c.a);
        s.load_at("b", c.b);
    }

    // input-config channel: only the domain is user-configurable (tolerant load)
    template <simplemc::serializer S>
    friend void simplemc_save_input_config(S s, const mc_config& c) {
        s.save_at("a", c.a);
        s.save_at("b", c.b);
    }
    template <simplemc::serializer S>
    friend void simplemc_load_input_config(const S& s, mc_config& c) {
        s.try_load_at("a", c.a);
        s.try_load_at("b", c.b);
    }
};
```

Compared to @ref tut_mc_1, the domain \f$ [a, b] \f$ consists of runtime members instead of
compile-time constants, so the user can change the integral's domain in the input-config file. The
current sample \f$ x \f$ is pure chain *state* and therefore only appears in the checkpoint
channel. The domain appears in **both**: it is user *configuration*, but a continued run must keep
integrating over the same domain, so the checkpoint stores it too and a load overrides whatever
the config file says.

The measurement checkpoints its accumulator, so a restored measurement resumes accumulating where
the previous run stopped:

```cpp
// Integral measurement (as in tut_mc_1), now with checkpoint hooks for its accumulator so a
// restored measurement resumes accumulating.
struct integral {
    const mc_config* cfg;
    simplemc::mean_acc<double> acc {};

    void measure() { acc << f(cfg->x); }

    template <simplemc::serializer S>
    friend void simplemc_save(S s, const integral& m) {
        s.save_at("acc", m.acc);
    }
    template <simplemc::serializer S>
    friend void simplemc_load(const S& s, integral& m) {
        s.load_at("acc", m.acc);
    }
};
```

The update `uniform_update` is byte-for-byte the one from @ref tut_mc_1 — it has no persistent
state of its own (its counters live in the simplemc::update wrapper, which the library serializes
for us).

@subsection tut_mc_2_helpers The file helpers

The library ships file-level pairs for both channels (`mc/json.hpp`, part of the `mc.hpp`
umbrella). simplemc::save_json_checkpoint / simplemc::load_json_checkpoint write and read the
standard components under `"rng"` / `"stats"` / `"updates"` / `"measurements"`; the overloads
taking a config component additionally handle it under `"config"`, so the key cannot get out of
sync between the two sides. Every save goes through simplemc::atomic_file_write (sibling
`.tmp.<pid>` file + rename), so a failed or interrupted write can never corrupt the previous
checkpoint.

simplemc::save_json_input_config / simplemc::load_json_input_config are the analogous
input-config pair: pretty-printed writes, tolerant reads, the same config overloads routed through
the type's input-config channel. Free-form sibling keys go through the optional extra-keys hook —
a callback receiving the root serializer handle.

For fully custom documents, the underlying open-document helpers simplemc::save_json_file /
simplemc::load_json_file hand the callback an empty (resp. freshly read) root handle and take care
of the serializer, atomicity, and file I/O. For HDF5 checkpoints, `mc/hdf5.hpp` provides
`save_hdf5_checkpoint` / `load_hdf5_checkpoint` and `save_hdf5_file` / `load_hdf5_file` with the
same shape (include it directly under `#ifdef SIMPLEMC_USE_HDF5`).

@subsection tut_mc_2_main Bootstrap, read, and optional continuation

`main` assembles the same components as in @ref tut_mc_1 plus a simplemc::simulation_stats that
tracks the cumulative progress across runs. The three-invocation workflow is then a straight
transcription of the steps above:

```cpp
// 1) bootstrap: without an input-config file, write one with default parameters and exit; the
// cfg overload stores the MC configuration under "config", the hook adds "load_checkpoint"
if (!std::filesystem::exists(config_path)) {
    const simplemc::simulation_params defaults { .max_steps = 1'000'000,
        .max_time = 60.0,
        .steps_per_cycle = 1,
        .cycles_per_check = 10'000,
        .checkpoint_after_steps = 250'000 };
    simplemc::save_json_input_config(config_path, defaults, us, ms, cfg,
        [](simplemc::json_serializer& s) { s.save_at("load_checkpoint", false); });
    fmt::println("wrote default input config to {} -- edit it and rerun", config_path.string());
    return 0;
}

// 2) read the input-config file (tolerant: missing keys leave the defaults untouched)
simplemc::simulation_params params;
bool load_checkpoint = false;
simplemc::load_json_input_config(config_path, params, us, ms, cfg,
    [&](const simplemc::json_serializer& s) { s.try_load_at("load_checkpoint", load_checkpoint); });

// 3) optionally load the previous run (strict: overwrites the RNG state, the counters, the
// accumulator and the MC configuration)
if (load_checkpoint && std::filesystem::exists(checkpoint_path)) {
    simplemc::load_json_checkpoint(checkpoint_path, rng, us, ms, stats, cfg);
    fmt::println("loaded checkpoint at cumulative step {}", stats.cumulative_steps);
}
```

Note the new simplemc::simulation_params field compared to @ref tut_mc_1 —
`checkpoint_after_steps = 250'000` asks the driver to fire a checkpoint callback every 250'000
steps.

@subsection tut_mc_2_running Running with checkpoints

@ref tut_mc_1 called simplemc::run without callbacks. Here we pass a simplemc::run_callbacks whose
`on_checkpoint` hook writes a checkpoint; the driver invokes it every `checkpoint_after_steps`
steps (and simplemc::run_callbacks compiles down to no-ops for the hooks we leave out):

```cpp
// 4) run with periodic checkpoints: the update counters are cumulative and always
// checkpoint-ready; the simulation stats are checkpointed as the non-mutating totals stats + c
simplemc::metropolis_kernel kernel { us };
const auto cbs = simplemc::run_callbacks {
    .on_checkpoint =
        [&](const simplemc::simulation_ctx& c) {
            simplemc::save_json_checkpoint(checkpoint_path, rng, us, ms, stats + c, cfg);
            fmt::println("wrote checkpoint at step {}", c.steps_done);
        },
};
stats += simplemc::run(rng, kernel, ms, params, cbs);

// final checkpoint with the finished run folded in
simplemc::save_json_checkpoint(checkpoint_path, rng, us, ms, stats, cfg);
```

The update counters need no bookkeeping at all: they are *cumulative* — simplemc::update bumps
simplemc::update_stats::nprops / naccs / nimps directly and the checkpoint saves and restores them
as they stand, so a checkpoint written at any moment carries the correct all-time proposal and
acceptance statistics. (To exclude a warm-up run, zero them once with
update_set::reset_counters() before the measurement run.)

The simulation statistics follow a two-rule convention built on the stats/ctx algebra
(simplemc::simulation_ctx is the per-run delta to the cumulative simplemc::simulation_stats):

- **during** a run, form totals without mutating anything — `stats + c` at every save site (the
  driver refreshes simulation_ctx::runtime just before `on_checkpoint` fires, so the sum is
  correct at any checkpoint);
- **after** a run, fold it in exactly once — `stats += simplemc::run(...)` — so the next run
  starts from the updated cumulative record.

@subsection tut_mc_2_output Output

The first invocation bootstraps the input-config file and exits:

```
wrote default input config to tut_mc_2_input_config.json -- edit it and rerun
```

The generated document contains the standard `"params"` / `"updates"` / `"measurements"` blocks,
our `"config"` block with the domain, and the `"load_checkpoint"` flag:

```json
{
    "config": {
        "a": 0.0,
        "b": 2.0
    },
    "load_checkpoint": false,
    "measurements": {
        "integral": {
            "is_active": true
        }
    },
    "params": {
        "checkpoint_after_steps": 250000,
        "checkpoint_after_time": 1.7976931348623157e+308,
        "cycles_per_check": 10000,
        "max_steps": 1000000,
        "max_time": 60.0,
        "skip_measurements": false,
        "steps_per_cycle": 1
    },
    "updates": {
        "uniform": {
            "weight": 1.0
        }
    }
}
```

The second invocation runs the simulation, checkpointing every 250'000 steps:

```
wrote checkpoint at step 250000
wrote checkpoint at step 500000
wrote checkpoint at step 750000
wrote checkpoint at step 1000000
cumulative steps: 1000000
result:           0.9078870389280093
exact:            0.9092974268256817
```

The result is identical to @ref tut_mc_1 — same seed, same chain. Now we set
`"load_checkpoint": true` in the input-config file and run a third time:

```
loaded checkpoint at cumulative step 1000000
wrote checkpoint at step 250000
wrote checkpoint at step 500000
wrote checkpoint at step 750000
wrote checkpoint at step 1000000
cumulative steps: 2000000
result:           0.9090328272109088
exact:            0.9092974268256817
```

The run picks up exactly where the previous one stopped: the cumulative step count doubles to
\f$ 2 \times 10^6 \f$, the restored accumulator keeps counting, and the estimate moves closer to
the exact value, as expected for a doubled sample size. Changing `"b"` in the `"config"` block
changes the computed integral on the next *fresh* run (try `"b": 1.0` and compare against
\f$ \sin(1) \approx 0.8415 \f$) — but not on a continuation, where the checkpointed domain wins.

@section tut_mc_2_code Full code

@include tutorials/tut_mc_2.cpp
