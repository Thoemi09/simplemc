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

The @ref simplemc-serialize and @ref simplemc-mc sublibraries distinguish two serialization channels:

- The **checkpoint channel** (`simplemc_save` / `simplemc_load`) captures everything needed to
continue a run: the RNG state, the update counters, the measurement accumulators, and the cumulative
simulation statistics. Loading is *strict*: a missing key throws a simplemc::simplemc_exception,
because an incomplete checkpoint is corrupt.
- The **input-config channel** (`simplemc_save_input_config` / `simplemc_load_input_config`) covers
only the user-facing control parameters. Loading is *tolerant*: missing keys are ignored, so a partial
config file is valid input.

Both channels write through the same serializer API (here the JSON backend via
@ref simplemc-serialize-json).

Also note that the save hooks take the serializer handle **by value** — serializers are cheap handles
into a shared document.

@section tut_mc_2_workflow The workflow

This tutorial implements a classic three-invocation workflow of a production MC code:

1. **Bootstrap**: no input-config file exists yet \f$ \rightarrow \f$ write one with the default
parameters and exit, so the user can inspect and edit it.
2. **Fresh run**: the input-config file exists \f$ \rightarrow \f$ read it, run the simulation, and
write a checkpoint every `checkpoint_after_steps` steps as well as at the very end of the simulation.
3. **Continuation**: if the user sets `"load_checkpoint": true` in the config file \f$ \rightarrow \f$
the program restores the previous run from the checkpoint and continues it.

Let's take a closer look at the new pieces of code that implement this workflow.

@section tut_mc_2_details Step-by-step guide

The overall structure of the code is similar to @ref tut_mc_1, except for some additional includes:

```cpp
#include <fmt/base.h>
#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/mc.hpp>
#include <simplemc/random/xoshiro256.hpp>

#include <cmath>
#include <filesystem>
#include <random>

namespace {
    // user types go here
} // namespace

int main() {
    // code snippets go here
}
```

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

    // Checkpoint channel: all fields are saved/loaded.
    template <simplemc::serializer S>
    friend void simplemc_save(S s, const mc_config& cfg) {
        s.save_at("x", cfg.x);
        s.save_at("a", cfg.a);
        s.save_at("b", cfg.b);
    }
    template <simplemc::serializer S>
    friend void simplemc_load(const S& s, mc_config& cfg) {
        s.load_at("x", cfg.x);
        s.load_at("a", cfg.a);
        s.load_at("b", cfg.b);
    }

    // Input-config channel: only the domain is user-configurable.
    template <simplemc::serializer S>
    friend void simplemc_save_input_config(S s, const mc_config& cfg) {
        s.save_at("a", cfg.a);
        s.save_at("b", cfg.b);
    }
    template <simplemc::serializer S>
    friend void simplemc_load_input_config(const S& s, mc_config& cfg) {
        s.try_load_at("a", cfg.a);
        s.try_load_at("b", cfg.b);
    }
};
```

Compared to @ref tut_mc_1, the domain \f$ [a, b] \f$ consists of runtime members instead of
compile-time constants, so the user can change the integral's domain in the input-config file. 

The current sample \f$ x \f$ is the actual *state* of our fictitious Markov chain and therefore only 
appears in the checkpoint channel. 

The domain appears in **both**: it is user *configuration*, but a continued run must keep integrating 
over the same domain, so the checkpoint stores it too, and loading a checkpoint overrides whatever
the config file says.

The measurement checkpoints its accumulator, so a restored measurement resumes accumulating where
the previous run stopped:

```cpp
// Integral measurement: I = \int_a^b f(x) dx
// - acc: accumulator for the sample mean of f(x), checkpointed so a restored measurement resumes
//   accumulating.
// - measure(): evaluates f(x) and accumulates the result.
struct integral {
    const mc_config* cfg;
    simplemc::mean_acc<double> acc {};

    void measure() { acc << f(cfg->x); }

    // Checkpoint channel: only the accumulator is part of the checkpoint.
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

@subsection tut_mc_2_assembly Assembling the components

`main` starts just like in @ref tut_mc_1 — the same components, assembled in the same order — plus 
the cumulative simplemc::simulation_stats:

```cpp
// random number generator
simplemc::xoshiro256ss rng;

// MC configuration
mc_config cfg;

// construct the update set with our uniform update
simplemc::update_set us { simplemc::update { uniform_update { .cfg = &cfg, .rng = &rng }, "uniform", 1.0 } };

// construct the measurement set with our integral measurement
simplemc::measurement_set ms { simplemc::measurement { integral { .cfg = &cfg }, "integral" } };

// construct the Metropolis kernel and give it access to the update set
simplemc::metropolis_kernel kernel { us };

// set the default simulation parameters, now with periodic checkpointing
simplemc::simulation_params params { .max_steps = 1'000'000,
    .max_time = 60.0,
    .steps_per_cycle = 1,
    .cycles_per_check = 10'000,
    .checkpoint_after_steps = 250'000 };

// cumulative simulation statistics across runs
simplemc::simulation_stats stats;
```

Compared to @ref tut_mc_1, two things are new:

- The parameters additionally set simplemc::simulation_params::checkpoint_after_steps, which asks
the driver to fire a checkpoint callback every 250'000 steps. These parameters are only *defaults*: 
they seed the bootstrapped input-config file, which the user can edit to change the behavior of the
simulation.
- The simplemc::simulation_stats now tracks the cumulative progress *across* runs in case simulations
are continued from a checkpoint.

@subsection tut_mc_2_main Bootstrap, read, and optional continuation

We then define the file paths of the **input-config** and **checkpoint** files:

```cpp
// paths of the input-config and checkpoint files
const std::filesystem::path config_path { "input_config_tut_mc_2.json" };
const std::filesystem::path checkpoint_path { "checkpoint_tut_mc_2.json" };
```

Before we continue, let us note that the library ships file I/O helper functions for the input-config
and the checkpoint channels:

- simplemc::save_json_checkpoint / simplemc::load_json_checkpoint write and read the standard 
components under `"rng"` / `"stats"` / `"updates"` / `"measurements"` and, optionally, an MC 
configuration under `"config"`. Every save goes through simplemc::atomic_file_write, so a failed or 
interrupted write can never corrupt the previous checkpoint.
- simplemc::save_json_input_config / simplemc::load_json_input_config are the analogous input-config 
pair. They do pretty-printed writes and tolerant reads, i.e. a missing key is simply ignored.

Next we implement the **bootstrapping**: check if the input-config file exists, and if not, write one 
with the default options and exit:

```cpp
// if there is no input-config file, write one with the default parameters and exit
if (!std::filesystem::exists(config_path)) {
    fmt::print("No input config file found");
    simplemc::save_json_input_config(config_path, params, us, ms, cfg,
        [](simplemc::json_serializer& s) { s.save_at("load_checkpoint", false); });
    fmt::println(" --> generated {}.", config_path.string());
    fmt::println("Edit the file and re-run to start a simulation.");
    return 0;
}
```

Note that we use simplemc::save_json_input_config to write the simulation parameters, the update set, 
the measurement set and the MC configuration. Additionally, we use its optional hook to write the 
boolean `"load_checkpoint"` flag that indicates if a simulation should continue from a previous 
checkpoint.

On the other hand, if the input-config file exists, we read it using simplemc::load_json_input_config:

```cpp
// if there is an input-config file, read it to configure the simulation accordingly
fmt::println("Reading input config from {}.", config_path.string());
bool load_checkpoint = false;
simplemc::load_json_input_config(config_path, params, us, ms, cfg,
    [&](const simplemc::json_serializer& s) { s.try_load_at("load_checkpoint", load_checkpoint); });
```

After reading the input-config file, we check if the user has set `"load_checkpoint": true`. In that
case, we try to load the checkpoint file using simplemc::load_json_checkpoint:

```cpp
// if requested, load a checkpoint file to resume a previous simulation
if (load_checkpoint && std::filesystem::exists(checkpoint_path)) {
    fmt::println("Loading checkpoint from {}.", checkpoint_path.string());
    simplemc::load_json_checkpoint(checkpoint_path, rng, us, ms, stats, cfg);
}
```

@subsection tut_mc_2_running Running with checkpoints

Once we have prepared all the components according to the user input, we can run the simulation.

The run section looks exactly like in @ref tut_mc_1 — the only differences are 
- that simplemc::run now receives a simplemc::run_callbacks object whose `on_checkpoint` hook writes a
checkpoint, and 
- that the finished run is checkpointed one last time:

```cpp
// checkpoint callback
const auto cbs = simplemc::run_callbacks {
    .on_checkpoint =
        [&](const simplemc::simulation_ctx& c) {
            fmt::println("Writing checkpoint at step {}.", c.steps_done);
            simplemc::save_json_checkpoint(checkpoint_path, rng, us, ms, stats + c, cfg);
        },
};

// run the simulation
fmt::println("\nRunning the simulation with the following parameters:");
simplemc::print(params);
fmt::println("");

stats += simplemc::run(rng, kernel, ms, params, cbs);

fmt::println("\nSimulation finished. Final statistics:");
simplemc::print(stats);

// write the final checkpoint to disk
fmt::println("\nWriting final checkpoint.");
simplemc::save_json_checkpoint(checkpoint_path, rng, us, ms, stats, cfg);
```

The checkpoint callback is very straightforward. It simply calls simplemc::save_json_checkpoint, the
analog of simplemc::load_json_checkpoint.

The only subtlety is that we pass `stats + c` to the save function, where `c` is the current 
simplemc::simulation_ctx of the run. The reason is that the simulation statistics are only updated
after the simulation has finished (see the line `stats += simplemc::run(...)`). Therefore, we need to 
add the current run's context to the cumulative statistics to make sure that the checkpoint contains 
the correct MC steps done and runtime.

Obtaining and printing the final results at the end of `main` is identical to @ref tut_mc_1.

@subsection tut_mc_2_output Output

Let's look at some of the output from the program.

The first invocation bootstraps the input-config file and exits:

```
No input config file found --> generated input_config_tut_mc_2.json.
Edit the file and re-run to start a simulation.
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
Reading input config from input_config_tut_mc_2.json.

Running the simulation with the following parameters:
Max. steps              = 1000000
Max. time               = 60 sec
Steps per cycle         = 1
Cycles per check        = 10000
Skip measurements       = false
Checkpoint after steps  = 250000
Checkpoint after time   = 1.7976931348623157e+308 sec

Writing checkpoint at step 250000.
Writing checkpoint at step 500000.
Writing checkpoint at step 750000.
Writing checkpoint at step 1000000.

Simulation finished. Final statistics:
Runtime        = 0.256964541 sec
MC steps done  = 1000000
Steps per sec  = 3.89159e+06

Writing final checkpoint.

Results:
I_MC = 0.9092568281756135
I    = 0.9092974268256817
```

The result is identical to @ref tut_mc_1 — same seed, same Markov chain, same result. 

Now let's open the input-config file and set `"load_checkpoint": true`, before running for a third 
time:

```
Reading input config from input_config_tut_mc_2.json.
Loading checkpoint from checkpoint_tut_mc_2.json.

Running the simulation with the following parameters:
Max. steps              = 1000000
Max. time               = 60 sec
Steps per cycle         = 1
Cycles per check        = 10000
Skip measurements       = false
Checkpoint after steps  = 250000
Checkpoint after time   = 1.7976931348623157e+308 sec

Writing checkpoint at step 250000.
Writing checkpoint at step 500000.
Writing checkpoint at step 750000.
Writing checkpoint at step 1000000.

Simulation finished. Final statistics:
Runtime        = 0.5237529160000001 sec
MC steps done  = 2000000
Steps per sec  = 3.81859e+06

Writing final checkpoint.

Results:
I_MC = 0.9088469661519145
I    = 0.9092974268256817
```

The run picks up exactly where the previous one stopped: the final statistics now show the cumulative 
record of both runs (\f$ 2 \times 10^6 \f$ MC steps and \f$ 0.52 \f$ seconds runtime), the
restored accumulator keeps counting, and the estimate stays consistent with the exact value at the
expected \f$ \mathcal{O}(1/\sqrt{N}) \f$ accuracy.

In the next tutorial (@ref tut_mc_3), we will see how to use MPI to run multiple independent
simulations in parallel.

@section tut_mc_2_code Full code

@include tutorials/tut_mc_2.cpp
