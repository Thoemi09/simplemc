@page tut_mc_3 MC Tutorial 3: Parallelizing with MPI

[TOC]

In this tutorial, we parallelize the MC integration from @ref tut_mc_2 with @ref simplemc-mpi. Every
MPI rank runs its own independent Markov chain, and the results of all ranks are combined into a 
single estimate at the end.

The simulation itself is mostly untouched — the update, the kernel, and the whole
input-config/checkpoint workflow are unchanged from @ref tut_mc_2. The measurement gains one new
MPI hook, described below — so this page only walks through the differences.

@section tut_mc_3_strategy The parallelization strategy

MC simulations are *embarrassingly parallel*: instead of one Markov chain with \f$ N \f$ samples, we
run \f$ M \f$ statistically independent chains with \f$ N \f$ samples each and average their results, 
reducing the statistical error like \f$ 1/\sqrt{M N} \f$ at the wall-clock time of a single chain.

Concretely, compared to @ref tut_mc_2, this tutorial adds four MPI-related features:

1. Every rank seeds its RNG with a deterministic, rank-dependent call to simplemc::seed_rng, so that 
the chains are independent *and* reproducible.
2. All ranks share **one input-config file**, bootstrapped and written only by the root rank.
3. Every rank writes **its own checkpoint file**, so a parallel run can be continued exactly — each
chain picks up its own RNG state, sample, and accumulator. A continuation must therefore be started
with the same number of ranks.
4. After the final checkpoint, the update counters, the measurement accumulators, and the simulation
statistics of all ranks are combined **in place** with the free function `simplemc_mpi_collect`, and 
the root rank reports the collected results.

@section tut_mc_3_details Step-by-step guide

The overall structure of the code is the same as in @ref tut_mc_2, except for the MPI-related includes 
and the `main` signature, which now takes the command-line arguments needed to initialize MPI:

```cpp
#include <fmt/format.h>
#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/mc.hpp>
#include <simplemc/mpi.hpp>
#include <simplemc/random/seed_rng.hpp>
#include <simplemc/random/xoshiro256.hpp>

#include <cmath>
#include <filesystem>
#include <random>

namespace {
    // user types go here
} // namespace

int main(int argc, char** argv) {
    // code snippets go here
}
```

@subsection tut_mc_3_collect Making the measurement MPI-collectable

The library already knows how to combine its own components across ranks: the free function
`simplemc_mpi_collect` sums the counters of a simplemc::update_set and the steps/runtime of a
simplemc::simulation_stats over all ranks.

For a *user* measurement, the library cannot know what "combining" means, so — just like for
serialization — the type opts in via an ADL hook. Our `integral` measurement simply delegates to the
hook of its simplemc::mean_acc, which merges the accumulators of all ranks in place:

```cpp
struct integral {
    const mc_config* cfg;
    simplemc::mean_acc<double> acc {};

    void measure() { acc << f(cfg->x); }

    // ... checkpoint channel as before ...

    // MPI channel: combine the accumulators of all ranks in place.
    friend void simplemc_mpi_collect(const simplemc::mpi::communicator& comm, integral& m) {
        simplemc_mpi_collect(comm, m.acc);
    }
};
```

The hook takes the value by mutable reference and reduces it in place (see 
simplemc::has_simplemc_mpi_collect). After the call, every rank holds the combined value. Once it is
present, the simplemc::measurement wrapper picks it up automatically when the measurement set is
collected. A measurement without the hook is silently skipped.

Everything else in the user types — `mc_config`, `uniform_update`, and the two serialization channels 
— is byte-for-byte the code from @ref tut_mc_2.

@subsection tut_mc_3_env MPI environment and per-rank seeding

`main` now starts by initializing MPI and constructing the world communicator (see @ref tut_mpi_1):

```cpp
// initialize the MPI environment and communicator
simplemc::mpi::environment env { argc, argv };
simplemc::mpi::communicator comm;
const bool is_root = comm.rank() == 0;
```

The chains are only independent if every rank draws different random numbers. simplemc::seed_rng
derives a deterministic stream per rank by mixing a base seed (here the default seed) with the rank
through a simplemc::splitmix64 sequence:

```cpp
// random number generator with a deterministic, rank-dependent seed
simplemc::xoshiro256ss rng;
simplemc::seed_rng(rng, comm.rank());
```

Re-running the program therefore reproduces the exact same result. 

Note that simplemc::seed_rng reseeds through a `std::seed_seq`, so even rank 0's chain differs from 
the serial chains of @ref tut_mc_1 and @ref tut_mc_2.

Assembling the components (update set, measurement set, kernel, parameters, statistics) is unchanged.

@subsection tut_mc_3_files One config file, one checkpoint file per rank

All ranks share the input-config file, while the checkpoint path becomes rank-local:

```cpp
// paths of the shared input-config file and the rank-local checkpoint file
const std::filesystem::path config_path { "input_config_tut_mc_3.json" };
const std::filesystem::path checkpoint_path { fmt::format("checkpoint_tut_mc_3_rank_{}.json", comm.rank()) };
```

The bootstrapping needs a little care — not because of file corruption (the write is atomic, see
simplemc::atomic_file_write), but because ranks probing the file system on their own could take
*different branches*: a rank probing only after the root rank has already written the file would see 
it and start a simulation, while all other ranks have long exited — and the lone rank would then hang 
in the first collective call. Instead, only the root rank checks for the file and broadcasts its 
finding, so all ranks take the same branch:

```cpp
// only the root rank checks for the input-config file and broadcasts the result, so all ranks
// take the same branch (independently probing ranks could disagree on the file's existence)
int have_config = 0;
if (is_root) {
    have_config = std::filesystem::exists(config_path);
}
simplemc::mpi::broadcast(have_config, 0, comm);

// if there is no input-config file, let the root rank write one with the default parameters,
// then exit on all ranks
if (have_config == 0) {
    if (is_root) {
        fmt::print("No input config file found");
        simplemc::save_json_input_config(config_path, params, us, ms, cfg,
            [](simplemc::json_serializer& s) { s.save_at("load_checkpoint", false); });
        fmt::println(" --> generated {}.", config_path.string());
        fmt::println("Edit the file and re-run to start a simulation.");
    }
    return 0;
}
```

(The flag is an `int` because `bool` has no corresponding MPI datatype, see simplemc::mpi::mpi_type.)

*Reading* the shared file concurrently is harmless, so all ranks load the config exactly as in
@ref tut_mc_2, and each rank then restores its own checkpoint file if a continuation was requested:

```cpp
// if there is an input-config file, all ranks read it to configure the simulation accordingly
simplemc::mpi::println(comm, "Reading input config from {}.", config_path.string());
bool load_checkpoint = false;
simplemc::load_json_input_config(config_path, params, us, ms, cfg,
    [&](const simplemc::json_serializer& s) { s.try_load_at("load_checkpoint", load_checkpoint); });

// if requested, each rank loads its own checkpoint file to resume a previous simulation
if (load_checkpoint && std::filesystem::exists(checkpoint_path)) {
    simplemc::mpi::println(comm, "Loading per-rank checkpoints.");
    simplemc::load_json_checkpoint(checkpoint_path, rng, us, ms, stats, cfg);
}
```

The status messages use simplemc::mpi::println, the root-gated counterpart of `fmt::println`. It
prints only on one rank of the communicator (rank `0` unless another root rank is passed), so every
rank can call it unconditionally. Without it, each message would appear once per rank. Blocks that
do more than printing (like the bootstrap above, which also writes a file) still use an explicit
`is_root` guard.

Since every Markov chain is restored from its own file, a continued parallel run is exact — but only 
if it is launched with the same number of ranks as the run that wrote the checkpoints.

@subsection tut_mc_3_running Running, collecting, and reporting

The run section is the one from @ref tut_mc_2, with two adjustments: console output goes through the 
root-gated printing functions (simplemc::mpi::println and the communicator overload of
simplemc::print), and the checkpoint callback writes to the rank-local path:

```cpp
// checkpoint callback: every rank writes its own checkpoint file
const auto cbs = simplemc::run_callbacks {
    .on_checkpoint =
        [&](const simplemc::simulation_ctx& c) {
            simplemc::mpi::println(comm, "Writing checkpoints at step {}.", c.steps_done);
            simplemc::save_json_checkpoint(checkpoint_path, rng, us, ms, stats + c, cfg);
        },
};

// run one independent simulation per rank
simplemc::mpi::println(comm, "\nRunning {} independent simulations with the following parameters:", comm.size());
simplemc::print(comm, params);
simplemc::mpi::println(comm, "");

stats += simplemc::run(rng, kernel, ms, params, cbs);
```

The end of `main` is where the ranks finally talk to each other. The **order matters**:

```cpp
// write the final per-rank checkpoint
simplemc::mpi::println(comm, "\nWriting final checkpoints.");
simplemc::save_json_checkpoint(checkpoint_path, rng, us, ms, stats, cfg);

// collect data and results from all ranks
simplemc_mpi_collect(comm, us, ms, stats);
```

In our example, the `simplemc_mpi_collect` call all-reduces the update counters, the mean accumulator
from the `integral` measurement, and the simulation statistics. 

This has two consequences:

- It must run **after** the final checkpoint is written: A checkpoint has to store the state of the
rank-specific Markov chain, not the sum of all of them, or a continued run would count every sample 
\f$ M \f$ times.
- It must run **exactly once**: Collecting is not idempotent. A second call would double the results. 
For the same reason we do not collect inside the checkpoint callback.

After the collect, every rank holds the combined components, so the closing block of @ref tut_mc_2
can run on all ranks. The root-gated printing functions make sure the report appears only once:

```cpp
simplemc::mpi::println(comm, "\nSimulation finished. Collected statistics of all ranks:");
simplemc::print(comm, stats);

// fetch the integral measurement by name and rescale \bar{f} by (b - a) to estimate I_MC
const auto* m = ms.get<integral>("integral");
const double I_MC = m->acc.mean() * (cfg.b - cfg.a);

// print results and compare with the exact value
simplemc::mpi::println(comm, "\nResults:");
simplemc::mpi::println(comm, "I_MC = {}", I_MC);
simplemc::mpi::println(comm, "I    = {}", std::sin(cfg.b) - std::sin(cfg.a));
```

@subsection tut_mc_3_output Output

The three-invocation workflow of @ref tut_mc_2 stays the same, now launched with `mpirun`. The first
invocation bootstraps the input-config file (its content is identical to the one shown in 
@ref tut_mc_2) and exits on all ranks:

```
$ mpirun -np 4 ./tut_mc_3
No input config file found --> generated input_config_tut_mc_3.json.
Edit the file and re-run to start a simulation.
```

The second invocation runs four independent chains and writes one checkpoint file per rank:

```
$ mpirun -np 4 ./tut_mc_3
Reading input config from input_config_tut_mc_3.json.

Running 4 independent simulations with the following parameters:
Max. steps              = 1000000
Max. time               = 60 sec
Steps per cycle         = 1
Cycles per check        = 10000
Skip measurements       = false
Checkpoint after steps  = 250000
Checkpoint after time   = 1.7976931348623157e+308 sec

Writing checkpoints at step 250000.
Writing checkpoints at step 500000.
Writing checkpoints at step 750000.
Writing checkpoints at step 1000000.

Writing final checkpoints.

Simulation finished. Collected statistics of all ranks:
Runtime        = 1.037476917 sec
MC steps done  = 4000000
Steps per sec  = 3.85551e+06

Results:
I_MC = 0.9092888615013313
I    = 0.9092974268256817
```

In roughly the wall-clock time of the serial run from @ref tut_mc_2, we now collect \f$ 4 \times 10^6 
\f$ samples. 

Note that the collected runtime is the *sum* of the per-rank runtimes (about four times the wall-clock 
time), so the reported rate is the average per-rank rate. 

The working directory now contains  `checkpoint_tut_mc_3_rank_0.json` ... 
`checkpoint_tut_mc_3_rank_3.json`.

Finally, we set `"load_checkpoint": true` in the input-config file and run a third time:

```
$ mpirun -np 4 ./tut_mc_3
Reading input config from input_config_tut_mc_3.json.
Loading per-rank checkpoints.

Running 4 independent simulations with the following parameters:
Max. steps              = 1000000
Max. time               = 60 sec
Steps per cycle         = 1
Cycles per check        = 10000
Skip measurements       = false
Checkpoint after steps  = 250000
Checkpoint after time   = 1.7976931348623157e+308 sec

Writing checkpoints at step 250000.
Writing checkpoints at step 500000.
Writing checkpoints at step 750000.
Writing checkpoints at step 1000000.

Writing final checkpoints.

Simulation finished. Collected statistics of all ranks:
Runtime        = 2.118384667 sec
MC steps done  = 8000000
Steps per sec  = 3.77646e+06

Results:
I_MC = 0.9093505464286017
I    = 0.9092974268256817
```

Every chain picks up exactly where it stopped: the collected statistics show the cumulative record
of both runs (\f$ 8 \times 10^6 \f$ MC steps), and the estimate stays consistent with the exact
value at the expected \f$ \mathcal{O}(1/\sqrt{N}) \f$ accuracy.

@section tut_mc_3_code Full code

@include tutorials/tut_mc_3.cpp
