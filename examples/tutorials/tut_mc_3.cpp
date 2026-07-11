// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

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

// The integrand.
double f(double x) {
    return std::cos(x);
}

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

// Uniform-resampling update:
// - attempt(): proposes a new x uniformly in [a, b] and always returns 1.0.
// - accept(): accepts the proposal by updating the current sample x.
struct uniform_update {
    mc_config* cfg;
    simplemc::xoshiro256ss* rng;
    std::uniform_real_distribution<double> uni01 { 0.0, 1.0 };
    double new_x = 0.0;

    double attempt() {
        new_x = cfg->a + (cfg->b - cfg->a) * uni01(*rng);
        return 1.0;
    }
    void accept() { cfg->x = new_x; }
};

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

    // MPI channel: combine the accumulators of all ranks in place.
    friend void simplemc_mpi_collect(const simplemc::mpi::communicator& comm, integral& m) {
        simplemc_mpi_collect(comm, m.acc);
    }
};

} // namespace

int main(int argc, char** argv) {
    // initialize the MPI environment and communicator
    simplemc::mpi::environment env { argc, argv };
    simplemc::mpi::communicator comm;
    const bool is_root = comm.rank() == 0;

    // random number generator with a deterministic, rank-dependent seed
    simplemc::xoshiro256ss rng;
    simplemc::seed_rng(rng, comm.rank());

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

    // paths of the shared input-config file and the rank-local checkpoint file
    const std::filesystem::path config_path { "input_config_tut_mc_3.json" };
    const std::filesystem::path checkpoint_path { fmt::format("checkpoint_tut_mc_3_rank_{}.json", comm.rank()) };

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

    // write the final per-rank checkpoint
    simplemc::mpi::println(comm, "\nWriting final checkpoints.");
    simplemc::save_json_checkpoint(checkpoint_path, rng, us, ms, stats, cfg);

    // collect data and results from all ranks
    simplemc_mpi_collect(comm, us, ms, stats);

    simplemc::mpi::println(comm, "\nSimulation finished. Collected statistics of all ranks:");
    simplemc::print(comm, stats);

    // fetch the integral measurement by name and rescale \bar{f} by (b - a) to estimate I_MC
    const auto* m = ms.get<integral>("integral");
    const double I_MC = m->acc.mean() * (cfg.b - cfg.a);

    // print results and compare with the exact value
    simplemc::mpi::println(comm, "\nResults:");
    simplemc::mpi::println(comm, "I_MC = {}", I_MC);
    simplemc::mpi::println(comm, "I    = {}", std::sin(cfg.b) - std::sin(cfg.a));
}
