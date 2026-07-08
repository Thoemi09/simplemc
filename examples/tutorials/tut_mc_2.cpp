#include <fmt/format.h>
#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/mc.hpp>
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
// - x: the current sample,
// - [a, b]: the integration domain.
struct mc_config {
    double x = 1.0;
    double a = 0.0;
    double b = 2.0;

    // Checkpoint channel: the domain and the current sample are part of the checkpoint.
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

    // Input-config channel: only the domain is user-configurable.
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

// Integral measurement (as in tut_mc_1), now with checkpoint hooks for its accumulator so a
// restored measurement resumes accumulating.
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

} // namespace

int main() {
    const std::filesystem::path config_path { "input_config_tut_mc_2.json" };
    const std::filesystem::path checkpoint_path { "checkpoint_tut_mc_2.json" };

    // the MC components: MC configuration, RNG, cumulative statistics, updates and measurements,
    // kernel, simulation parameters, and callbacks for checkpointing
    mc_config cfg;
    simplemc::xoshiro256ss rng { 0xc0ffee };
    simplemc::simulation_stats stats;
    simplemc::update_set us { simplemc::update { uniform_update { .cfg = &cfg, .rng = &rng }, "uniform", 1.0 } };
    simplemc::measurement_set ms { simplemc::measurement { integral { .cfg = &cfg }, "integral" } };
    simplemc::metropolis_kernel kernel { us };
    simplemc::simulation_params params;
    // checkpoints carry the standard components plus the MC configuration under "config" (the cfg
    // overload of save_json_checkpoint); stats + c forms the correct mid-run totals
    const auto cbs = simplemc::run_callbacks {
        .on_checkpoint =
            [&](const simplemc::simulation_ctx& c) {
                fmt::println("Writing checkpoint at step {}.", c.steps_done);
                simplemc::save_json_checkpoint(checkpoint_path, rng, us, ms, stats + c, cfg);
            },
    };

    // if there is no input-config file, write one with default parameters and exit; the input
    // config carries params/updates/measurements plus the MC configuration under "config" and a
    // free-form "load_checkpoint" flag via the extra-keys hook
    if (!std::filesystem::exists(config_path)) {
        fmt::print("No input config file found");
        simplemc::save_json_input_config(config_path, params, us, ms, cfg,
            [](simplemc::json_serializer& s) { s.save_at("load_checkpoint", false); });
        fmt::println(" --> generated {}.", config_path.string());
        fmt::println("Edit the file and re-run to start a simulation.");
        return 0;
    }

    // if there is an input-config file, read it to configure the simulation accordingly
    fmt::println("Reading input config from {}.", config_path.string());
    bool load_checkpoint = false;
    simplemc::load_json_input_config(config_path, params, us, ms, cfg,
        [&](const simplemc::json_serializer& s) { s.try_load_at("load_checkpoint", load_checkpoint); });

    // if requested, load a checkpoint file to resume a previous simulation
    if (load_checkpoint && std::filesystem::exists(checkpoint_path)) {
        fmt::println("Loading checkpoint from {}.", checkpoint_path.string());
        simplemc::load_json_checkpoint(checkpoint_path, rng, us, ms, stats, cfg);
    }

    // run the simulation and fold it into the cumulative simulation statistics
    fmt::println("Starting the simulation.");
    stats += simplemc::run(rng, kernel, ms, params, cbs);

    // write the final checkpoint to disk
    fmt::println("Writing final checkpoint.");
    simplemc::save_json_checkpoint(checkpoint_path, rng, us, ms, stats, cfg);

    // result readout as in tut_mc_1, plus the cumulative step count across restores
    const auto* m = ms.get<integral>("integral");
    const double result = m->acc.mean() * (cfg.b - cfg.a);
    const double exact = std::sin(cfg.b) - std::sin(cfg.a);
    fmt::println("cumulative steps: {}", stats.cumulative_steps);
    fmt::println("result:           {}", result);
    fmt::println("exact:            {}", exact);
}
