#include <fmt/base.h>
#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/mc.hpp>
#include <simplemc/random/xoshiro256.hpp>

#include <cmath>
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
    static constexpr double a = 0.0;
    static constexpr double b = 2.0;
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

// Integral measurement: I = \int_0^2 f(x) dx
// - acc: accumulator for the sample mean of f(x), i.e. <f> \approx 1/N \sum_{i=1}^N f(x_i).
// - measure(): evaluates f(x) and accumulates the result.
struct integral {
    const mc_config* cfg;
    simplemc::mean_acc<double> acc {};

    void measure() { acc << f(cfg->x); }
};

} // namespace

int main() {
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

    // set the simulation parameters
    simplemc::simulation_params params {
        .max_steps = 1'000'000, .max_time = 60.0, .steps_per_cycle = 1, .cycles_per_check = 10'000
    };

    // run the simulation
    fmt::println("Running the simulation with the following parameters:");
    simplemc::print(params);

    simplemc::simulation_stats stats;
    stats += simplemc::run(rng, kernel, ms, params);

    fmt::println("\nSimulation finished. Final statistics:");
    simplemc::print(stats);

    // fetch the integral measurement by name and rescale \bar{f} by (b - a) to estimate I_MC
    const auto* m = ms.get<integral>("integral");
    const double I_MC = m->acc.mean() * (cfg.b - cfg.a);

    // print results and compare with the exact value
    fmt::println("\nResults:");
    fmt::println("I_MC = {}", I_MC);
    fmt::println("I    = {}", std::sin(cfg.b) - std::sin(cfg.a));
}
