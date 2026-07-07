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
    // the MC configuration and the random number generator driving the simulation
    mc_config cfg;
    simplemc::xoshiro256ss rng { 0xc0ffee };

    // register the update with a unique name and a selection weight
    simplemc::update_set us { simplemc::update { uniform_update { .cfg = &cfg, .rng = &rng }, "uniform", 1.0 } };

    // register the measurement with a unique name
    simplemc::measurement_set ms { simplemc::measurement { integral { .cfg = &cfg }, "integral" } };

    // the kernel implements the Metropolis algorithm on top of the update set
    simplemc::metropolis_kernel kernel { us };

    // run the simulation: measurements are taken once per cycle, so steps_per_cycle = 1 measures
    // after every step
    const auto ctx = simplemc::run(rng, kernel, ms,
        { .max_steps = 1'000'000, .max_time = 60.0, .steps_per_cycle = 1, .cycles_per_check = 10'000 });

    // fetch the integral measurement by name and rescale the sample mean by the volume to estimate
    // the integral
    const auto* m = ms.get<integral>("integral");
    const double result = m->acc.mean() * (cfg.b - cfg.a);
    const double exact = std::sin(cfg.b) - std::sin(cfg.a);

    // print results
    fmt::println("runtime: {}", ctx.runtime);
    fmt::println("steps:   {}", ctx.steps_done);
    fmt::println("result:  {}", result);
    fmt::println("exact:   {}", exact);
}
