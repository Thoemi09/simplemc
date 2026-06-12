// MC integration of cos(x) over [0, 2], assembled directly from the modular building blocks.

#include <fmt/format.h>
#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/mc.hpp>
#include <simplemc/random/xoshiro256.hpp>

#include <cmath>
#include <functional>

namespace {

// User state shared between the update and the measurement.
struct my_state {
    double x = 1.0;
    double a = 0.0;
    double b = 2.0;
    std::function<double(double)> f = [](double v) { return std::cos(v); };
};

// Uniform-resampling update: propose a new x uniformly in [a, b], always accept.
struct uniform_update {
    my_state* s;
    simplemc::xoshiro256ss* rng;
    std::uniform_real_distribution<double> uni01 { 0.0, 1.0 };
    double new_x = 0.0;

    double attempt() {
        new_x = s->a + (s->b - s->a) * uni01(*rng);
        return 1.0;
    }
    void accept() { s->x = new_x; }
};

// Observer that accumulates samples of f(x) and owns its own accumulator.
struct integral_observer {
    const my_state* s;
    simplemc::mean_acc<double> acc {};

    void measure() { acc << s->f(s->x); }
};

} // namespace

int main() {
    my_state state;
    simplemc::xoshiro256ss rng { 0xc0ffee };
    simplemc::simulation_stats stats;

    // Assemble the pieces by hand instead of using the simulation aggregate.
    simplemc::update_set updates;
    updates.add({ uniform_update { .s = &state, .rng = &rng }, "uniform", 1.0 });

    simplemc::measurement_set meas;
    meas.add({ integral_observer { .s = &state }, "integral" });

    simplemc::metropolis_kernel kernel { updates };

    // Optional callbacks: print progress every cycle batch (i.e. once per cycles_per_check).
    auto cbs = simplemc::run_callbacks {
        .on_checkpoint = [](const simplemc::simulation_ctx& x) {
            fmt::print("  ... checkpoint at step {} (runtime {:.3f} s)\n", x.steps_done, x.elapsed());
        },
    };

    simplemc::run(kernel, meas,
        { .max_steps = 1'000'000, .max_time = 60.0, .steps_per_cycle = 1, .cycles_per_check = 10'000,
            .checkpoint_after_steps = 250'000 },
        stats, rng, cbs);

    const auto* m = meas.get<integral_observer>("integral");
    const double result = m->acc.mean() * (state.b - state.a);
    const double exact = std::sin(state.b) - std::sin(state.a);
    fmt::print("steps:  {}\nresult: {}\nexact:  {}\n", stats.last_steps_done, result, exact);
}
