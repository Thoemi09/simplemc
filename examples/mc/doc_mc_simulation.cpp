// MC integration of cos(x) over [0, 2], driven by the simulation convenience aggregate.

#include <fmt/format.h>
#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/mc.hpp>
#include <simplemc/random/xoshiro256.hpp>

#include <cmath>
#include <functional>
#include <random>

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

    simplemc::simulation<> sim { simplemc::xoshiro256ss { 0xc0ffee } };
    sim.add_update(uniform_update { .s = &state, .rng = &sim.rng() }, "uniform", 1.0);
    sim.add_measurement(integral_observer { .s = &state }, "integral");

    sim.run({ .max_steps = 1'000'000, .max_time = 60.0, .steps_per_cycle = 1, .cycles_per_check = 10'000 });

    const auto* m = sim.get_measurement<integral_observer>("integral");
    const double result = m->acc.mean() * (state.b - state.a);
    const double exact = std::sin(state.b) - std::sin(state.a);
    fmt::print("steps:  {}\nresult: {}\nexact:  {}\n", sim.stats().steps_done, result, exact);
}
