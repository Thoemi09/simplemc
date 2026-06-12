// MC integration of cos(x) over [0, 2], checkpointed and restored via the component-borrowing
// json_checkpoint_writer with an attached user-state object.

#include <fmt/format.h>
#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/mc.hpp>
#include <simplemc/random/xoshiro256.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/serialize/json/eigen.hpp> // nlohmann <-> Eigen adapters, needed to (de)serialize mean_acc

#include <cmath>
#include <filesystem>
#include <functional>

namespace {

// User state shared between the update and the measurement. Only the serializable fields are
// persisted; the sampling function is reconstructed by the caller after a restore.
struct my_state {
    double x = 1.0;
    double a = 0.0;
    double b = 2.0;
    std::function<double(double)> f = [](double v) { return std::cos(v); };

    template <simplemc::serializer S>
    friend void simplemc_save(S& s, const my_state& st) {
        s.save_at("x", st.x);
        s.save_at("a", st.a);
        s.save_at("b", st.b);
    }
    template <simplemc::serializer S>
    friend void simplemc_load(const S& s, my_state& st) {
        s.load_at("x", st.x);
        s.load_at("a", st.a);
        s.load_at("b", st.b);
    }
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

// Observer that accumulates samples of f(x) and owns its own accumulator. The accumulator is
// checkpointed so a restored observer can keep reporting (and resume accumulating).
struct integral_observer {
    const my_state* s;
    simplemc::mean_acc<double> acc {};

    void measure() { acc << s->f(s->x); }

    template <simplemc::serializer S>
    friend void simplemc_save(S& sr, const integral_observer& o) {
        sr.save_at("acc", o.acc);
    }
    template <simplemc::serializer S>
    friend void simplemc_load(const S& sr, integral_observer& o) {
        sr.load_at("acc", o.acc);
    }
};

} // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path() / "doc_mc_checkpoint.json";

    my_state state;
    simplemc::xoshiro256ss rng { 0xc0ffee };
    simplemc::simulation_stats stats;

    simplemc::update_set updates;
    updates.add({ uniform_update { .s = &state, .rng = &rng }, "uniform", 1.0 });

    simplemc::measurement_set meas;
    meas.add({ integral_observer { .s = &state }, "integral" });

    simplemc::metropolis_kernel kernel { updates };

    simplemc::run(kernel, meas,
        { .max_steps = 1'000'000, .max_time = 60.0, .steps_per_cycle = 1, .cycles_per_check = 10'000 },
        stats, rng);

    // Fold the run into the cumulative counters, then snapshot the components plus the user state.
    accumulate_simulation_stats(stats);
    const auto writer = simplemc::make_json_checkpoint_writer(rng, updates, meas, stats, &state, path);
    writer(simplemc::simulation_ctx {});
    fmt::print("wrote checkpoint to {}\n", path.string());

    // Restore into a fresh set of components with the same registration.
    my_state restored;
    simplemc::xoshiro256ss rng2;
    simplemc::simulation_stats stats2;

    simplemc::update_set updates2;
    updates2.add({ uniform_update { .s = &restored, .rng = &rng2 }, "uniform", 1.0 });

    simplemc::measurement_set meas2;
    meas2.add({ integral_observer { .s = &restored }, "integral" });

    simplemc::load_json_checkpoint(rng2, updates2, meas2, stats2, &restored, path);

    const auto* m = meas2.get<integral_observer>("integral");
    const double result = m->acc.mean() * (restored.b - restored.a);
    const double exact = std::sin(restored.b) - std::sin(restored.a);
    fmt::print("restored cumulative steps: {}\nrestored x: {}\nresult: {}\nexact:  {}\n",
        stats2.cumulative_steps, restored.x, result, exact);

    std::filesystem::remove(path);
}
