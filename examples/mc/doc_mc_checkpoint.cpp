// MC integration of cos(x) over [0, 2] with user-written file I/O: an input-config file carrying an
// RNG seed next to the standard "params"/"updates"/"measurements" blocks, and a checkpoint written
// through simplemc::atomic_file_write with an attached user-state object. The document shape and the
// backend are call-site choices (for HDF5, swap the helper bodies to
// simplemc::hdf5_serializer { tmp, hdf5_file_mode::truncate }, the rest is unchanged). To checkpoint
// mid-run, put the same write call in an `on_checkpoint` lambda of simplemc::run_callbacks.

#include <fmt/format.h>
#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/mc.hpp>
#include <simplemc/random/seed_rng.hpp>
#include <simplemc/random/xoshiro256.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/serialize/json/eigen.hpp> // nlohmann <-> Eigen adapters, needed to (de)serialize mean_acc
#include <simplemc/serialize/json/file_io.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/file_io.hpp>

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <utility>

namespace {

// User state shared between the update and the measurement. Only the serializable fields are
// persisted; the sampling function is reconstructed by the caller after a restore.
struct my_state {
    double x = 1.0;
    double a = 0.0;
    double b = 2.0;
    std::function<double(double)> f = [](double v) { return std::cos(v); };

    template <simplemc::serializer S>
    friend void simplemc_save(S s, const my_state& st) {
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
    friend void simplemc_save(S sr, const integral_observer& o) {
        sr.save_at("acc", o.acc);
    }
    template <simplemc::serializer S>
    friend void simplemc_load(const S& sr, integral_observer& o) {
        sr.load_at("acc", o.acc);
    }
};

using updates_t = simplemc::update_set<uniform_update>;
using meas_t = simplemc::measurement_set<integral_observer>;

// User-written checkpoint I/O: compose simplemc::atomic_file_write + the composite simplemc_save /
// simplemc_load + JSON file I/O. Extra keys (here "state") are just more save_at calls next to the
// composite's standard sub-keys — the document shape is yours.
void write_checkpoint(const std::filesystem::path& path, const simplemc::xoshiro256ss& rng, const updates_t& updates,
    const meas_t& meas, const simplemc::simulation_stats& stats, const my_state& state) {
    simplemc::atomic_file_write(path, [&](const std::filesystem::path& tmp) {
        simplemc::json_serializer s;
        simplemc_save(s, rng, updates, meas, stats);
        s.save_at("state", state);
        simplemc::write_json_file(s.root(), tmp);
    });
}

void read_checkpoint(const std::filesystem::path& path, simplemc::xoshiro256ss& rng, updates_t& updates, meas_t& meas,
    simplemc::simulation_stats& stats, my_state& state) {
    nlohmann::json doc;
    simplemc::read_json_file(doc, path);
    const simplemc::json_serializer s { std::move(doc) };
    simplemc_load(s, rng, updates, meas, stats);
    s.load_at("state", state);
}

} // namespace

int main() {
    my_state state;
    simplemc::xoshiro256ss rng;
    simplemc::simulation_stats stats;

    simplemc::update_set updates { simplemc::update { uniform_update { .s = &state, .rng = &rng }, "uniform", 1.0 } };

    simplemc::measurement_set meas { simplemc::measurement { integral_observer { .s = &state }, "integral" } };

    // Write a default input-config file: the composite writes "params"/"updates"/"measurements", and
    // the RNG seed is an extra top-level key of the same open document.
    const auto config_path = std::filesystem::temp_directory_path() / "doc_mc_input_config.json";
    {
        const simplemc::simulation_params defaults {
            .max_steps = 1'000'000, .max_time = 60.0, .steps_per_cycle = 1, .cycles_per_check = 10'000
        };
        simplemc::json_serializer s;
        simplemc_save_input_config(s, defaults, updates, meas);
        s.save_at("seed", std::uint64_t { 0xc0ffee });
        simplemc::write_json_file(s.root(), config_path, { .indent = 4 });
        fmt::print("wrote input config to {}\n", config_path.string());
    }

    // Read it back (tolerant: missing keys leave destinations untouched) and seed the RNG from it.
    simplemc::simulation_params params;
    {
        std::uint64_t seed = 0;
        nlohmann::json doc;
        simplemc::read_json_file(doc, config_path);
        const simplemc::json_serializer s { std::move(doc) };
        simplemc_load_input_config(s, params, updates, meas);
        s.try_load_at("seed", seed);
        simplemc::seed_rng(rng, /*rank=*/0, seed); // pass the communicator rank under MPI
    }

    simplemc::metropolis_kernel kernel { updates };

    const auto ctx = simplemc::run(rng, kernel, meas, params);

    // Fold the run into the cumulative counters, then snapshot the components plus the user state.
    const auto path = std::filesystem::temp_directory_path() / "doc_mc_checkpoint.json";
    accumulate_simulation_stats(stats, ctx);
    write_checkpoint(path, rng, updates, meas, stats, state);
    fmt::print("wrote checkpoint to {}\n", path.string());

    // Restore into a fresh set of components with the same registration.
    my_state restored;
    simplemc::xoshiro256ss rng2;
    simplemc::simulation_stats stats2;

    simplemc::update_set updates2 {
        simplemc::update { uniform_update { .s = &restored, .rng = &rng2 }, "uniform", 1.0 }
    };

    simplemc::measurement_set meas2 { simplemc::measurement { integral_observer { .s = &restored }, "integral" } };

    read_checkpoint(path, rng2, updates2, meas2, stats2, restored);

    const auto* m = meas2.get<integral_observer>("integral");
    const double result = m->acc.mean() * (restored.b - restored.a);
    const double exact = std::sin(restored.b) - std::sin(restored.a);
    fmt::print("restored cumulative steps: {}\nrestored x: {}\nresult: {}\nexact:  {}\n", stats2.cumulative_steps,
        restored.x, result, exact);

    std::filesystem::remove(config_path);
    std::filesystem::remove(path);
}
