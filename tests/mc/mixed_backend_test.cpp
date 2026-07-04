#include <simplemc/mc.hpp>
#include <simplemc/random/xoshiro256.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <memory>

using namespace simplemc;

namespace {

// The mc types are no longer tied to any one serializer: they serialize through whatever concrete
// serializer they are handed (their save/load are generic over the `serializer` concept). The
// checkpoint and input-config channels stay independent because they are driven by different free
// functions (simplemc_save / simplemc_load vs simplemc_save_input_config / simplemc_load_input_config)
// writing into different serializer instances. These tests pin down that disjointness and the
// per-user-type opt-in, plus that the same components round-trip through more than one backend chosen at
// runtime.

// User update opting into BOTH channels via generic-over-serializer ADL hooks.
struct dual_update {
    std::shared_ptr<int> state_counter = std::make_shared<int>(0);
    std::shared_ptr<double> config_threshold = std::make_shared<double>(0.0);
    double attempt() { return 1.0; }
    void accept() {}
};

template <serializer S>
void simplemc_save(S& s, const dual_update& u) {
    s.save_at("state_counter", *u.state_counter);
}
template <serializer S>
void simplemc_load(const S& s, dual_update& u) {
    s.load_at("state_counter", *u.state_counter);
}
template <serializer S>
void simplemc_save_input_config(S& s, const dual_update& u) {
    s.save_at("config_threshold", *u.config_threshold);
}
template <serializer S>
void simplemc_load_input_config(const S& s, dual_update& u) {
    s.try_load_at("config_threshold", *u.config_threshold);
}

// User update with ONLY the checkpoint hook — the input-config channel must silently skip it.
struct state_only_update {
    std::shared_ptr<int> ticks = std::make_shared<int>(0);
    double attempt() { return 1.0; }
    void accept() {}
};

template <serializer S>
void simplemc_save(S& s, const state_only_update& u) {
    s.save_at("ticks", *u.ticks);
}
template <serializer S>
void simplemc_load(const S& s, state_only_update& u) {
    s.load_at("ticks", *u.ticks);
}

// Drive a short run, then fold counters into the cumulative state.
template <typename UpdateSet, typename MeasSet>
void run_and_accumulate(
    UpdateSet& updates, MeasSet& meas, simulation_stats& stats, xoshiro256ss& rng, const simulation_params& p) {
    metropolis_kernel kernel { updates };
    const auto ctx = run(rng, kernel, meas, p);
    accumulate_simulation_stats(stats, ctx);
    updates.accumulate_counters();
}

} // namespace

TEST(MCMixedBackend, InstantiationCompiles) {
    update_set updates { update { dual_update {}, "tunable", 2.5 } };
    EXPECT_DOUBLE_EQ(updates.at<0>().weight, 2.5);
}

TEST(MCMixedBackend, StateAndInputConfigDispatchIndependently) {
    xoshiro256ss rng { 7 };
    simulation_stats stats;
    dual_update src_u;
    *src_u.state_counter = 42;
    *src_u.config_threshold = 1.25;
    update_set updates { update { src_u, "tunable", 3.0 } };
    measurement_set<> meas;

    run_and_accumulate(
        updates, meas, stats, rng, { .max_steps = 5, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 5 });

    // Save state and input config into two separate JSON serializer instances.
    json_serializer state_w;
    simplemc_save(state_w, rng, updates, meas, stats);

    json_serializer input_w;
    const simulation_params params;
    simplemc_save_input_config(input_w, params, updates, meas);

    // Checkpoint channel carries the state-only fields:
    EXPECT_TRUE(state_w.root().contains("rng"));
    EXPECT_TRUE(state_w.root().contains("stats"));
    EXPECT_TRUE(state_w.root()["updates"]["tunable"].contains("cumulative_nprops"));
    EXPECT_TRUE(state_w.root()["updates"]["tunable"]["user"].contains("state_counter"));

    // Input-config channel carries only the input-config fields, no state fields:
    EXPECT_FALSE(input_w.root().contains("rng"));
    EXPECT_FALSE(input_w.root().contains("stats"));
    EXPECT_FALSE(input_w.root()["updates"]["tunable"].contains("cumulative_nprops"));
    EXPECT_TRUE(input_w.root()["updates"]["tunable"].contains("weight"));
    EXPECT_TRUE(input_w.root()["updates"]["tunable"]["user"].contains("config_threshold"));

    // Destination: load state from the state doc, then input-config from the input-config doc.
    // Cross-talk would show up as wrong values or extra throws.
    xoshiro256ss dst_rng;
    simulation_stats dst_stats;
    simulation_params dst_params;
    dual_update dst_u;
    auto dst_state_counter = dst_u.state_counter;
    auto dst_config_threshold = dst_u.config_threshold;
    update_set dst_updates { update { dst_u, "tunable", 1.0 } };
    measurement_set<> dst_meas;

    const json_serializer state_r { state_w.root() };
    simplemc_load(state_r, dst_rng, dst_updates, dst_meas, dst_stats);

    const json_serializer input_r { input_w.root() };
    simplemc_load_input_config(input_r, dst_params, dst_updates, dst_meas);

    // State round-trip:
    EXPECT_EQ(dst_stats.cumulative_steps, stats.cumulative_steps);
    EXPECT_EQ(*dst_state_counter, 42);

    // Input-config round-trip:
    EXPECT_DOUBLE_EQ(dst_updates.at<0>().weight, 3.0);
    EXPECT_DOUBLE_EQ(*dst_config_threshold, 1.25);
}

TEST(MCMixedBackend, OneSidedOptInSilentlySkipsOtherChannel) {
    // state_only_update has only the checkpoint hook. Input-config save/load should be no-ops on the
    // user slot; no exceptions, no fields written.
    xoshiro256ss rng;
    simulation_stats stats;
    state_only_update u;
    *u.ticks = 99;
    update_set updates { update { u, "ticker", 1.0 } };
    measurement_set<> meas;

    json_serializer state_w;
    simplemc_save(state_w, rng, updates, meas, stats);
    EXPECT_TRUE(state_w.root()["updates"]["ticker"]["user"].contains("ticks"));

    json_serializer input_w;
    const simulation_params params;
    EXPECT_NO_THROW(simplemc_save_input_config(input_w, params, updates, meas));
    // Per-entry "weight" is still written (it's on the update entry, not the user value), but the
    // user slot is empty because state_only_update doesn't opt into input-config.
    const auto& ticker = input_w.root()["updates"]["ticker"];
    EXPECT_TRUE(ticker.contains("weight"));
    EXPECT_FALSE(ticker.contains("user"));
}

TEST(MCMixedBackend, SameComponentsRoundTripThroughRuntimeChosenBackend) {
    // The sets carry no serializer type, so the backend is a pure runtime choice at the call site
    // (here deduced from the file extension). The same components round-trip through JSON and, when
    // built with HDF5, through HDF5 as well.
    xoshiro256ss rng { 5 };
    simulation_stats stats;
    state_only_update u;
    *u.ticks = 7;
    update_set updates { update { u, "ticker", 1.0 } };
    measurement_set<> meas;
    run_and_accumulate(
        updates, meas, stats, rng, { .max_steps = 8, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 4 });

    const auto tmp = std::filesystem::temp_directory_path();
    auto check_backend = [&](const std::filesystem::path& path) {
        save_checkpoint(path, rng, updates, meas, stats);

        xoshiro256ss dst_rng { 1 };
        simulation_stats dst_stats;
        state_only_update dst_u;
        auto dst_ticks = dst_u.ticks;
        update_set dst_updates { update { dst_u, "ticker", 1.0 } };
        measurement_set<> dst_meas;
        load_checkpoint(path, dst_rng, dst_updates, dst_meas, dst_stats);

        EXPECT_EQ(dst_stats.cumulative_steps, stats.cumulative_steps);
        EXPECT_EQ(*dst_ticks, 7);
        std::filesystem::remove(path);
    };

    check_backend(tmp / "mc_mixed_backend.json");
#ifdef SIMPLEMC_USE_HDF5
    check_backend(tmp / "mc_mixed_backend.h5");
#endif
}
