#include <simplemc/mc.hpp>
#include <simplemc/random/xoshiro256.hpp>
#include <simplemc/serialize/concepts.hpp>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <memory>

using namespace simplemc;

namespace {

// All mc types now serialize through the single library-wide mc_serializer (a backend-erasing
// variant_serializer). The checkpoint and input-config channels stay independent not because they
// use different *types*, but because they are driven by different free functions
// (simplemc_save / simplemc_load vs simplemc_save_input_config / simplemc_load_input_config) writing
// into different serializer instances. These tests pin down that disjointness, plus the per-payload
// opt-in into each channel.

// Pull the underlying nlohmann document out of a JSON-backed mc_serializer for inspection.
const nlohmann::json& doc_of(const mc_serializer& s) {
    return std::get<json_serializer>(s.backend()).root();
}

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
void run_and_accumulate(update_set& updates, measurement_set& meas, simulation_stats& stats, xoshiro256ss& rng,
    const simulation_params& p) {
    metropolis_kernel kernel { updates };
    run(rng, kernel, meas, stats, p);
    accumulate_simulation_stats(stats);
    updates.accumulate_counters();
}

} // namespace

TEST(MCMixedBackend, InstantiationCompiles) {
    update_set updates;
    updates.add({ dual_update {}, "tunable", 2.5 });
    EXPECT_DOUBLE_EQ(updates.data()[0].weight, 2.5);
}

TEST(MCMixedBackend, StateAndInputConfigDispatchIndependently) {
    xoshiro256ss rng { 7 };
    update_set updates;
    measurement_set meas;
    simulation_stats stats;
    dual_update src_u;
    *src_u.state_counter = 42;
    *src_u.config_threshold = 1.25;
    updates.add({ src_u, "tunable", 3.0 });

    run_and_accumulate(
        updates, meas, stats, rng, { .max_steps = 5, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 5 });

    // Save state and input config into two separate (JSON-backed) serializer instances.
    mc_serializer state_w { json_serializer {} };
    simplemc_save(state_w, rng, updates, meas, stats);

    mc_serializer input_w { json_serializer {} };
    const simulation_params params;
    simplemc_save_input_config(input_w, params, updates, meas);

    // Checkpoint channel carries the state-only fields:
    EXPECT_TRUE(doc_of(state_w).contains("rng"));
    EXPECT_TRUE(doc_of(state_w).contains("stats"));
    EXPECT_TRUE(doc_of(state_w)["updates"]["tunable"].contains("cumulative_nprops"));
    EXPECT_TRUE(doc_of(state_w)["updates"]["tunable"]["user"].contains("state_counter"));

    // Input-config channel carries only the input-config fields, no state fields:
    EXPECT_FALSE(doc_of(input_w).contains("rng"));
    EXPECT_FALSE(doc_of(input_w).contains("stats"));
    EXPECT_FALSE(doc_of(input_w)["updates"]["tunable"].contains("cumulative_nprops"));
    EXPECT_TRUE(doc_of(input_w)["updates"]["tunable"].contains("weight"));
    EXPECT_TRUE(doc_of(input_w)["updates"]["tunable"]["user"].contains("config_threshold"));

    // Destination: load state from the state doc, then input-config from the input-config doc.
    // Cross-talk would show up as wrong values or extra throws.
    xoshiro256ss dst_rng;
    update_set dst_updates;
    measurement_set dst_meas;
    simulation_stats dst_stats;
    simulation_params dst_params;
    dual_update dst_u;
    auto dst_state_counter = dst_u.state_counter;
    auto dst_config_threshold = dst_u.config_threshold;
    dst_updates.add({ dst_u, "tunable", 1.0 });

    const mc_serializer state_r { json_serializer { doc_of(state_w) } };
    simplemc_load(state_r, dst_rng, dst_updates, dst_meas, dst_stats);

    const mc_serializer input_r { json_serializer { doc_of(input_w) } };
    simplemc_load_input_config(input_r, dst_params, dst_updates, dst_meas);

    // State round-trip:
    EXPECT_EQ(dst_stats.cumulative_steps, stats.cumulative_steps);
    EXPECT_EQ(*dst_state_counter, 42);

    // Input-config round-trip:
    EXPECT_DOUBLE_EQ(dst_updates.data()[0].weight, 3.0);
    EXPECT_DOUBLE_EQ(*dst_config_threshold, 1.25);
}

TEST(MCMixedBackend, OneSidedOptInSilentlySkipsOtherChannel) {
    // state_only_update has only the checkpoint hook. Input-config save/load should be no-ops on the
    // user slot; no exceptions, no fields written.
    xoshiro256ss rng;
    update_set updates;
    measurement_set meas;
    simulation_stats stats;
    state_only_update u;
    *u.ticks = 99;
    updates.add({ u, "ticker", 1.0 });

    mc_serializer state_w { json_serializer {} };
    simplemc_save(state_w, rng, updates, meas, stats);
    EXPECT_TRUE(doc_of(state_w)["updates"]["ticker"]["user"].contains("ticks"));

    mc_serializer input_w { json_serializer {} };
    const simulation_params params;
    EXPECT_NO_THROW(simplemc_save_input_config(input_w, params, updates, meas));
    // Per-entry "weight" is still written (it's on the update entry, not the user payload), but the
    // user slot is empty because state_only_update doesn't opt into input-config.
    const auto& ticker = doc_of(input_w)["updates"]["ticker"];
    EXPECT_TRUE(ticker.contains("weight"));
    EXPECT_FALSE(ticker.contains("user"));
}
