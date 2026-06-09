#include <simplemc/mc.hpp>
#include <simplemc/random/xoshiro256.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <memory>

using namespace simplemc;

namespace {

struct always_accept {
    double attempt() { return 1.0; }
    void accept() { }
};

struct counting_measurement {
    void measure() { }
};

// User update that exposes only input-config hooks (no state simplemc_save).
struct configurable_update {
    std::shared_ptr<double> threshold = std::make_shared<double>(0.0);
    double attempt() { return 1.0; }
    void accept() { }
};

template <serializer S>
void simplemc_save_input_config(S& s, const configurable_update& u) {
    s.save_at("threshold", *u.threshold);
}

template <serializer S>
void simplemc_load_input_config(const S& s, configurable_update& u) {
    if (s.has("threshold")) {
        s.load_at("threshold", *u.threshold);
    }
}

// User measurement with input-config hooks.
struct configurable_measurement {
    std::shared_ptr<int> bins = std::make_shared<int>(1);
    void measure() { }
};

template <serializer S>
void simplemc_save_input_config(S& s, const configurable_measurement& m) {
    s.save_at("bins", *m.bins);
}

template <serializer S>
void simplemc_load_input_config(const S& s, configurable_measurement& m) {
    if (s.has("bins")) {
        s.load_at("bins", *m.bins);
    }
}

} // namespace

TEST(MCSimulationInputConfig, RoundTripPersistsWeightAndActiveAndUserConfig) {
    simulation<> src;
    configurable_update src_up;
    *src_up.threshold = 0.75;
    src.add_update(src_up, "tunable", 2.5);
    src.add_update(always_accept {}, "stateless", 1.0);

    configurable_measurement src_meas;
    *src_meas.bins = 64;
    src.add_measurement(src_meas, "histogram", /*is_active=*/false);
    src.add_measurement(counting_measurement {}, "stateless_meas");

    json_serializer w;
    auto w_sim = w["sim"];
    simplemc_save_input_config(w_sim, src);

    // Destination: same registration, different initial values.
    simulation<> dst;
    configurable_update dst_up;
    auto dst_threshold = dst_up.threshold;
    *dst_up.threshold = 0.0;
    dst.add_update(dst_up, "tunable", 1.0);
    dst.add_update(always_accept {}, "stateless", 1.0);

    configurable_measurement dst_meas;
    auto dst_bins = dst_meas.bins;
    *dst_meas.bins = 1;
    dst.add_measurement(dst_meas, "histogram", /*is_active=*/true);
    dst.add_measurement(counting_measurement {}, "stateless_meas");

    const json_serializer r { w.root() };
    simplemc_load_input_config(r["sim"], dst);

    // Weight and is_active from input config:
    EXPECT_DOUBLE_EQ(dst.update_data()[0].weight, 2.5);
    EXPECT_DOUBLE_EQ(dst.update_data()[1].weight, 1.0);
    EXPECT_FALSE(dst.measurement_data()[0].is_active);
    EXPECT_TRUE(dst.measurement_data()[1].is_active);

    // User-state round-trip on opt-in types:
    EXPECT_DOUBLE_EQ(*dst_threshold, 0.75);
    EXPECT_EQ(*dst_bins, 64);
}

TEST(MCSimulationInputConfig, OmitsStateOnlyFields) {
    simulation<> src { xoshiro256ss { 42 } };
    src.add_update(always_accept {}, "a", 1.0, always_accept {}, "b", 2.0); // pair: gets inv_name/ratio
    src.add_measurement(counting_measurement {}, "obs");
    src.run({ .max_steps = 5, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 5 });
    src.accumulate_stats();

    json_serializer w;
    auto w_sim = w["sim"];
    simplemc_save_input_config(w_sim, src);

    const auto& sim = w.root()["sim"];

    // No RNG, no cumulative simulation-stats fields:
    EXPECT_FALSE(sim.contains("rng"));
    EXPECT_FALSE(sim.contains("cumulative_steps"));
    EXPECT_FALSE(sim.contains("cumulative_time"));

    // Per-update: only "weight", no inv_name/ratio/cumulative_*:
    ASSERT_TRUE(sim.contains("updates"));
    const auto& a = sim["updates"]["a"];
    EXPECT_TRUE(a.contains("weight"));
    EXPECT_FALSE(a.contains("inv_name"));
    EXPECT_FALSE(a.contains("ratio"));
    EXPECT_FALSE(a.contains("cumulative_nprops"));
    EXPECT_FALSE(a.contains("cumulative_naccs"));
    EXPECT_FALSE(a.contains("cumulative_nimps"));

    // Per-measurement: only "is_active":
    ASSERT_TRUE(sim.contains("measurements"));
    const auto& obs = sim["measurements"]["obs"];
    EXPECT_TRUE(obs.contains("is_active"));
}

TEST(MCSimulationInputConfig, UserSlotSilentlySkippedForNonOptIn) {
    // Update / measurement without input-config hooks — the wrapper's
    // save_input_config_at silently no-ops, so "user" key is absent.
    simulation<> src;
    src.add_update(always_accept {}, "a", 1.0);
    src.add_measurement(counting_measurement {}, "obs");

    json_serializer w;
    auto w_sim = w["sim"];
    EXPECT_NO_THROW(simplemc_save_input_config(w_sim, src));

    const auto& a = w.root()["sim"]["updates"]["a"];
    EXPECT_FALSE(a.contains("user"));

    const auto& obs = w.root()["sim"]["measurements"]["obs"];
    EXPECT_FALSE(obs.contains("user"));

    // Loading the absent "user" slot must not throw either.
    simulation<> dst;
    dst.add_update(always_accept {}, "a", 1.0);
    dst.add_measurement(counting_measurement {}, "obs");

    const json_serializer r { w.root() };
    EXPECT_NO_THROW(simplemc_load_input_config(r["sim"], dst));
}

TEST(MCSimulationInputConfig, LoadThrowsOnMissingUpdate) {
    simulation<> src;
    src.add_update(always_accept {}, "a", 1.0);

    json_serializer w;
    auto w_sim = w["sim"];
    simplemc_save_input_config(w_sim, src);

    simulation<> dst;
    dst.add_update(always_accept {}, "b", 1.0); // different name

    const json_serializer r { w.root() };
    EXPECT_THROW(simplemc_load_input_config(r["sim"], dst), simplemc_exception);
}

TEST(MCSimulationInputConfig, LoadThrowsOnMissingMeasurement) {
    simulation<> src;
    src.add_update(always_accept {}, "a", 1.0);
    src.add_measurement(counting_measurement {}, "obs1");

    json_serializer w;
    auto w_sim = w["sim"];
    simplemc_save_input_config(w_sim, src);

    simulation<> dst;
    dst.add_update(always_accept {}, "a", 1.0);
    dst.add_measurement(counting_measurement {}, "obs2");

    const json_serializer r { w.root() };
    EXPECT_THROW(simplemc_load_input_config(r["sim"], dst), simplemc_exception);
}

TEST(MCSimulationInputConfig, ParamsAndSimLiveSideBySide) {
    // The canonical pattern: serialize params and sim under separate top-level keys.
    const simulation_params src_params { .max_steps = 1234,
        .max_time = 2.5,
        .steps_per_cycle = 7,
        .cycles_per_check = 13,
        .skip_measurements = true };
    simulation<> src;
    src.add_update(always_accept {}, "a", 3.0);

    json_serializer w;
    {
        auto w_params = w["params"];
        simplemc_save_input_config(w_params, src_params);
    }
    {
        auto w_sim = w["sim"];
        simplemc_save_input_config(w_sim, src);
    }

    simulation_params dst_params;
    simulation<> dst;
    dst.add_update(always_accept {}, "a", 1.0);

    const json_serializer r { w.root() };
    simplemc_load_input_config(r["params"], dst_params);
    simplemc_load_input_config(r["sim"], dst);

    EXPECT_EQ(dst_params.max_steps, 1234u);
    EXPECT_DOUBLE_EQ(dst_params.max_time, 2.5);
    EXPECT_EQ(dst_params.steps_per_cycle, 7u);
    EXPECT_EQ(dst_params.cycles_per_check, 13u);
    EXPECT_TRUE(dst_params.skip_measurements);
    EXPECT_DOUBLE_EQ(dst.update_data()[0].weight, 3.0);
}
