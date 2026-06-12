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
    void accept() {}
};

struct counting_measurement {
    void measure() {}
};

// User update that exposes only input-config hooks (no state simplemc_save).
struct configurable_update {
    std::shared_ptr<double> threshold = std::make_shared<double>(0.0);
    double attempt() { return 1.0; }
    void accept() {}
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
    void measure() {}
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

// Pull the underlying nlohmann document out of a JSON-backed mc_serializer for inspection / reload.
const nlohmann::json& doc(const mc_serializer& s) { return std::get<json_serializer>(s.backend()).root(); }

} // namespace

TEST(MCInputConfig, RoundTripPersistsParamsWeightActiveAndUserConfig) {
    const simulation_params src_params { .max_steps = 1234,
        .max_time = 2.5,
        .steps_per_cycle = 7,
        .cycles_per_check = 13,
        .skip_measurements = true };

    update_set updates;
    configurable_update src_up;
    *src_up.threshold = 0.75;
    updates.add({ src_up, "tunable", 2.5 });
    updates.add({ always_accept {}, "stateless", 1.0 });

    measurement_set meas;
    configurable_measurement src_meas;
    *src_meas.bins = 64;
    meas.add({ src_meas, "histogram", /*is_active=*/false });
    meas.add({ counting_measurement {}, "stateless_meas" });

    mc_serializer w { json_serializer {} };
    simplemc_save_input_config(w, src_params, updates, meas);

    // Destination: same registration, different initial values.
    simulation_params dst_params;
    update_set dst_updates;
    configurable_update dst_up;
    auto dst_threshold = dst_up.threshold;
    dst_updates.add({ dst_up, "tunable", 1.0 });
    dst_updates.add({ always_accept {}, "stateless", 1.0 });

    measurement_set dst_meas;
    configurable_measurement dst_meas_obj;
    auto dst_bins = dst_meas_obj.bins;
    dst_meas.add({ dst_meas_obj, "histogram", /*is_active=*/true });
    dst_meas.add({ counting_measurement {}, "stateless_meas" });

    const mc_serializer r { json_serializer { doc(w) } };
    simplemc_load_input_config(r, dst_params, dst_updates, dst_meas);

    // Params from input config:
    EXPECT_EQ(dst_params.max_steps, 1234u);
    EXPECT_DOUBLE_EQ(dst_params.max_time, 2.5);
    EXPECT_EQ(dst_params.steps_per_cycle, 7u);
    EXPECT_EQ(dst_params.cycles_per_check, 13u);
    EXPECT_TRUE(dst_params.skip_measurements);

    // Weight and is_active from input config:
    EXPECT_DOUBLE_EQ(dst_updates.data()[0].weight, 2.5);
    EXPECT_DOUBLE_EQ(dst_updates.data()[1].weight, 1.0);
    EXPECT_FALSE(dst_meas.data()[0].is_active);
    EXPECT_TRUE(dst_meas.data()[1].is_active);

    // User-state round-trip on opt-in types:
    EXPECT_DOUBLE_EQ(*dst_threshold, 0.75);
    EXPECT_EQ(*dst_bins, 64);
}

TEST(MCInputConfig, OmitsStateOnlyFields) {
    const simulation_params params;
    update_set updates;
    updates.add_pair({ always_accept {}, "a", 1.0 }, { always_accept {}, "b", 2.0 }); // gets inv_name/ratio
    measurement_set meas;
    meas.add({ counting_measurement {}, "obs" });

    mc_serializer w { json_serializer {} };
    simplemc_save_input_config(w, params, updates, meas);

    const auto& root = doc(w);

    // No RNG, no cumulative simulation-stats fields at the top level:
    EXPECT_FALSE(root.contains("rng"));
    EXPECT_FALSE(root.contains("cumulative_steps"));
    EXPECT_FALSE(root.contains("cumulative_time"));

    // Per-update: only "weight", no inv_name/ratio/cumulative_*:
    ASSERT_TRUE(root.contains("updates"));
    const auto& a = root["updates"]["a"];
    EXPECT_TRUE(a.contains("weight"));
    EXPECT_FALSE(a.contains("inv_name"));
    EXPECT_FALSE(a.contains("ratio"));
    EXPECT_FALSE(a.contains("cumulative_nprops"));
    EXPECT_FALSE(a.contains("cumulative_naccs"));
    EXPECT_FALSE(a.contains("cumulative_nimps"));

    // Per-measurement: only "is_active":
    ASSERT_TRUE(root.contains("measurements"));
    const auto& obs = root["measurements"]["obs"];
    EXPECT_TRUE(obs.contains("is_active"));
}

TEST(MCInputConfig, UserSlotSilentlySkippedForNonOptIn) {
    // Update / measurement without input-config hooks — the wrapper's
    // save_input_config_at silently no-ops, so "user" key is absent.
    const simulation_params params;
    update_set updates;
    updates.add({ always_accept {}, "a", 1.0 });
    measurement_set meas;
    meas.add({ counting_measurement {}, "obs" });

    mc_serializer w { json_serializer {} };
    EXPECT_NO_THROW(simplemc_save_input_config(w, params, updates, meas));

    EXPECT_FALSE(doc(w)["updates"]["a"].contains("user"));
    EXPECT_FALSE(doc(w)["measurements"]["obs"].contains("user"));

    // Loading the absent "user" slot must not throw either.
    simulation_params dst_params;
    update_set dst_updates;
    dst_updates.add({ always_accept {}, "a", 1.0 });
    measurement_set dst_meas;
    dst_meas.add({ counting_measurement {}, "obs" });

    const mc_serializer r { json_serializer { doc(w) } };
    EXPECT_NO_THROW(simplemc_load_input_config(r, dst_params, dst_updates, dst_meas));
}

TEST(MCInputConfig, LoadThrowsOnMissingUpdate) {
    const simulation_params params;
    update_set updates;
    updates.add({ always_accept {}, "a", 1.0 });
    measurement_set meas;

    mc_serializer w { json_serializer {} };
    simplemc_save_input_config(w, params, updates, meas);

    simulation_params dst_params;
    update_set dst_updates;
    dst_updates.add({ always_accept {}, "b", 1.0 }); // different name
    measurement_set dst_meas;

    const mc_serializer r { json_serializer { doc(w) } };
    EXPECT_THROW(simplemc_load_input_config(r, dst_params, dst_updates, dst_meas), simplemc_exception);
}

TEST(MCInputConfig, LoadThrowsOnMissingMeasurement) {
    const simulation_params params;
    update_set updates;
    updates.add({ always_accept {}, "a", 1.0 });
    measurement_set meas;
    meas.add({ counting_measurement {}, "obs1" });

    mc_serializer w { json_serializer {} };
    simplemc_save_input_config(w, params, updates, meas);

    simulation_params dst_params;
    update_set dst_updates;
    dst_updates.add({ always_accept {}, "a", 1.0 });
    measurement_set dst_meas;
    dst_meas.add({ counting_measurement {}, "obs2" });

    const mc_serializer r { json_serializer { doc(w) } };
    EXPECT_THROW(simplemc_load_input_config(r, dst_params, dst_updates, dst_meas), simplemc_exception);
}
