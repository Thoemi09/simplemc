#include <simplemc/mc.hpp>
#include <simplemc/random/xoshiro256.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

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
    double threshold = 0.0;
    double attempt() { return 1.0; }
    void accept() {}
};

template <serializer S>
void simplemc_save_input_config(S& s, const configurable_update& u) {
    s.save_at("threshold", u.threshold);
}

template <serializer S>
void simplemc_load_input_config(const S& s, configurable_update& u) {
    if (s.has("threshold")) {
        s.load_at("threshold", u.threshold);
    }
}

// User measurement with input-config hooks.
struct configurable_measurement {
    int bins = 1;
    void measure() {}
};

template <serializer S>
void simplemc_save_input_config(S& s, const configurable_measurement& m) {
    s.save_at("bins", m.bins);
}

template <serializer S>
void simplemc_load_input_config(const S& s, configurable_measurement& m) {
    if (s.has("bins")) {
        s.load_at("bins", m.bins);
    }
}

} // namespace

TEST(MCInputConfig, RoundTripPersistsParamsWeightActiveAndUserConfig) {
    const simulation_params src_params { .max_steps = 1234,
        .max_time = 2.5,
        .steps_per_cycle = 7,
        .cycles_per_check = 13,
        .skip_measurements = true };

    configurable_update src_up;
    src_up.threshold = 0.75;
    update_set updates { update { src_up, "tunable", 2.5 }, update { always_accept {}, "stateless", 1.0 } };

    configurable_measurement src_meas;
    src_meas.bins = 64;
    measurement_set meas { measurement { src_meas, "histogram", /*is_active=*/false },
        measurement { counting_measurement {}, "stateless_meas" } };

    json_serializer w;
    simplemc_save_input_config(w, src_params, updates, meas);

    // Destination: same registration, different initial values.
    simulation_params dst_params;
    update_set dst_updates { update { configurable_update {}, "tunable", 1.0 },
        update { always_accept {}, "stateless", 1.0 } };

    measurement_set dst_meas { measurement { configurable_measurement {}, "histogram", /*is_active=*/true },
        measurement { counting_measurement {}, "stateless_meas" } };

    const json_serializer r { w.root() };
    simplemc_load_input_config(r, dst_params, dst_updates, dst_meas);

    // Params from input config:
    EXPECT_EQ(dst_params.max_steps, 1234u);
    EXPECT_DOUBLE_EQ(dst_params.max_time, 2.5);
    EXPECT_EQ(dst_params.steps_per_cycle, 7u);
    EXPECT_EQ(dst_params.cycles_per_check, 13u);
    EXPECT_TRUE(dst_params.skip_measurements);

    // Weight and is_active from input config:
    EXPECT_DOUBLE_EQ(dst_updates.get<0>().weight, 2.5);
    EXPECT_DOUBLE_EQ(dst_updates.get<1>().weight, 1.0);
    EXPECT_FALSE(dst_meas.get<0>().is_active);
    EXPECT_TRUE(dst_meas.get<1>().is_active);

    // User-state round-trip on opt-in types:
    EXPECT_DOUBLE_EQ(dst_updates.get<0>().value.threshold, 0.75);
    EXPECT_EQ(dst_meas.get<0>().value.bins, 64);
}

TEST(MCInputConfig, OmitsStateOnlyFields) {
    const simulation_params params;
    update_set updates { update { always_accept {}, "a", 1.0 }, update { always_accept {}, "b", 2.0 } };
    updates.link_pair("a", "b"); // gets inv_name; ratio derived by rebuild_distribution
    measurement_set meas { measurement { counting_measurement {}, "obs" } };

    json_serializer w;
    simplemc_save_input_config(w, params, updates, meas);

    const auto& root = w.root();

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
    // Update / measurement without input-config hooks — the wrapper's save_input_config silently
    // no-ops, so the "user" key is absent.
    const simulation_params params;
    update_set updates { update { always_accept {}, "a", 1.0 } };
    measurement_set meas { measurement { counting_measurement {}, "obs" } };

    json_serializer w;
    EXPECT_NO_THROW(simplemc_save_input_config(w, params, updates, meas));

    EXPECT_FALSE(w.root()["updates"]["a"].contains("user"));
    EXPECT_FALSE(w.root()["measurements"]["obs"].contains("user"));

    // Loading the absent "user" slot must not throw either.
    simulation_params dst_params;
    update_set dst_updates { update { always_accept {}, "a", 1.0 } };
    measurement_set dst_meas { measurement { counting_measurement {}, "obs" } };

    const json_serializer r { w.root() };
    EXPECT_NO_THROW(simplemc_load_input_config(r, dst_params, dst_updates, dst_meas));
}

TEST(MCInputConfig, LoadThrowsOnMissingUpdate) {
    const simulation_params params;
    update_set updates { update { always_accept {}, "a", 1.0 } };
    measurement_set<> meas;

    json_serializer w;
    simplemc_save_input_config(w, params, updates, meas);

    simulation_params dst_params;
    update_set dst_updates { update { always_accept {}, "b", 1.0 } }; // different name
    measurement_set<> dst_meas;

    const json_serializer r { w.root() };
    EXPECT_THROW(simplemc_load_input_config(r, dst_params, dst_updates, dst_meas), simplemc_exception);
}

TEST(MCInputConfig, LoadThrowsOnMissingMeasurement) {
    const simulation_params params;
    update_set updates { update { always_accept {}, "a", 1.0 } };
    measurement_set meas { measurement { counting_measurement {}, "obs1" } };

    json_serializer w;
    simplemc_save_input_config(w, params, updates, meas);

    simulation_params dst_params;
    update_set dst_updates { update { always_accept {}, "a", 1.0 } };
    measurement_set dst_meas { measurement { counting_measurement {}, "obs2" } };

    const json_serializer r { w.root() };
    EXPECT_THROW(simplemc_load_input_config(r, dst_params, dst_updates, dst_meas), simplemc_exception);
}
