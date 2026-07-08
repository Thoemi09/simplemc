#include "./mc_test_utils.hpp"

#include <simplemc/mc.hpp>
#include <simplemc/random/seed_rng.hpp>
#include <simplemc/random/xoshiro256.hpp>
#include <simplemc/serialize/json/file_io.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <string>

using namespace simplemc;

namespace {

// User update that exposes only input-config hooks (no state simplemc_save).
struct configurable_update {
    double threshold = 0.0;
    double attempt() { return 1.0; }
    void accept() {}
};

template <serializer S>
void simplemc_save_input_config(S s, const configurable_update& u) {
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
void simplemc_save_input_config(S s, const configurable_measurement& m) {
    s.save_at("bins", m.bins);
}

template <serializer S>
void simplemc_load_input_config(const S& s, configurable_measurement& m) {
    if (s.has("bins")) {
        s.load_at("bins", m.bins);
    }
}

// User update with ONLY the checkpoint hooks (simplemc_save / simplemc_load) — the input-config
// channel must not fall through to them.
struct state_only_update {
    int ticks = 0;
    double attempt() { return 1.0; }
    void accept() {}
};

template <serializer S>
void simplemc_save(S s, const state_only_update& u) {
    s.save_at("ticks", u.ticks);
}

template <serializer S>
void simplemc_load(const S& s, state_only_update& u) {
    s.load_at("ticks", u.ticks);
}

// Serializable user-state object attached to an input config via the config overloads.
struct run_config {
    int seed = 0;
    double beta = 0.0;

    template <serializer S>
    friend void simplemc_save(S s, const run_config& c) {
        s.save_at("seed", c.seed);
        s.save_at("beta", c.beta);
    }
    template <serializer S>
    friend void simplemc_load(const S& s, run_config& c) {
        s.load_at("seed", c.seed);
        s.load_at("beta", c.beta);
    }
};

} // namespace

// Test that the composite save/load input-config round-trip preserves the simulation params, update
// weights, measurements activation state, and user-state for opt-in types.
TEST(MCInputConfig, RoundTripPersistsParamsWeightActiveAndUserConfig) {
    // prepare sources
    const simulation_params src_params {
        .max_steps = 1234, .max_time = 2.5, .steps_per_cycle = 7, .cycles_per_check = 13, .skip_measurements = true
    };

    configurable_update src_up;
    src_up.threshold = 0.75;
    update_set updates { update { src_up, "tunable", 2.5 }, update { dummy_update {}, "stateless", 1.0 } };

    configurable_measurement src_meas;
    src_meas.bins = 64;
    measurement_set meas { measurement { src_meas, "histogram", /*is_active=*/false },
        measurement { dummy_measurement {}, "stateless_meas" } };

    // save to serializer
    json_serializer w;
    simplemc_save_input_config(w, src_params, updates, meas);

    // prepare destinations
    simulation_params dst_params;
    update_set dst_updates { update { configurable_update {}, "tunable", 1.0 },
        update { dummy_update {}, "stateless", 1.0 } };

    measurement_set dst_meas { measurement { configurable_measurement {}, "histogram", /*is_active=*/true },
        measurement { dummy_measurement {}, "stateless_meas" } };

    // load from serializer
    const json_serializer r { w.root() };
    simplemc_load_input_config(r, dst_params, dst_updates, dst_meas);

    // check params
    EXPECT_EQ(dst_params.max_steps, 1234u);
    EXPECT_DOUBLE_EQ(dst_params.max_time, 2.5);
    EXPECT_EQ(dst_params.steps_per_cycle, 7u);
    EXPECT_EQ(dst_params.cycles_per_check, 13u);
    EXPECT_TRUE(dst_params.skip_measurements);

    // check update weights and measurement activation state
    EXPECT_DOUBLE_EQ(dst_updates.get<0>().stats().weight, 2.5);
    EXPECT_DOUBLE_EQ(dst_updates.get<1>().stats().weight, 1.0);
    EXPECT_FALSE(dst_meas.get<0>().is_active());
    EXPECT_TRUE(dst_meas.get<1>().is_active());

    // check user-specific state of updates and measurements
    EXPECT_DOUBLE_EQ(dst_updates.get<0>().value().threshold, 0.75);
    EXPECT_EQ(dst_meas.get<0>().value().bins, 64);
}

// Test that the save hooks take the serializer handle by value, so rvalue sub-handles returned by
// operator[] bind directly without a named sub-serializer variable.
TEST(MCInputConfig, SaveHooksBindRvalueSubHandles) {
    // prepare sources
    const simulation_params src_params {
        .max_steps = 42, .max_time = 1.5, .steps_per_cycle = 3, .cycles_per_check = 9
    };
    configurable_update src_up;
    src_up.threshold = 0.25;
    const run_config src_cfg { .seed = 7, .beta = 0.5 };

    // save each piece through an rvalue sub-handle (both channels)
    json_serializer w;
    simplemc_save_input_config(w["params"], src_params);
    simplemc_save_input_config(w["tunable"], src_up);
    simplemc_save(w["config"], src_cfg);

    // prepare destinations
    const json_serializer r { w.root() };
    simulation_params dst_params;
    configurable_update dst_up;
    run_config dst_cfg;

    // load back through rvalue sub-handles as well (const S& binds them on the load side)
    simplemc_load_input_config(r["params"], dst_params);
    simplemc_load_input_config(r["tunable"], dst_up);
    simplemc_load(r["config"], dst_cfg);

    // check the round-trip against the sources
    EXPECT_EQ(dst_params.max_steps, 42u);
    EXPECT_DOUBLE_EQ(dst_params.max_time, 1.5);
    EXPECT_EQ(dst_params.steps_per_cycle, 3u);
    EXPECT_EQ(dst_params.cycles_per_check, 9u);
    EXPECT_DOUBLE_EQ(dst_up.threshold, 0.25);
    EXPECT_EQ(dst_cfg.seed, 7);
    EXPECT_DOUBLE_EQ(dst_cfg.beta, 0.5);
}

// Test that the composite save/load input-config routines only expose input-config fields.
TEST(MCInputConfig, OmitsStateOnlyFields) {
    // prepare sources
    const simulation_params params;
    update_set updates { update { dummy_update {}, "a", 1.0 }, update { dummy_update {}, "b", 2.0 } };
    updates.link_pair("a", "b");
    measurement_set meas { measurement { dummy_measurement {}, "obs" } };

    // save to serializer and get nlohmann::json root
    json_serializer w;
    simplemc_save_input_config(w, params, updates, meas);
    const auto& root = w.root();

    // RNG state is non input-config
    EXPECT_FALSE(root.contains("rng"));
    EXPECT_FALSE(root.contains("cumulative_steps"));
    EXPECT_FALSE(root.contains("cumulative_time"));

    // only the update weight is input-config
    ASSERT_TRUE(root.contains("updates"));
    const auto& a = root["updates"]["a"];
    EXPECT_TRUE(a.contains("weight"));
    EXPECT_FALSE(a.contains("inv_name"));
    EXPECT_FALSE(a.contains("ratio"));
    EXPECT_FALSE(a.contains("nprops"));
    EXPECT_FALSE(a.contains("naccs"));
    EXPECT_FALSE(a.contains("nimps"));

    // measurement activation state is input-config
    ASSERT_TRUE(root.contains("measurements"));
    const auto& obs = root["measurements"]["obs"];
    EXPECT_TRUE(obs.contains("is_active"));
}

// Test that the composite save/load input-config routines silently skip the "user" slot for
// non-opt-in types.
TEST(MCInputConfig, UserSlotSilentlySkippedForNonOptIn) {
    // prepare sources
    const simulation_params params;
    update_set updates { update { dummy_update {}, "a", 1.0 } };
    measurement_set meas { measurement { dummy_measurement {}, "obs" } };

    // save to serializer
    json_serializer w;
    EXPECT_NO_THROW(simplemc_save_input_config(w, params, updates, meas));

    // no "user" slots
    EXPECT_FALSE(w.root()["updates"]["a"].contains("user"));
    EXPECT_FALSE(w.root()["measurements"]["obs"].contains("user"));

    // prepare destinations
    simulation_params dst_params;
    update_set dst_updates { update { dummy_update {}, "a", 1.0 } };
    measurement_set dst_meas { measurement { dummy_measurement {}, "obs" } };

    // loading the absent "user" slot must not throw
    const json_serializer r { w.root() };
    EXPECT_NO_THROW(simplemc_load_input_config(r, dst_params, dst_updates, dst_meas));
}

// Test that the composite save/load input-config routines skip the "user" slot for a type that opts
// into only the checkpoint channel: its simplemc_save must not be picked up here.
TEST(MCInputConfig, UserSlotSkippedForStateOnlyOptIn) {
    // prepare sources
    const simulation_params params;
    state_only_update u;
    u.ticks = 99;
    update_set updates { update { u, "ticker", 1.0 } };
    measurement_set<> meas;

    // save to serializer
    json_serializer w;
    EXPECT_NO_THROW(simplemc_save_input_config(w, params, updates, meas));

    // per-entry "weight" is still written (it's on the update entry, not the user value), but the
    // user slot is empty because state_only_update doesn't opt into input-config
    const auto& ticker = w.root()["updates"]["ticker"];
    EXPECT_TRUE(ticker.contains("weight"));
    EXPECT_FALSE(ticker.contains("user"));

    // loading leaves the user value untouched
    simulation_params dst_params;
    update_set dst_updates { update { state_only_update {}, "ticker", 1.0 } };
    measurement_set<> dst_meas;
    const json_serializer r { w.root() };
    EXPECT_NO_THROW(simplemc_load_input_config(r, dst_params, dst_updates, dst_meas));
    EXPECT_EQ(dst_updates.get<0>().value().ticks, 0);
}

// Test that the composite load input-config routine throws if an update or measurement is missing.
TEST(MCInputConfig, LoadThrowsOnMissingUpdate) {
    const simulation_params params;
    update_set updates { update { dummy_update {}, "a", 1.0 } };
    measurement_set<> meas;

    json_serializer w;
    simplemc_save_input_config(w, params, updates, meas);

    simulation_params dst_params;
    update_set dst_updates { update { dummy_update {}, "b", 1.0 } }; // different name
    measurement_set<> dst_meas;

    const json_serializer r { w.root() };
    EXPECT_THROW(simplemc_load_input_config(r, dst_params, dst_updates, dst_meas), simplemc_exception);
}

TEST(MCInputConfig, LoadThrowsOnMissingMeasurement) {
    const simulation_params params;
    update_set updates { update { dummy_update {}, "a", 1.0 } };
    measurement_set meas { measurement { dummy_measurement {}, "obs1" } };

    json_serializer w;
    simplemc_save_input_config(w, params, updates, meas);

    simulation_params dst_params;
    update_set dst_updates { update { dummy_update {}, "a", 1.0 } };
    measurement_set dst_meas { measurement { dummy_measurement {}, "obs2" } };

    const json_serializer r { w.root() };
    EXPECT_THROW(simplemc_load_input_config(r, dst_params, dst_updates, dst_meas), simplemc_exception);
}

// Test writing/reading input-config to/from a file.
TEST(MCInputConfig, FileRoundTrip) {
    // prepare sources
    const simulation_params src_params { .max_steps = 500, .steps_per_cycle = 3, .cycles_per_check = 11 };

    configurable_update src_up;
    src_up.threshold = 0.25;
    update_set updates { update { src_up, "tunable", 4.0 } };

    configurable_measurement src_meas;
    src_meas.bins = 32;
    measurement_set meas { measurement { src_meas, "histogram", /*is_active=*/false } };

    // write to a temporary file
    const auto path = std::filesystem::temp_directory_path() / "mc_input_config_file_roundtrip.json";
    {
        json_serializer s;
        simplemc_save_input_config(s, src_params, updates, meas);
        write_json_file(s.root(), path, { .indent = 4 });
    }

    // prepare destinations
    simulation_params dst_params;
    update_set dst_updates { update { configurable_update {}, "tunable", 1.0 } };
    measurement_set dst_meas { measurement { configurable_measurement {}, "histogram" } };

    // read from the file
    json_serializer r;
    read_json_file(r.root(), path);
    simplemc_load_input_config(r, dst_params, dst_updates, dst_meas);

    // check that the round-trip preserved the input-config fields
    EXPECT_EQ(dst_params.max_steps, 500u);
    EXPECT_EQ(dst_params.steps_per_cycle, 3u);
    EXPECT_EQ(dst_params.cycles_per_check, 11u);
    EXPECT_DOUBLE_EQ(dst_updates.get<0>().stats().weight, 4.0);
    EXPECT_FALSE(dst_meas.get<0>().is_active());
    EXPECT_DOUBLE_EQ(dst_updates.get<0>().value().threshold, 0.25);
    EXPECT_EQ(dst_meas.get<0>().value().bins, 32);

    std::filesystem::remove(path);
}

TEST(MCInputConfig, FileRoundTripWithConfig) {
    // prepare sources
    const simulation_params src_params { .max_steps = 100 };
    update_set updates { update { dummy_update {}, "a", 1.0 } };
    measurement_set<> meas;
    const run_config src_config { .seed = 42, .beta = 1.5 };

    // write to a temporary file
    const auto path = std::filesystem::temp_directory_path() / "mc_input_config_file_config.json";
    {
        json_serializer s;
        simplemc_save_input_config(s, src_params, updates, meas);
        s.save_at("config", src_config);
        write_json_file(s.root(), path, { .indent = 2 });
    }

    // prepare destinations
    simulation_params dst_params;
    update_set dst_updates { update { dummy_update {}, "a", 1.0 } };
    measurement_set<> dst_meas;
    run_config dst_config;

    // read from the file
    json_serializer r;
    read_json_file(r.root(), path);
    simplemc_load_input_config(r, dst_params, dst_updates, dst_meas);
    r.try_load_at("config", dst_config);

    // check that the round-trip preserved the input-config fields and user config
    EXPECT_EQ(dst_params.max_steps, 100u);
    EXPECT_EQ(dst_config.seed, 42);
    EXPECT_DOUBLE_EQ(dst_config.beta, 1.5);

    std::filesystem::remove(path);
}

// Test that loading an input-config file with only a subset of the fields is allowed.
TEST(MCInputConfig, ReadToleratesPartialFile) {
    // write a partial input-config file
    const auto path = std::filesystem::temp_directory_path() / "mc_input_config_partial.json";
    {
        std::ofstream ofs { path };
        ofs << R"({ "params": { "max_steps": 77 } })";
    }

    // prepare destinations
    simulation_params params;
    update_set updates { update { configurable_update {}, "tunable", 3.0 } };
    measurement_set meas { measurement { configurable_measurement {}, "histogram", /*is_active=*/false } };
    run_config config { .seed = 7, .beta = 0.5 };

    // read from the file
    json_serializer r;
    read_json_file(r.root(), path);
    EXPECT_NO_THROW(simplemc_load_input_config(r, params, updates, meas));
    EXPECT_FALSE(r.try_load_at("config", config));

    // check that the round-trip preserved the fields that were present in the file and left the
    // others unchanged
    EXPECT_EQ(params.max_steps, 77u);
    EXPECT_DOUBLE_EQ(updates.get<0>().stats().weight, 3.0);
    EXPECT_FALSE(meas.get<0>().is_active());
    EXPECT_EQ(config.seed, 7);
    EXPECT_DOUBLE_EQ(config.beta, 0.5);

    std::filesystem::remove(path);
}

// Test that reading a non-existent input-config file throws an exception.
TEST(MCInputConfig, ReadThrowsOnMissingFile) {
    const auto path = std::filesystem::temp_directory_path() / "mc_input_config_does_not_exist.json";
    json_serializer r;
    EXPECT_THROW(read_json_file(r.root(), path), simplemc_exception);
}
