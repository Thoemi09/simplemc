#include <simplemc/mc.hpp>
#include <simplemc/random/xoshiro256.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <memory>

using namespace simplemc;

namespace {

struct always_accept {
    double attempt() { return 1.0; }
    void accept() {}
};

struct counting_measurement {
    std::shared_ptr<int> count = std::make_shared<int>(0);
    void measure() { ++*count; }
};

// Stateful update with an ADL simplemc_save / simplemc_load that round-trips an internal counter.
struct stateful_update {
    std::shared_ptr<int> counter = std::make_shared<int>(0);
    double attempt() {
        ++*counter;
        return 1.0;
    }
    void accept() {}
};

template <serializer S>
void simplemc_save(S& s, const stateful_update& u) {
    s.save_at("counter", *u.counter);
}

template <serializer S>
void simplemc_load(const S& s, stateful_update& u) {
    s.load_at("counter", *u.counter);
}

// Stateful measurement with ADL save/load.
struct stateful_measurement {
    std::shared_ptr<int> tally = std::make_shared<int>(0);
    void measure() { ++*tally; }
};

template <serializer S>
void simplemc_save(S& s, const stateful_measurement& m) {
    s.save_at("tally", *m.tally);
}

template <serializer S>
void simplemc_load(const S& s, stateful_measurement& m) {
    s.load_at("tally", *m.tally);
}

// User update with only nlohmann to_json / from_json (no simplemc_save). Because the mc types now
// serialize through whatever concrete serializer they are handed (here json_serializer), the
// nlohmann-native fallback works regardless of whether HDF5 is compiled in.
struct nlohmann_only_update {
    int counter = 0;
    double attempt() {
        ++counter;
        return 1.0;
    }
    void accept() {}
};
inline void to_json(nlohmann::json& j, const nlohmann_only_update& u) {
    j = nlohmann::json { { "counter", u.counter } };
}
inline void from_json(const nlohmann::json& j, nlohmann_only_update& u) { j.at("counter").get_to(u.counter); }

// Drive a short run over the assembled components, then fold counters into the cumulative state.
template <typename UpdateSet, typename MeasSet>
void run_and_accumulate(
    UpdateSet& updates, MeasSet& meas, simulation_stats& stats, xoshiro256ss& rng, const simulation_params& p) {
    metropolis_kernel kernel { updates };
    const auto ctx = run(rng, kernel, meas, p);
    accumulate_simulation_stats(stats, ctx);
    updates.accumulate_counters();
}

} // namespace

TEST(MCCheckpoint, JsonRoundTripPersistsCumulativeAndUserState) {
    xoshiro256ss rng { 42 };
    simulation_stats stats;

    stateful_update stateful;
    auto stateful_counter = stateful.counter;
    stateful_measurement m;
    auto m_tally = m.tally;
    update_set updates { update { stateful, "stateful", 2.5 }, update { always_accept {}, "stateless", 1.0 } };
    measurement_set meas { measurement { m, "obs" }, measurement { counting_measurement {}, "stateless_meas" } };

    run_and_accumulate(updates, meas, stats, rng,
        { .max_steps = 50, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 10 });
    const int src_counter = *stateful_counter;
    const int src_tally = *m_tally;
    ASSERT_GT(stats.cumulative_steps, 0u);
    ASSERT_GT(src_counter, 0);
    ASSERT_GT(src_tally, 0);

    json_serializer writer;
    simplemc_save(writer, rng, updates, meas, stats);

    // Destination: same structure, different RNG seed.
    xoshiro256ss dst_rng { 999 };
    simulation_stats dst_stats;
    stateful_update dst_stateful;
    auto dst_stateful_counter = dst_stateful.counter;
    stateful_measurement dst_m;
    auto dst_m_tally = dst_m.tally;
    update_set dst_updates { update { dst_stateful, "stateful", 2.5 },
        update { always_accept {}, "stateless", 1.0 } };
    measurement_set dst_meas { measurement { dst_m, "obs" },
        measurement { counting_measurement {}, "stateless_meas" } };

    const json_serializer reader { writer.root() };
    simplemc_load(reader, dst_rng, dst_updates, dst_meas, dst_stats);

    // Persistent fields propagated.
    EXPECT_EQ(dst_stats.cumulative_steps, stats.cumulative_steps);
    EXPECT_DOUBLE_EQ(dst_stats.cumulative_time, stats.cumulative_time);
    EXPECT_DOUBLE_EQ(dst_updates.at<0>().weight, 2.5);
    EXPECT_DOUBLE_EQ(dst_updates.at<1>().weight, 1.0);
    EXPECT_EQ(dst_updates.at<0>().cumulative_nprops, updates.at<0>().cumulative_nprops);
    EXPECT_EQ(dst_updates.at<0>().inv_name, updates.at<0>().inv_name);
    EXPECT_DOUBLE_EQ(dst_updates.at<0>().ratio, updates.at<0>().ratio);
    EXPECT_EQ(dst_meas.at<0>().is_active, meas.at<0>().is_active);

    // Per-run update counters are zero on dst.
    EXPECT_EQ(dst_updates.at<0>().nprops, 0u);

    // User-state round-trip via simplemc_save / simplemc_load.
    EXPECT_EQ(*dst_stateful_counter, src_counter);
    EXPECT_EQ(*dst_m_tally, src_tally);

    // RNG continuation: a draw from dst matches a draw from src.
    EXPECT_EQ(dst_rng(), rng());
}

TEST(MCCheckpoint, JsonLoadThrowsOnMissingUpdate) {
    xoshiro256ss rng { 1 };
    simulation_stats stats;
    update_set updates { update { always_accept {}, "a", 1.0 } };
    measurement_set<> meas;
    run_and_accumulate(updates, meas, stats, rng,
        { .max_steps = 5, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 5 });

    json_serializer writer;
    simplemc_save(writer, rng, updates, meas, stats);

    xoshiro256ss dst_rng { 2 };
    simulation_stats dst_stats;
    update_set dst_updates { update { always_accept {}, "b", 1.0 } }; // different name
    measurement_set<> dst_meas;

    const json_serializer reader { writer.root() };
    EXPECT_THROW(simplemc_load(reader, dst_rng, dst_updates, dst_meas, dst_stats), simplemc_exception);
}

TEST(MCCheckpoint, JsonLoadThrowsOnMissingMeasurement) {
    xoshiro256ss rng { 1 };
    simulation_stats stats;
    update_set updates { update { always_accept {}, "a", 1.0 } };
    measurement_set meas { measurement { counting_measurement {}, "obs1" } };
    run_and_accumulate(updates, meas, stats, rng,
        { .max_steps = 5, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 5 });

    json_serializer writer;
    simplemc_save(writer, rng, updates, meas, stats);

    xoshiro256ss dst_rng { 2 };
    simulation_stats dst_stats;
    update_set dst_updates { update { always_accept {}, "a", 1.0 } };
    measurement_set dst_meas { measurement { counting_measurement {}, "different_name" } };

    const json_serializer reader { writer.root() };
    EXPECT_THROW(simplemc_load(reader, dst_rng, dst_updates, dst_meas, dst_stats), simplemc_exception);
}

TEST(MCCheckpoint, JsonRoundTripWorksWithStatelessUserTypes) {
    // Stateless user types have neither simplemc_save nor nlohmann::to_json — the wrapper's
    // save_at silently no-ops. Round-trip still succeeds.
    xoshiro256ss rng { 1 };
    simulation_stats stats;
    update_set updates { update { always_accept {}, "aa", 1.0 } };
    measurement_set meas { measurement { counting_measurement {}, "obs" } };
    run_and_accumulate(updates, meas, stats, rng,
        { .max_steps = 5, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 5 });

    json_serializer writer;
    EXPECT_NO_THROW(simplemc_save(writer, rng, updates, meas, stats));

    xoshiro256ss dst_rng { 2 };
    simulation_stats dst_stats;
    update_set dst_updates { update { always_accept {}, "aa", 1.0 } };
    measurement_set dst_meas { measurement { counting_measurement {}, "obs" } };

    const json_serializer reader { writer.root() };
    EXPECT_NO_THROW(simplemc_load(reader, dst_rng, dst_updates, dst_meas, dst_stats));
    EXPECT_EQ(dst_stats.cumulative_steps, stats.cumulative_steps);
}

TEST(MCCheckpoint, JsonRoundTripWorksWithNlohmannToJsonUserTypes) {
    // User type with only nlohmann to_json / from_json (no simplemc_save). Serializing through
    // json_serializer engages nlohmann's native machinery for the payload.
    xoshiro256ss rng { 1 };
    simulation_stats stats;
    update_set updates { update { nlohmann_only_update {}, "nlh", 1.0 } };
    measurement_set<> meas;
    run_and_accumulate(updates, meas, stats, rng,
        { .max_steps = 10, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 5 });

    json_serializer writer;
    simplemc_save(writer, rng, updates, meas, stats);

    // Confirm the JSON has a "user" sub-object with the counter populated (i.e., the wrapper
    // fell through to nlohmann's to_json for the user type).
    const auto& root = writer.root();
    ASSERT_TRUE(root.contains("updates"));
    ASSERT_TRUE(root["updates"].contains("nlh"));
    ASSERT_TRUE(root["updates"]["nlh"].contains("user"));
    EXPECT_GT(root["updates"]["nlh"]["user"]["counter"].get<int>(), 0);

    xoshiro256ss dst_rng { 2 };
    simulation_stats dst_stats;
    update_set dst_updates { update { nlohmann_only_update {}, "nlh", 1.0 } };
    measurement_set<> dst_meas;

    const json_serializer reader { writer.root() };
    EXPECT_NO_THROW(simplemc_load(reader, dst_rng, dst_updates, dst_meas, dst_stats));
    EXPECT_EQ(dst_stats.cumulative_steps, stats.cumulative_steps);
}

TEST(MCCheckpoint, FileRoundTripViaSaveCheckpoint) {
    // Exercise the file-based dispatcher (backend deduced from the .json extension) plus atomic write.
    xoshiro256ss rng { 7 };
    simulation_stats stats;
    stateful_update stateful;
    auto stateful_counter = stateful.counter;
    update_set updates { update { stateful, "stateful", 1.0 } };
    measurement_set<> meas;
    run_and_accumulate(updates, meas, stats, rng,
        { .max_steps = 20, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 5 });
    const int src_counter = *stateful_counter;

    const auto path = std::filesystem::temp_directory_path() / "mc_checkpoint_file_roundtrip.json";
    save_checkpoint(path, rng, updates, meas, stats);

    xoshiro256ss dst_rng { 123 };
    simulation_stats dst_stats;
    stateful_update dst_stateful;
    auto dst_counter = dst_stateful.counter;
    update_set dst_updates { update { dst_stateful, "stateful", 1.0 } };
    measurement_set<> dst_meas;
    load_checkpoint(path, dst_rng, dst_updates, dst_meas, dst_stats);

    EXPECT_EQ(dst_stats.cumulative_steps, stats.cumulative_steps);
    EXPECT_EQ(*dst_counter, src_counter);
    EXPECT_EQ(dst_rng(), rng()); // RNG continuation

    std::filesystem::remove(path);
}
