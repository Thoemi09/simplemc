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

#ifndef SIMPLEMC_USE_HDF5
// User update that has only nlohmann to_json / from_json (no simplemc_save). The to_json fallback
// only engages when JSON is the sole backend in mc_serializer: a variant_serializer that also
// includes HDF5 requires the payload be serializable by *every* backend (the capability is the
// intersection), so this scenario is meaningful only when HDF5 is not compiled in.
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
#endif // SIMPLEMC_USE_HDF5

// Pull the underlying nlohmann document out of a JSON-backed mc_serializer for inspection / reload.
const nlohmann::json& doc(const mc_serializer& s) { return std::get<json_serializer>(s.backend()).root(); }

// Drive a short run over the assembled components, then fold counters into the cumulative state.
void run_and_accumulate(update_set& updates, measurement_set& meas, simulation_stats& stats,
    xoshiro256ss& rng, const simulation_params& p) {
    metropolis_kernel kernel { updates };
    const auto ctx = run(rng, kernel, meas, p);
    accumulate_simulation_stats(stats, ctx);
    updates.accumulate_counters();
}

} // namespace

TEST(MCCheckpoint, JsonRoundTripPersistsCumulativeAndConfig) {
    xoshiro256ss rng { 42 };
    update_set updates;
    measurement_set meas;
    simulation_stats stats;

    stateful_update stateful;
    auto stateful_counter = stateful.counter;
    updates.add({ stateful, "stateful", 2.5 });
    updates.add({ always_accept {}, "stateless", 1.0 });
    stateful_measurement m;
    auto m_tally = m.tally;
    meas.add({ m, "obs" });
    meas.add({ counting_measurement {}, "stateless_meas" });

    run_and_accumulate(updates, meas, stats, rng,
        { .max_steps = 50, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 10 });
    const int src_counter = *stateful_counter;
    const int src_tally = *m_tally;
    ASSERT_GT(stats.cumulative_steps, 0u);
    ASSERT_GT(src_counter, 0);
    ASSERT_GT(src_tally, 0);

    mc_serializer writer { json_serializer {} };
    simplemc_save(writer, rng, updates, meas, stats);

    // Destination: same structure, different RNG seed.
    xoshiro256ss dst_rng { 999 };
    update_set dst_updates;
    measurement_set dst_meas;
    simulation_stats dst_stats;
    stateful_update dst_stateful;
    auto dst_stateful_counter = dst_stateful.counter;
    dst_updates.add({ dst_stateful, "stateful", 2.5 });
    dst_updates.add({ always_accept {}, "stateless", 1.0 });
    stateful_measurement dst_m;
    auto dst_m_tally = dst_m.tally;
    dst_meas.add({ dst_m, "obs" });
    dst_meas.add({ counting_measurement {}, "stateless_meas" });

    const mc_serializer reader { json_serializer { doc(writer) } };
    simplemc_load(reader, dst_rng, dst_updates, dst_meas, dst_stats);

    // Persistent fields propagated.
    EXPECT_EQ(dst_stats.cumulative_steps, stats.cumulative_steps);
    EXPECT_DOUBLE_EQ(dst_stats.cumulative_time, stats.cumulative_time);
    EXPECT_DOUBLE_EQ(dst_updates.data()[0].weight, 2.5);
    EXPECT_DOUBLE_EQ(dst_updates.data()[1].weight, 1.0);
    EXPECT_EQ(dst_updates.data()[0].cumulative_nprops, updates.data()[0].cumulative_nprops);
    EXPECT_EQ(dst_updates.data()[0].inv_name, updates.data()[0].inv_name);
    EXPECT_DOUBLE_EQ(dst_updates.data()[0].ratio, updates.data()[0].ratio);
    EXPECT_EQ(dst_meas.data()[0].is_active, meas.data()[0].is_active);

    // Per-run update counters are zero on dst.
    EXPECT_EQ(dst_updates.data()[0].nprops, 0u);

    // User-state round-trip via simplemc_save / simplemc_load.
    EXPECT_EQ(*dst_stateful_counter, src_counter);
    EXPECT_EQ(*dst_m_tally, src_tally);

    // RNG continuation: a draw from dst matches a draw from src.
    EXPECT_EQ(dst_rng(), rng());
}

TEST(MCCheckpoint, JsonLoadThrowsOnMissingUpdate) {
    xoshiro256ss rng { 1 };
    update_set updates;
    measurement_set meas;
    simulation_stats stats;
    updates.add({ always_accept {}, "a", 1.0 });
    run_and_accumulate(updates, meas, stats, rng,
        { .max_steps = 5, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 5 });

    mc_serializer writer { json_serializer {} };
    simplemc_save(writer, rng, updates, meas, stats);

    xoshiro256ss dst_rng { 2 };
    update_set dst_updates;
    measurement_set dst_meas;
    simulation_stats dst_stats;
    dst_updates.add({ always_accept {}, "b", 1.0 }); // different name

    const mc_serializer reader { json_serializer { doc(writer) } };
    EXPECT_THROW(simplemc_load(reader, dst_rng, dst_updates, dst_meas, dst_stats), simplemc_exception);
}

TEST(MCCheckpoint, JsonLoadThrowsOnMissingMeasurement) {
    xoshiro256ss rng { 1 };
    update_set updates;
    measurement_set meas;
    simulation_stats stats;
    updates.add({ always_accept {}, "a", 1.0 });
    meas.add({ counting_measurement {}, "obs1" });
    run_and_accumulate(updates, meas, stats, rng,
        { .max_steps = 5, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 5 });

    mc_serializer writer { json_serializer {} };
    simplemc_save(writer, rng, updates, meas, stats);

    xoshiro256ss dst_rng { 2 };
    update_set dst_updates;
    measurement_set dst_meas;
    simulation_stats dst_stats;
    dst_updates.add({ always_accept {}, "a", 1.0 });
    dst_meas.add({ counting_measurement {}, "different_name" });

    const mc_serializer reader { json_serializer { doc(writer) } };
    EXPECT_THROW(simplemc_load(reader, dst_rng, dst_updates, dst_meas, dst_stats), simplemc_exception);
}

TEST(MCCheckpoint, JsonRoundTripWorksWithStatelessUserTypes) {
    // Stateless user types have neither simplemc_save nor nlohmann::to_json — the wrapper's
    // save_at silently no-ops. Round-trip still succeeds.
    xoshiro256ss rng { 1 };
    update_set updates;
    measurement_set meas;
    simulation_stats stats;
    updates.add({ always_accept {}, "aa", 1.0 });
    meas.add({ counting_measurement {}, "obs" });
    run_and_accumulate(updates, meas, stats, rng,
        { .max_steps = 5, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 5 });

    mc_serializer writer { json_serializer {} };
    EXPECT_NO_THROW(simplemc_save(writer, rng, updates, meas, stats));

    xoshiro256ss dst_rng { 2 };
    update_set dst_updates;
    measurement_set dst_meas;
    simulation_stats dst_stats;
    dst_updates.add({ always_accept {}, "aa", 1.0 });
    dst_meas.add({ counting_measurement {}, "obs" });

    const mc_serializer reader { json_serializer { doc(writer) } };
    EXPECT_NO_THROW(simplemc_load(reader, dst_rng, dst_updates, dst_meas, dst_stats));
    EXPECT_EQ(dst_stats.cumulative_steps, stats.cumulative_steps);
}

#ifndef SIMPLEMC_USE_HDF5
TEST(MCCheckpoint, JsonRoundTripWorksWithNlohmannToJsonUserTypes) {
    // User type with only nlohmann to_json / from_json (no simplemc_save). Only meaningful when JSON
    // is the sole mc_serializer backend (see the nlohmann_only_update note above).
    xoshiro256ss rng { 1 };
    update_set updates;
    measurement_set meas;
    simulation_stats stats;
    updates.add({ nlohmann_only_update {}, "nlh", 1.0 });
    run_and_accumulate(updates, meas, stats, rng,
        { .max_steps = 10, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 5 });

    mc_serializer writer { json_serializer {} };
    simplemc_save(writer, rng, updates, meas, stats);

    // Confirm the JSON has a "user" sub-object with the counter populated (i.e., the wrapper
    // fell through to nlohmann's to_json for the user type).
    const auto& root = doc(writer);
    ASSERT_TRUE(root.contains("updates"));
    ASSERT_TRUE(root["updates"].contains("nlh"));
    ASSERT_TRUE(root["updates"]["nlh"].contains("user"));
    EXPECT_GT(root["updates"]["nlh"]["user"]["counter"].get<int>(), 0);

    xoshiro256ss dst_rng { 2 };
    update_set dst_updates;
    measurement_set dst_meas;
    simulation_stats dst_stats;
    dst_updates.add({ nlohmann_only_update {}, "nlh", 1.0 });

    const mc_serializer reader { json_serializer { doc(writer) } };
    EXPECT_NO_THROW(simplemc_load(reader, dst_rng, dst_updates, dst_meas, dst_stats));
    EXPECT_EQ(dst_stats.cumulative_steps, stats.cumulative_steps);
}
#endif // SIMPLEMC_USE_HDF5
