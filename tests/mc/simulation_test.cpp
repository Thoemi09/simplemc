#include <simplemc/mc.hpp>
#include <simplemc/random/xoshiro256.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <limits>
#include <memory>

using namespace simplemc;

namespace {

struct always_accept {
    std::shared_ptr<int> attempts = std::make_shared<int>(0);
    std::shared_ptr<int> accepts = std::make_shared<int>(0);
    std::shared_ptr<int> rejects = std::make_shared<int>(0);
    double attempt() {
        ++*attempts;
        return 1.0;
    }
    void accept() { ++*accepts; }
    void reject() { ++*rejects; }
};

struct half_accept {
    std::shared_ptr<int> accepts = std::make_shared<int>(0);
    std::shared_ptr<int> rejects = std::make_shared<int>(0);
    double attempt() { return 0.5; }
    void accept() { ++*accepts; }
    void reject() { ++*rejects; }
};

struct impossible_attempt {
    std::shared_ptr<int> accepts = std::make_shared<int>(0);
    std::shared_ptr<int> rejects = std::make_shared<int>(0);
    double attempt() { return 0.0; }
    void accept() { ++*accepts; }
    void reject() { ++*rejects; }
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
    void accept() { }
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

// User update that has only nlohmann to_json / from_json (no simplemc_save).
struct nlohmann_only_update {
    int counter = 0;
    double attempt() {
        ++counter;
        return 1.0;
    }
    void accept() { }
};
inline void to_json(nlohmann::json& j, const nlohmann_only_update& u) { j = nlohmann::json { { "counter", u.counter } }; }
inline void from_json(const nlohmann::json& j, nlohmann_only_update& u) { j.at("counter").get_to(u.counter); }

} // namespace

TEST(MCSimulation, AddUpdateRejectsDuplicateName) {
    simulation<> sim;
    sim.add_update(always_accept {}, "a", 1.0);
    EXPECT_THROW(sim.add_update(always_accept {}, "a", 1.0), simplemc_exception);
}

TEST(MCSimulation, AddUpdateRejectsNegativeWeight) {
    simulation<> sim;
    EXPECT_THROW(sim.add_update(always_accept {}, "a", -1.0), simplemc_exception);
}

TEST(MCSimulation, AddUpdatePairSetsInvNameAndRatio) {
    simulation<> sim;
    sim.add_update(always_accept {}, "a", 2.0, always_accept {}, "b", 1.0);

    sim.initialize_update_distribution(); // derives the ratios from the current weights

    const auto us = sim.update_data();
    ASSERT_EQ(us.size(), 2u);

    EXPECT_EQ(us[0].name, "a");
    EXPECT_EQ(us[0].inv_name, "b");
    EXPECT_DOUBLE_EQ(us[0].weight, 2.0);
    EXPECT_DOUBLE_EQ(us[0].ratio, 0.5);

    EXPECT_EQ(us[1].name, "b");
    EXPECT_EQ(us[1].inv_name, "a");
    EXPECT_DOUBLE_EQ(us[1].weight, 1.0);
    EXPECT_DOUBLE_EQ(us[1].ratio, 2.0);
}

TEST(MCSimulation, AddUpdatePairRejectsAsymmetricZeroWeight) {
    simulation<> sim;
    EXPECT_THROW(
        sim.add_update(always_accept {}, "a", 1.0, always_accept {}, "b", 0.0), simplemc_exception);
}

TEST(MCSimulation, AddUpdatePairAllowsBothZero) {
    simulation<> sim;
    // both weights zero is allowed; ratios stay at 1.0
    EXPECT_NO_THROW(sim.add_update(always_accept {}, "a", 0.0, always_accept {}, "b", 0.0));
    const auto us = sim.update_data();
    ASSERT_EQ(us.size(), 2u);
    EXPECT_DOUBLE_EQ(us[0].ratio, 1.0);
    EXPECT_DOUBLE_EQ(us[1].ratio, 1.0);
}

TEST(MCSimulation, AddUpdatePairRejectsSameName) {
    simulation<> sim;
    EXPECT_THROW(
        sim.add_update(always_accept {}, "a", 1.0, always_accept {}, "a", 1.0), simplemc_exception);
}

TEST(MCSimulation, AddMeasurementRejectsDuplicateName) {
    simulation<> sim;
    sim.add_measurement(counting_measurement {}, "m");
    EXPECT_THROW(sim.add_measurement(counting_measurement {}, "m"), simplemc_exception);
}

TEST(MCSimulation, RunThrowsWhenNoUpdates) {
    simulation<> sim;
    EXPECT_THROW(
        sim.run({ .max_steps = 10, .max_time = 1.0, .steps_per_cycle = 1, .cycles_per_check = 1 }),
        simplemc_exception);
}

TEST(MCSimulation, RunThrowsWhenAllWeightsZero) {
    simulation<> sim;
    sim.add_update(always_accept {}, "a", 0.0);
    EXPECT_THROW(
        sim.run({ .max_steps = 10, .max_time = 1.0, .steps_per_cycle = 1, .cycles_per_check = 1 }),
        simplemc_exception);
}

TEST(MCSimulation, StepIncrementsCountersByOne) {
    simulation<> sim { xoshiro256ss { 42 } };
    always_accept aa;
    auto accepts = aa.accepts;
    sim.add_update(aa, "aa", 1.0);

    constexpr int N = 100;
    sim.run({ .max_steps = N, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = N });

    EXPECT_EQ(sim.stats().last_steps_done, static_cast<std::uint64_t>(N));
    EXPECT_EQ(sim.update_data()[0].nprops, static_cast<std::uint64_t>(N));
    EXPECT_EQ(sim.update_data()[0].naccs, static_cast<std::uint64_t>(N));
    EXPECT_EQ(sim.update_data()[0].nimps, 0u);
    EXPECT_EQ(*accepts, N);
}

TEST(MCSimulation, StepRejectsWhenAttemptBelowUniform) {
    simulation<> sim { xoshiro256ss { 7 } };
    half_accept ha;
    auto rejects = ha.rejects;
    sim.add_update(ha, "h", 1.0);

    constexpr int N = 1000;
    sim.run({ .max_steps = N, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = N });

    const auto& us = sim.update_data()[0];
    EXPECT_EQ(us.nprops, static_cast<std::uint64_t>(N));
    EXPECT_EQ(us.nimps, 0u);
    EXPECT_GT(us.naccs, 0u);
    EXPECT_GT(*rejects, 0);
    EXPECT_EQ(us.naccs + static_cast<std::uint64_t>(*rejects), us.nprops);
}

TEST(MCSimulation, ImpossibleIsRecorded) {
    simulation<> sim { xoshiro256ss { 0 } };
    impossible_attempt ia;
    auto accepts = ia.accepts;
    auto rejects = ia.rejects;
    sim.add_update(ia, "imp", 1.0);

    constexpr int N = 50;
    sim.run({ .max_steps = N, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = N });

    const auto& us = sim.update_data()[0];
    EXPECT_EQ(us.nprops, static_cast<std::uint64_t>(N));
    EXPECT_EQ(us.nimps, static_cast<std::uint64_t>(N));
    EXPECT_EQ(us.naccs, 0u);
    EXPECT_EQ(*accepts, 0);
    // Impossible proposals are rejected: attempt() is always followed by accept() or reject().
    EXPECT_EQ(*rejects, N);
}

TEST(MCSimulation, ZeroWeightUpdateNeverPicked) {
    simulation<> sim { xoshiro256ss { 1234 } };
    sim.add_update(always_accept {}, "a", 1.0);
    sim.add_update(always_accept {}, "b", 0.0);
    sim.add_update(always_accept {}, "c", 1.0);

    constexpr int N = 1000;
    sim.run({ .max_steps = N, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = N });

    const auto us = sim.update_data();
    EXPECT_EQ(us[1].nprops, 0u);
    EXPECT_EQ(us[0].nprops + us[2].nprops, static_cast<std::uint64_t>(N));
}

TEST(MCSimulation, CycleCallsMeasureOncePerCycle) {
    simulation<> sim { xoshiro256ss { 0 } };
    sim.add_update(always_accept {}, "aa", 1.0);
    counting_measurement cm;
    auto count = cm.count;
    sim.add_measurement(cm, "counter");

    constexpr std::uint64_t steps_per_cycle = 5;
    constexpr int M = 10;
    sim.run({ .max_steps = M * steps_per_cycle,
        .max_time = 1000.0,
        .steps_per_cycle = steps_per_cycle,
        .cycles_per_check = M });

    EXPECT_EQ(sim.stats().last_steps_done, static_cast<std::uint64_t>(M * steps_per_cycle));
    EXPECT_EQ(*count, M);
    EXPECT_EQ(sim.update_data()[0].nprops, static_cast<std::uint64_t>(M * steps_per_cycle));
    EXPECT_EQ(sim.measurement_data()[0].name, "counter");
}

TEST(MCSimulation, RunStopsOnMaxSteps) {
    simulation<> sim { xoshiro256ss { 0 } };
    sim.add_update(always_accept {}, "aa", 1.0);

    constexpr std::uint64_t max_steps = 100;
    constexpr std::uint64_t steps_per_cycle = 3;
    constexpr std::uint64_t cycles_per_check = 5;
    sim.run({ .max_steps = max_steps,
        .max_time = 1000.0,
        .steps_per_cycle = steps_per_cycle,
        .cycles_per_check = cycles_per_check });

    const std::uint64_t block = steps_per_cycle * cycles_per_check;
    EXPECT_GE(sim.stats().last_steps_done, max_steps);
    EXPECT_LT(sim.stats().last_steps_done, max_steps + block);
}

TEST(MCSimulation, RunStopsOnMaxTime) {
    simulation<> sim { xoshiro256ss { 0 } };
    sim.add_update(always_accept {}, "aa", 1.0);

    constexpr double max_time = 0.001;
    constexpr std::uint64_t max_steps = std::numeric_limits<std::uint64_t>::max();
    sim.run({ .max_steps = max_steps, .max_time = max_time, .steps_per_cycle = 50, .cycles_per_check = 200 });

    EXPECT_GE(sim.stats().last_runtime, max_time);
    EXPECT_LT(sim.stats().last_steps_done, max_steps);
}

TEST(MCSimulation, AccumulateStatsFoldsAllCounters) {
    simulation<> sim { xoshiro256ss { 0 } };
    sim.add_update(always_accept {}, "aa", 1.0);

    const simulation_params p { .max_steps = 30, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 10 };
    sim.run(p);

    const std::uint64_t first_steps = sim.stats().last_steps_done;
    const std::uint64_t first_nprops = sim.update_data()[0].nprops;
    ASSERT_GT(first_steps, 0u);

    sim.accumulate_stats();

    EXPECT_EQ(sim.stats().last_steps_done, 0u);
    EXPECT_DOUBLE_EQ(sim.stats().last_runtime, 0.0);
    EXPECT_EQ(sim.stats().cumulative_steps, first_steps);
    EXPECT_EQ(sim.update_data()[0].nprops, 0u);
    EXPECT_EQ(sim.update_data()[0].cumulative_nprops, first_nprops);
}

TEST(MCSimulation, RunAutoResetsCurrentCountersAfterAccumulate) {
    simulation<> sim { xoshiro256ss { 0 } };
    sim.add_update(always_accept {}, "aa", 1.0);

    const simulation_params p { .max_steps = 30, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 10 };

    sim.run(p);
    const std::uint64_t first_steps = sim.stats().last_steps_done;
    const std::uint64_t first_nprops = sim.update_data()[0].nprops;

    sim.accumulate_stats();
    EXPECT_EQ(sim.stats().cumulative_steps, first_steps);
    EXPECT_EQ(sim.update_data()[0].cumulative_nprops, first_nprops);

    sim.run(p);
    const std::uint64_t second_steps = sim.stats().last_steps_done;
    const std::uint64_t second_nprops = sim.update_data()[0].nprops;

    // run() reset current counters at entry — cumulative survived through the second run
    EXPECT_EQ(sim.stats().cumulative_steps, first_steps);
    EXPECT_EQ(sim.update_data()[0].cumulative_nprops, first_nprops);

    sim.accumulate_stats();
    EXPECT_EQ(sim.stats().cumulative_steps, first_steps + second_steps);
    EXPECT_EQ(sim.update_data()[0].cumulative_nprops, first_nprops + second_nprops);
}

TEST(MCSimulation, FindUpdateByName) {
    simulation<> sim;
    sim.add_update(always_accept {}, "alpha", 1.0);
    sim.add_update(always_accept {}, "beta", 1.0);
    EXPECT_EQ(sim.find_update("alpha"), 0u);
    EXPECT_EQ(sim.find_update("beta"), 1u);
    EXPECT_FALSE(sim.find_update("missing").has_value());
}

TEST(MCSimulation, FindMeasurementByName) {
    simulation<> sim;
    sim.add_measurement(counting_measurement {}, "obs1");
    sim.add_measurement(counting_measurement {}, "obs2");
    EXPECT_EQ(sim.find_measurement("obs1"), 0u);
    EXPECT_EQ(sim.find_measurement("obs2"), 1u);
    EXPECT_FALSE(sim.find_measurement("missing").has_value());
}

TEST(MCSimulation, SetUpdateWeightChangesSelection) {
    simulation<> sim { xoshiro256ss { 0 } };
    sim.add_update(always_accept {}, "a", 1.0);
    sim.add_update(always_accept {}, "b", 0.0);

    constexpr int N = 50;
    const simulation_params p { .max_steps = N, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = N };
    sim.run(p);
    EXPECT_EQ(sim.update_data()[0].nprops, static_cast<std::uint64_t>(N));
    EXPECT_EQ(sim.update_data()[1].nprops, 0u);

    sim.accumulate_stats();
    sim.set_update_weight("a", 0.0);
    sim.set_update_weight("b", 1.0);

    sim.run(p);
    EXPECT_EQ(sim.update_data()[0].nprops, 0u);                              // none in second batch
    EXPECT_EQ(sim.update_data()[1].nprops, static_cast<std::uint64_t>(N));   // all-second batch
    EXPECT_EQ(sim.update_data()[0].cumulative_nprops, static_cast<std::uint64_t>(N)); // first batch survives
    EXPECT_EQ(sim.update_data()[1].cumulative_nprops, 0u);
    EXPECT_DOUBLE_EQ(sim.update_data()[0].weight, 0.0);
    EXPECT_DOUBLE_EQ(sim.update_data()[1].weight, 1.0);
}

TEST(MCSimulation, SetUpdateWeightRejectsNegative) {
    simulation<> sim;
    sim.add_update(always_accept {}, "a", 1.0);
    EXPECT_THROW(sim.set_update_weight("a", -1.0), simplemc_exception);
    // weight unchanged
    EXPECT_DOUBLE_EQ(sim.update_data()[0].weight, 1.0);
}

TEST(MCSimulation, SetUpdateWeightRejectsMissingName) {
    simulation<> sim;
    sim.add_update(always_accept {}, "a", 1.0);
    EXPECT_THROW(sim.set_update_weight("missing", 2.0), simplemc_exception);
}

TEST(MCSimulation, AddMeasurementInactiveSkipsMeasure) {
    simulation<> sim { xoshiro256ss { 0 } };
    sim.add_update(always_accept {}, "aa", 1.0);
    counting_measurement cm;
    auto count = cm.count;
    sim.add_measurement(cm, "obs", /*is_active=*/false);

    sim.run({ .max_steps = 10, .max_time = 1000.0, .steps_per_cycle = 5, .cycles_per_check = 2 });

    EXPECT_EQ(*count, 0);
    EXPECT_FALSE(sim.measurement_data()[0].is_active);
    EXPECT_EQ(sim.stats().last_steps_done, 10u); // steps still happen
}

TEST(MCSimulation, SetMeasurementActiveTogglesAtRuntime) {
    simulation<> sim { xoshiro256ss { 0 } };
    sim.add_update(always_accept {}, "aa", 1.0);
    counting_measurement cm;
    auto count = cm.count;
    sim.add_measurement(cm, "obs");

    const simulation_params one_cycle { .max_steps = 1, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 1 };
    sim.run(one_cycle);
    EXPECT_EQ(*count, 1);

    sim.set_measurement_active("obs", false);
    sim.run({ .max_steps = 2, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 2 });
    EXPECT_EQ(*count, 1); // skipped both times

    sim.set_measurement_active("obs", true);
    sim.run(one_cycle);
    EXPECT_EQ(*count, 2);
}

TEST(MCSimulation, SetMeasurementActiveRejectsMissingName) {
    simulation<> sim;
    sim.add_measurement(counting_measurement {}, "obs");
    EXPECT_THROW(sim.set_measurement_active("missing", false), simplemc_exception);
}

TEST(MCSimulation, ResetStatsZeroesCurrentNotCumulative) {
    simulation<> sim { xoshiro256ss { 0 } };
    sim.add_update(always_accept {}, "aa", 1.0);

    sim.run({ .max_steps = 20, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 10 });
    sim.accumulate_stats();
    sim.run({ .max_steps = 20, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 10 });

    const std::uint64_t cumulative_before = sim.update_data()[0].cumulative_nprops;
    const std::uint64_t current_before = sim.update_data()[0].nprops;
    ASSERT_GT(current_before, 0u);
    ASSERT_GT(cumulative_before, 0u);

    sim.reset_stats();

    EXPECT_EQ(sim.update_data()[0].nprops, 0u);
    EXPECT_EQ(sim.update_data()[0].cumulative_nprops, cumulative_before); // survived
    EXPECT_EQ(sim.stats().last_steps_done, 0u);
}

TEST(MCSimulation, RunSkipsAllMeasurementsWhenSkipMeasurementsTrue) {
    simulation<> sim { xoshiro256ss { 0 } };
    sim.add_update(always_accept {}, "aa", 1.0);
    counting_measurement cm;
    auto count = cm.count;
    sim.add_measurement(cm, "obs"); // active by default

    sim.run({ .max_steps = 10,
        .max_time = 1000.0,
        .steps_per_cycle = 1,
        .cycles_per_check = 10,
        .skip_measurements = true });

    EXPECT_EQ(*count, 0);                        // no measurements happened
    EXPECT_EQ(sim.stats().last_steps_done, 10u);      // steps still happened
    EXPECT_TRUE(sim.measurement_data()[0].is_active); // per-measurement flag untouched
}

TEST(MCSimulation, JsonRoundTripPersistsCumulativeAndConfig) {
    // Source simulation: stateful + stateless toys + a stateful measurement
    simulation<> src { xoshiro256ss { 42 } };
    stateful_update stateful;
    auto stateful_counter = stateful.counter;
    src.add_update(stateful, "stateful", 2.5);
    src.add_update(always_accept {}, "stateless", 1.0);
    stateful_measurement m;
    auto m_tally = m.tally;
    src.add_measurement(m, "obs");
    src.add_measurement(counting_measurement {}, "stateless_meas");

    src.run({ .max_steps = 50, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 10 });
    const std::uint64_t src_steps = src.stats().last_steps_done;
    const int src_counter = *stateful_counter;
    const int src_tally = *m_tally;
    ASSERT_GT(src_steps, 0u);
    ASSERT_GT(src_counter, 0);
    ASSERT_GT(src_tally, 0);
    src.accumulate_stats();

    // Save
    json_serializer writer;
    writer.save_at("sim", src);

    // Destination: same structure, different RNG seed
    simulation<> dst { xoshiro256ss { 999 } };
    stateful_update dst_stateful;
    auto dst_stateful_counter = dst_stateful.counter;
    dst.add_update(dst_stateful, "stateful", 2.5);
    dst.add_update(always_accept {}, "stateless", 1.0);
    stateful_measurement dst_m;
    auto dst_m_tally = dst_m.tally;
    dst.add_measurement(dst_m, "obs");
    dst.add_measurement(counting_measurement {}, "stateless_meas");

    // Load
    json_serializer reader { writer.root() };
    reader.load_at("sim", dst);

    // Persistent fields propagated
    EXPECT_EQ(dst.stats().cumulative_steps, src.stats().cumulative_steps);
    EXPECT_DOUBLE_EQ(dst.stats().cumulative_time, src.stats().cumulative_time);
    EXPECT_DOUBLE_EQ(dst.update_data()[0].weight, 2.5);
    EXPECT_DOUBLE_EQ(dst.update_data()[1].weight, 1.0);
    EXPECT_EQ(
        dst.update_data()[0].cumulative_nprops, src.update_data()[0].cumulative_nprops);
    EXPECT_EQ(dst.update_data()[0].inv_name, src.update_data()[0].inv_name);
    EXPECT_DOUBLE_EQ(dst.update_data()[0].ratio, src.update_data()[0].ratio);
    EXPECT_EQ(dst.measurement_data()[0].is_active, src.measurement_data()[0].is_active);

    // Per-run fields are zero on dst
    EXPECT_EQ(dst.stats().last_steps_done, 0u);
    EXPECT_DOUBLE_EQ(dst.stats().last_runtime, 0.0);
    EXPECT_EQ(dst.update_data()[0].nprops, 0u);

    // User-state round-trip via simplemc_save / simplemc_load
    EXPECT_EQ(*dst_stateful_counter, src_counter);
    EXPECT_EQ(*dst_m_tally, src_tally);

    // RNG continuation: a draw from dst matches a draw from src
    EXPECT_EQ(dst.rng()(), src.rng()());
}

TEST(MCSimulation, JsonLoadThrowsOnMissingUpdate) {
    simulation<> src;
    src.add_update(always_accept {}, "a", 1.0);
    src.run({ .max_steps = 5, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 5 });

    json_serializer writer;
    writer.save_at("sim", src);

    simulation<> dst;
    dst.add_update(always_accept {}, "b", 1.0); // different name

    json_serializer reader { writer.root() };
    EXPECT_THROW(reader.load_at("sim", dst), simplemc_exception);
}

TEST(MCSimulation, JsonLoadThrowsOnMissingMeasurement) {
    simulation<> src;
    src.add_update(always_accept {}, "a", 1.0);
    src.add_measurement(counting_measurement {}, "obs1");
    src.run({ .max_steps = 5, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 5 });

    json_serializer writer;
    writer.save_at("sim", src);

    simulation<> dst;
    dst.add_update(always_accept {}, "a", 1.0);
    dst.add_measurement(counting_measurement {}, "different_name");

    json_serializer reader { writer.root() };
    EXPECT_THROW(reader.load_at("sim", dst), simplemc_exception);
}

TEST(MCSimulation, JsonRoundTripWorksWithStatelessUserTypes) {
    // Stateless user types have neither simplemc_save nor nlohmann::to_json — the wrapper's
    // save_at silently no-ops. Round-trip still succeeds.
    simulation<> src;
    src.add_update(always_accept {}, "aa", 1.0);
    src.add_measurement(counting_measurement {}, "obs");
    src.run({ .max_steps = 5, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 5 });
    src.accumulate_stats();

    json_serializer writer;
    EXPECT_NO_THROW(writer.save_at("sim", src));

    simulation<> dst;
    dst.add_update(always_accept {}, "aa", 1.0);
    dst.add_measurement(counting_measurement {}, "obs");

    json_serializer reader { writer.root() };
    EXPECT_NO_THROW(reader.load_at("sim", dst));
    EXPECT_EQ(dst.stats().cumulative_steps, src.stats().cumulative_steps);
}

TEST(MCSimulation, JsonRoundTripWorksWithNlohmannToJsonUserTypes) {
    // User type with only nlohmann to_json / from_json (no simplemc_save).
    simulation<> src;
    src.add_update(nlohmann_only_update {}, "nlh", 1.0);
    src.run({ .max_steps = 10, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 5 });
    src.accumulate_stats();

    json_serializer writer;
    writer.save_at("sim", src);

    // Confirm the JSON has a "user" sub-object with the counter populated (i.e., the wrapper
    // fell through to nlohmann's to_json for the user type).
    const auto& root = writer.root();
    ASSERT_TRUE(root.contains("sim"));
    ASSERT_TRUE(root["sim"].contains("updates"));
    ASSERT_TRUE(root["sim"]["updates"].contains("nlh"));
    ASSERT_TRUE(root["sim"]["updates"]["nlh"].contains("user"));
    EXPECT_GT(root["sim"]["updates"]["nlh"]["user"]["counter"].get<int>(), 0);

    simulation<> dst;
    dst.add_update(nlohmann_only_update {}, "nlh", 1.0);

    json_serializer reader { writer.root() };
    EXPECT_NO_THROW(reader.load_at("sim", dst));
    EXPECT_EQ(dst.stats().cumulative_steps, src.stats().cumulative_steps);
}

// =====================================================================================
// Callback forwarding through the simulation aggregate.
// =====================================================================================

TEST(MCSimulationCallbacks, OnStepFiresThroughAggregate) {
    simulation<> sim;
    sim.add_update(always_accept {}, "u", 1.0);

    std::uint64_t step_calls = 0;
    auto cbs = simplemc::run_callbacks { .on_step = [&](const simulation_ctx&) { ++step_calls; } };

    sim.run({ .max_steps = 100, .max_time = 1e6, .steps_per_cycle = 5, .cycles_per_check = 4 }, cbs);
    EXPECT_EQ(step_calls, sim.stats().last_steps_done);
}

TEST(MCSimulationCallbacks, StopWhenTrumpsMaxSteps) {
    simulation<> sim;
    sim.add_update(always_accept {}, "u", 1.0);

    auto cbs = simplemc::run_callbacks {
        .stop_when = [](const simulation_ctx& x) { return x.steps_done >= 50; },
    };
    sim.run({ .max_steps = std::numeric_limits<std::uint64_t>::max(), .max_time = 1e6,
                .steps_per_cycle = 5, .cycles_per_check = 4 },
        cbs);
    EXPECT_GE(sim.stats().last_steps_done, 50u);
    EXPECT_LT(sim.stats().last_steps_done, 200u);
}

// =====================================================================================
// Move semantics.
// =====================================================================================

TEST(MCSimulation, IsMovableAndRunsAfterMove) {
    static_assert(std::movable<simulation<>>);

    simulation<> src { xoshiro256ss { 7 } };
    src.add_update(always_accept {}, "u", 1.0);
    counting_measurement cm;
    auto count = cm.count;
    src.add_measurement(cm, "obs");

    constexpr int N = 50;
    const simulation_params p { .max_steps = N, .max_time = 1000.0, .steps_per_cycle = 5, .cycles_per_check = 2 };
    src.run(p);
    src.accumulate_stats();
    const int count_after_first = *count;
    EXPECT_GT(count_after_first, 0);

    simulation<> dst { std::move(src) };
    dst.run(p);
    EXPECT_EQ(dst.stats().cumulative_steps, static_cast<std::uint64_t>(N));
    EXPECT_EQ(dst.stats().last_steps_done, static_cast<std::uint64_t>(N));
    EXPECT_GT(*count, count_after_first); // measurements keep firing on the moved-to simulation
}
