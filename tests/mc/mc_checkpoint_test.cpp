#include "./mc_test_utils.hpp"

#include <simplemc/mc.hpp>
#include <simplemc/random/xoshiro256.hpp>
#include <simplemc/serialize/json/file_io.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/file_io.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#ifdef SIMPLEMC_USE_HDF5
#include <simplemc/serialize/hdf5/hdf5_serializer.hpp>
#endif // SIMPLEMC_USE_HDF5

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <utility>

using namespace simplemc;

namespace {

// Stateful update with ADL save/load.
struct stateful_update {
    int counter = 0;
    double attempt() {
        ++counter;
        return 1.0;
    }
    void accept() {}
};

template <serializer S>
void simplemc_save(S& s, const stateful_update& u) {
    s.save_at("counter", u.counter);
}

template <serializer S>
void simplemc_load(const S& s, stateful_update& u) {
    s.load_at("counter", u.counter);
}

// Stateful measurement with ADL save/load.
struct stateful_measurement {
    int tally = 0;
    void measure() { ++tally; }
};

template <serializer S>
void simplemc_save(S& s, const stateful_measurement& m) {
    s.save_at("tally", m.tally);
}

template <serializer S>
void simplemc_load(const S& s, stateful_measurement& m) {
    s.load_at("tally", m.tally);
}

// User update with only nlohmann to_json / from_json (no simplemc_save).
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

inline void from_json(const nlohmann::json& j, nlohmann_only_update& u) {
    j.at("counter").get_to(u.counter);
}

// Drive a short run over the assembled components, then fold counters into the cumulative state.
template <typename U, typename M>
void run_and_accumulate(U& us, M& ms, simulation_stats& stats, xoshiro256ss& rng, const simulation_params& p) {
    metropolis_kernel kernel { us };
    const auto ctx = run(rng, kernel, ms, p);
    accumulate_simulation_stats(stats, ctx);
    us.accumulate_counters();
}

// Serializable user-state.
struct run_config {
    int seed = 0;
    double beta = 0.0;

    template <serializer S>
    friend void simplemc_save(S& s, const run_config& c) {
        s.save_at("seed", c.seed);
        s.save_at("beta", c.beta);
    }
    template <serializer S>
    friend void simplemc_load(const S& s, run_config& c) {
        s.load_at("seed", c.seed);
        s.load_at("beta", c.beta);
    }
};

// Serializable user-state that throws on save.
struct throwing_config {
    template <serializer S>
    friend void simplemc_save(S&, const throwing_config&) {
        throw simplemc_exception("intentional save failure");
    }
    template <serializer S>
    friend void simplemc_load(const S&, throwing_config&) {}
};

// Remove and count leftover checkpoint temp files. Used to verify that a failed save cleans up after
// itself.
int remove_tmp_siblings(const std::filesystem::path& path) {
    const auto prefix = path.filename().string() + ".tmp";
    int n = 0;
    for (const auto& e : std::filesystem::directory_iterator { path.parent_path() }) {
        if (e.path().filename().string().starts_with(prefix)) {
            std::filesystem::remove(e.path());
            ++n;
        }
    }
    return n;
}

// Write/Read JSON checkpoint to file — the composed file idiom the library expects users to copy:
// atomic_file_write + serialization + write/read_json_file.
template <typename RNG, mc_update... Us, mc_measurement... Ms, typename F = decltype([](json_serializer&) {})>
void write_checkpoint(const std::filesystem::path& path, const RNG& rng, const update_set<Us...>& us,
    const measurement_set<Ms...>& ms, const simulation_stats& stats, json_file_mode mode = json_file_mode::text,
    F&& f = {}) {
    atomic_file_write(path, [&](const std::filesystem::path& tmp) {
        json_serializer s;
        simplemc_save(s, rng, us, ms, stats);
        std::forward<F>(f)(s);
        write_json_file(s.root(), tmp, { .mode = mode });
    });
}

template <typename RNG, mc_update... Us, mc_measurement... Ms, typename F = decltype([](const json_serializer&) {})>
void read_checkpoint(const std::filesystem::path& path, RNG& rng, update_set<Us...>& us, measurement_set<Ms...>& ms,
    simulation_stats& stats, json_file_mode mode = json_file_mode::text, F&& f = {}) {
    nlohmann::json doc;
    read_json_file(doc, path, { .mode = mode });
    const json_serializer s { std::move(doc) };
    simplemc_load(s, rng, us, ms, stats);
    std::forward<F>(f)(s);
}

} // namespace

// Test the composite save/load input-config routines.
TEST(MCCheckpoint, JsonRoundTripPersistsCumulativeAndUserState) {
    // prepare sources
    xoshiro256ss rng { 42 };
    simulation_stats stats;
    update_set updates { update { stateful_update {}, "stateful", 2.5 }, update { dummy_update {}, "stateless", 1.0 } };
    measurement_set meas { measurement { stateful_measurement {}, "obs" },
        measurement { dummy_measurement {}, "stateless_meas" } };

    // run a short simulation
    run_and_accumulate(updates, meas, stats, rng,
        { .max_steps = 50, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 10 });
    const int src_counter = updates.get<0>().value().counter;
    const int src_tally = meas.get<0>().value().tally;
    ASSERT_GT(stats.cumulative_steps, 0u);
    ASSERT_GT(src_counter, 0);
    ASSERT_GT(src_tally, 0);

    // save to serializer
    json_serializer writer;
    simplemc_save(writer, rng, updates, meas, stats);

    // prepare destinations
    xoshiro256ss dst_rng { 999 };
    simulation_stats dst_stats;
    update_set dst_updates { update { stateful_update {}, "stateful", 2.5 },
        update { dummy_update {}, "stateless", 1.0 } };
    measurement_set dst_meas { measurement { stateful_measurement {}, "obs" },
        measurement { dummy_measurement {}, "stateless_meas" } };

    // load from serializer
    const json_serializer reader { writer.root() };
    simplemc_load(reader, dst_rng, dst_updates, dst_meas, dst_stats);

    // check that the round-trip preserved cumulative counters, update weights and measurement
    // activation state
    EXPECT_EQ(dst_stats.cumulative_steps, stats.cumulative_steps);
    EXPECT_DOUBLE_EQ(dst_stats.cumulative_time, stats.cumulative_time);
    EXPECT_DOUBLE_EQ(dst_updates.get<0>().stats().weight, 2.5);
    EXPECT_DOUBLE_EQ(dst_updates.get<1>().stats().weight, 1.0);
    EXPECT_EQ(dst_updates.get<0>().stats().cumulative_nprops, updates.get<0>().stats().cumulative_nprops);
    EXPECT_EQ(dst_updates.get<0>().stats().inv_name, updates.get<0>().stats().inv_name);
    EXPECT_DOUBLE_EQ(dst_updates.get<0>().stats().ratio, updates.get<0>().stats().ratio);
    EXPECT_EQ(dst_meas.get<0>().is_active(), meas.get<0>().is_active());

    // per-run update counters should be untouched
    EXPECT_EQ(dst_updates.get<0>().stats().nprops, 0u);

    // check user-state round-trip
    EXPECT_EQ(dst_updates.get<0>().value().counter, src_counter);
    EXPECT_EQ(dst_meas.get<0>().value().tally, src_tally);

    // check RNG round-trip
    EXPECT_EQ(dst_rng(), rng());
}

// Test that the composite simplemc_load throws if an update/measurement is missing.
TEST(MCCheckpoint, JsonLoadThrowsOnMissingUpdate) {
    // prepare sources
    xoshiro256ss rng { 1 };
    simulation_stats stats;
    update_set updates { update { dummy_update {}, "a", 1.0 } };
    measurement_set<> meas;

    // run a short simulation
    run_and_accumulate(
        updates, meas, stats, rng, { .max_steps = 5, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 5 });

    // save to serializer
    json_serializer writer;
    simplemc_save(writer, rng, updates, meas, stats);

    // prepare destinations with a different update name
    xoshiro256ss dst_rng { 2 };
    simulation_stats dst_stats;
    update_set dst_updates { update { dummy_update {}, "b", 1.0 } }; // different name
    measurement_set<> dst_meas;

    // load from serializer
    const json_serializer reader { writer.root() };
    EXPECT_THROW(simplemc_load(reader, dst_rng, dst_updates, dst_meas, dst_stats), simplemc_exception);
}

TEST(MCCheckpoint, JsonLoadThrowsOnMissingMeasurement) {
    // prepare sources
    xoshiro256ss rng { 1 };
    simulation_stats stats;
    update_set updates { update { dummy_update {}, "a", 1.0 } };
    measurement_set meas { measurement { dummy_measurement {}, "obs1" } };

    // run a short simulation
    run_and_accumulate(
        updates, meas, stats, rng, { .max_steps = 5, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 5 });

    // save to serializer
    json_serializer writer;
    simplemc_save(writer, rng, updates, meas, stats);

    // prepare destinations with a different measurement name
    xoshiro256ss dst_rng { 2 };
    simulation_stats dst_stats;
    update_set dst_updates { update { dummy_update {}, "a", 1.0 } };
    measurement_set dst_meas { measurement { dummy_measurement {}, "different_name" } };

    // load from serializer
    const json_serializer reader { writer.root() };
    EXPECT_THROW(simplemc_load(reader, dst_rng, dst_updates, dst_meas, dst_stats), simplemc_exception);
}

// Test the composite save/load with stateless user types.
TEST(MCCheckpoint, JsonRoundTripWorksWithStatelessUserTypes) {
    // prepare sources
    xoshiro256ss rng { 1 };
    simulation_stats stats;
    update_set updates { update { dummy_update {}, "aa", 1.0 } };
    measurement_set meas { measurement { dummy_measurement {}, "obs" } };

    // run a short simulation
    run_and_accumulate(
        updates, meas, stats, rng, { .max_steps = 5, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 5 });

    // save to serializer
    json_serializer writer;
    EXPECT_NO_THROW(simplemc_save(writer, rng, updates, meas, stats));

    // prepare destinations
    xoshiro256ss dst_rng { 2 };
    simulation_stats dst_stats;
    update_set dst_updates { update { dummy_update {}, "aa", 1.0 } };
    measurement_set dst_meas { measurement { dummy_measurement {}, "obs" } };

    // load from serializer
    const json_serializer reader { writer.root() };
    EXPECT_NO_THROW(simplemc_load(reader, dst_rng, dst_updates, dst_meas, dst_stats));
    EXPECT_EQ(dst_stats.cumulative_steps, stats.cumulative_steps);
}

// Test that the composite save/load works with user types that only implement nlohmann
// to_json/from_json.
TEST(MCCheckpoint, JsonRoundTripWorksWithNlohmannToJsonUserTypes) {
    // prepare sources
    xoshiro256ss rng { 1 };
    simulation_stats stats;
    update_set updates { update { nlohmann_only_update {}, "nlh", 1.0 } };
    measurement_set<> meas;

    // run a short simulation
    run_and_accumulate(updates, meas, stats, rng,
        { .max_steps = 10, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 5 });

    // save to serializer
    json_serializer writer;
    simplemc_save(writer, rng, updates, meas, stats);

    // check that the nlohmann-only user type was serialized to JSON
    const auto& root = writer.root();
    ASSERT_TRUE(root.contains("updates"));
    ASSERT_TRUE(root["updates"].contains("nlh"));
    ASSERT_TRUE(root["updates"]["nlh"].contains("user"));
    EXPECT_GT(root["updates"]["nlh"]["user"]["counter"].get<int>(), 0);

    // prepare destinations
    xoshiro256ss dst_rng { 2 };
    simulation_stats dst_stats;
    update_set dst_updates { update { nlohmann_only_update {}, "nlh", 1.0 } };
    measurement_set<> dst_meas;

    // load from serializer
    const json_serializer reader { writer.root() };
    EXPECT_NO_THROW(simplemc_load(reader, dst_rng, dst_updates, dst_meas, dst_stats));
    EXPECT_EQ(dst_stats.cumulative_steps, stats.cumulative_steps);
}

// Test writing/reading checkpoints to/from a file.
TEST(MCCheckpoint, FileRoundTripViaComposedIdiom) {
    // prepare sources
    xoshiro256ss rng { 7 };
    simulation_stats stats;
    update_set updates { update { stateful_update {}, "stateful", 1.0 } };
    measurement_set<> meas;

    // run a short simulation
    run_and_accumulate(updates, meas, stats, rng,
        { .max_steps = 20, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 5 });
    const int src_counter = updates.get<0>().value().counter;

    // write checkpoint to a temporary file
    const auto path = std::filesystem::temp_directory_path() / "mc_checkpoint_file_roundtrip.json";
    write_checkpoint(path, rng, updates, meas, stats);

    // prepare destinations
    xoshiro256ss dst_rng { 123 };
    simulation_stats dst_stats;
    update_set dst_updates { update { stateful_update {}, "stateful", 1.0 } };
    measurement_set<> dst_meas;

    // read checkpoint from the file
    read_checkpoint(path, dst_rng, dst_updates, dst_meas, dst_stats);

    // check that the round-trip preserved expected fields
    EXPECT_EQ(dst_stats.cumulative_steps, stats.cumulative_steps);
    EXPECT_EQ(dst_updates.get<0>().value().counter, src_counter);
    EXPECT_EQ(dst_rng(), rng());

    std::filesystem::remove(path);
}

// Test that loading a corrupted checkpoint file throws an exception.
TEST(MCCheckpoint, LoadThrowsOnCorruptFile) {
    // prepare sources
    xoshiro256ss rng { 1 };
    simulation_stats stats;
    update_set updates { update { dummy_update {}, "a", 1.0 } };
    measurement_set<> meas;

    // write a truncated JSON file that is not valid JSON
    const auto truncated = std::filesystem::temp_directory_path() / "mc_checkpoint_truncated.json";
    {
        std::ofstream ofs { truncated };
        ofs << R"({ "rng": [1, 2)";
    }

    // check that reading the file throws an exception
    EXPECT_ANY_THROW(read_checkpoint(truncated, rng, updates, meas, stats));
    std::filesystem::remove(truncated);

    // write a valid JSON file that is not a valid checkpoint (missing keys)
    const auto empty_doc = std::filesystem::temp_directory_path() / "mc_checkpoint_empty_doc.json";
    {
        std::ofstream ofs { empty_doc };
        ofs << "{}";
    }

    // check that reading the file does not throw
    json_serializer s;
    EXPECT_NO_THROW(read_json_file(s.root(), empty_doc));

    // check that loading the empty document throws an exception
    EXPECT_ANY_THROW(simplemc_load(s, rng, updates, meas, stats));
    std::filesystem::remove(empty_doc);
}

// Test writing/reading checkpoints to/from a file for all JSON modes.
TEST(MCCheckpoint, FileRoundTripsAllJsonModes) {
    const auto path = std::filesystem::temp_directory_path() / "mc_checkpoint_modes.ckpt";

    // prepare sources
    xoshiro256ss rng;
    update_set updates { update { stateful_update {}, "u", 1.0 } };
    measurement_set<> meas;
    simulation_stats stats;
    stats.cumulative_steps = 77;

    // loop over all JSON modes
    for (const auto mode : { json_file_mode::text, json_file_mode::bson, json_file_mode::cbor, json_file_mode::msgpack,
             json_file_mode::ubjson }) {
        // write checkpoint file
        std::filesystem::remove(path);
        write_checkpoint(path, rng, updates, meas, stats, mode);
        ASSERT_TRUE(std::filesystem::exists(path));

        // prepare destinations
        xoshiro256ss dst_rng;
        update_set dst_updates { update { stateful_update {}, "u", 1.0 } };
        measurement_set<> dst_meas;
        simulation_stats dst_stats;

        // read checkpoint file
        read_checkpoint(path, dst_rng, dst_updates, dst_meas, dst_stats, mode);
        EXPECT_EQ(dst_stats.cumulative_steps, 77u);

        std::filesystem::remove(path);
    }
}

// Test writing/reading a binary JSON checkpoint file with an extra key for user config.
TEST(MCCheckpoint, FileRoundTripsExtraConfigKey) {
    const auto path = std::filesystem::temp_directory_path() / "mc_checkpoint_config.cbor";
    std::filesystem::remove(path);

    // prepare sources
    xoshiro256ss rng;
    update_set updates { update { stateful_update {}, "u", 1.0 } };
    measurement_set<> meas;
    simulation_stats stats;
    stats.cumulative_steps = 7;
    const run_config cfg { .seed = 5, .beta = 2.5 };

    // write checkpoint file with extra config key
    write_checkpoint(
        path, rng, updates, meas, stats, json_file_mode::cbor, [&](json_serializer& s) { s.save_at("config", cfg); });
    ASSERT_TRUE(std::filesystem::exists(path));

    // prepare destinations
    xoshiro256ss dst_rng;
    update_set dst_updates { update { stateful_update {}, "u", 1.0 } };
    measurement_set<> dst_meas;
    simulation_stats dst_stats;
    run_config dst_cfg;

    // read checkpoint file with extra config key
    read_checkpoint(path, dst_rng, dst_updates, dst_meas, dst_stats, json_file_mode::cbor,
        [&](const json_serializer& s) { s.load_at("config", dst_cfg); });

    // check that the round-trip preserved the checkpoint fields and user config
    EXPECT_EQ(dst_stats.cumulative_steps, 7u);
    EXPECT_EQ(dst_cfg.seed, 5);
    EXPECT_DOUBLE_EQ(dst_cfg.beta, 2.5);

    std::filesystem::remove(path);
}

// Test that using atomic_file_write for checkpointing is in fact atomic.
TEST(MCCheckpoint, FileWriteIsAtomic) {
    const auto path = std::filesystem::temp_directory_path() / "mc_checkpoint_atomic.json";
    std::filesystem::remove(path);
    remove_tmp_siblings(path);

    // prepare sources
    xoshiro256ss rng;
    update_set updates { update { stateful_update {}, "u", 1.0 } };
    measurement_set<> meas;
    simulation_stats stats;
    stats.cumulative_steps = 111;

    // write checkpoint file
    write_checkpoint(path, rng, updates, meas, stats);
    ASSERT_TRUE(std::filesystem::exists(path));

    // a failing write should not corrupt the existing checkpoint and should clean up after itself
    stats.cumulative_steps = 222;
    const auto failing_write = [&] {
        write_checkpoint(path, rng, updates, meas, stats, json_file_mode::text,
            [](json_serializer& s) { s.save_at("config", throwing_config {}); });
    };
    EXPECT_THROW(failing_write(), simplemc_exception);
    EXPECT_EQ(remove_tmp_siblings(path), 0);

    // prepare destinations
    xoshiro256ss dst_rng;
    update_set dst_updates { update { stateful_update {}, "u", 1.0 } };
    measurement_set<> dst_meas;
    simulation_stats dst_stats;

    // read checkpoint file
    read_checkpoint(path, dst_rng, dst_updates, dst_meas, dst_stats);
    EXPECT_EQ(dst_stats.cumulative_steps, 111u);

    std::filesystem::remove(path);
}

#ifdef SIMPLEMC_USE_HDF5

// Test writing/reading checkpoints to/from an HDF5 file.
TEST(MCCheckpoint, Hdf5FileRoundTripsComponentsAndConfig) {
    const auto path = std::filesystem::temp_directory_path() / "mc_checkpoint.h5";
    std::filesystem::remove(path);

    // prepare sources
    xoshiro256ss rng;
    update_set updates { update { stateful_update {}, "u", 1.0 } };
    measurement_set meas { measurement { stateful_measurement {}, "m" } };
    simulation_stats stats;
    stats.cumulative_steps = 9999;
    stats.cumulative_time = 12.25;
    const run_config cfg { .seed = 3, .beta = 1.5 };

    // write checkpoint file
    atomic_file_write(path, [&](const std::filesystem::path& tmp) {
        hdf5_serializer s { tmp, hdf5_file_mode::truncate };
        simplemc_save(s, rng, updates, meas, stats);
        s.save_at("config", cfg);
    });
    ASSERT_TRUE(std::filesystem::exists(path));

    // prepare destinations
    xoshiro256ss dst_rng;
    update_set dst_updates { update { stateful_update {}, "u", 1.0 } };
    measurement_set dst_meas { measurement { stateful_measurement {}, "m" } };
    simulation_stats dst_stats;
    run_config dst_cfg;

    // read checkpoint file
    const hdf5_serializer s { path, hdf5_file_mode::read };
    simplemc_load(s, dst_rng, dst_updates, dst_meas, dst_stats);
    s.load_at("config", dst_cfg);

    // check that the round-trip preserved the checkpoint fields and user config
    EXPECT_EQ(dst_stats.cumulative_steps, 9999u);
    EXPECT_DOUBLE_EQ(dst_stats.cumulative_time, 12.25);
    EXPECT_EQ(dst_cfg.seed, 3);
    EXPECT_DOUBLE_EQ(dst_cfg.beta, 1.5);

    std::filesystem::remove(path);
}

// Test that writing/reading checkpoints to/from an HDF5 file is atomic.
TEST(MCCheckpoint, Hdf5FileWriteIsAtomic) {
    const auto path = std::filesystem::temp_directory_path() / "mc_checkpoint_atomic.h5";
    std::filesystem::remove(path);
    remove_tmp_siblings(path);

    // prepare sources
    xoshiro256ss rng;
    update_set updates { update { stateful_update {}, "u", 1.0 } };
    measurement_set<> meas;
    simulation_stats stats;
    stats.cumulative_steps = 111;

    // write checkpoint file
    atomic_file_write(path, [&](const std::filesystem::path& tmp) {
        hdf5_serializer s { tmp, hdf5_file_mode::truncate };
        simplemc_save(s, rng, updates, meas, stats);
    });
    ASSERT_TRUE(std::filesystem::exists(path));

    // a failing write should not corrupt the existing checkpoint and should clean up after itself
    stats.cumulative_steps = 222;
    const auto failing_write = [&] {
        atomic_file_write(path, [&](const std::filesystem::path& tmp) {
            hdf5_serializer s { tmp, hdf5_file_mode::truncate };
            simplemc_save(s, rng, updates, meas, stats);
            s.save_at("config", throwing_config {});
        });
    };
    EXPECT_THROW(failing_write(), simplemc_exception);
    EXPECT_EQ(remove_tmp_siblings(path), 0);

    // prepare destinations
    xoshiro256ss dst_rng;
    update_set dst_updates { update { stateful_update {}, "u", 1.0 } };
    measurement_set<> dst_meas;
    simulation_stats dst_stats;

    // read checkpoint file
    const hdf5_serializer s { path, hdf5_file_mode::read };
    simplemc_load(s, dst_rng, dst_updates, dst_meas, dst_stats);
    EXPECT_EQ(dst_stats.cumulative_steps, 111u);

    std::filesystem::remove(path);
}

// Test that appending a checkpoint to an existing HDF5 file preserves unrelated datasets.
TEST(MCCheckpoint, Hdf5AppendModePreservesSiblingContent) {
    const auto path = std::filesystem::temp_directory_path() / "mc_checkpoint_append.h5";
    std::filesystem::remove(path);

    // prepare sources
    xoshiro256ss rng;
    update_set updates { update { stateful_update {}, "u", 1.0 } };
    measurement_set<> meas;
    simulation_stats stats;
    stats.cumulative_steps = 2;

    // write to an hdf5 file with an unrelated dataset first
    atomic_file_write(path, [](const std::filesystem::path& tmp) {
        hdf5_serializer s { tmp, hdf5_file_mode::truncate };
        s.save_at("unrelated", 42);
    });

    // write checkpoint to the same file in append mode
    atomic_file_write(
        path,
        [&](const std::filesystem::path& tmp) {
            hdf5_serializer s { tmp, hdf5_file_mode::append };
            simplemc_save(s, rng, updates, meas, stats);
        },
        /* copy_existing = */ true);

    // prepare destinations
    xoshiro256ss dst_rng;
    update_set dst_updates { update { stateful_update {}, "u", 1.0 } };
    measurement_set<> dst_meas;
    simulation_stats dst_stats;
    int unrelated = 0;

    // read checkpoint file and the unrelated dataset
    const hdf5_serializer s { path, hdf5_file_mode::read };
    simplemc_load(s, dst_rng, dst_updates, dst_meas, dst_stats);
    s.load_at("unrelated", unrelated);

    // check that the round-trip preserved the checkpoint fields and unrelated dataset
    EXPECT_EQ(dst_stats.cumulative_steps, 2u);
    EXPECT_EQ(unrelated, 42);

    std::filesystem::remove(path);
}

#endif // SIMPLEMC_USE_HDF5
