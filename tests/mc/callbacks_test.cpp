#include <simplemc/mc.hpp>
#include <simplemc/mc/callbacks.hpp>
#include <simplemc/random/xoshiro256.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <filesystem>

using namespace simplemc;

namespace {

// Trivial update used to advance a simulation under callbacks_test. Carries a serializable counter
// so the checkpoint round-trip has something to verify.
struct trivial_update {
    int counter = 0;
    double attempt() {
        ++counter;
        return 1.0;
    }
    void accept() {}
};

template <serializer S>
void simplemc_save(S& s, const trivial_update& u) {
    s.save_at("counter", u.counter);
}
template <serializer S>
void simplemc_load(const S& s, trivial_update& u) {
    s.load_at("counter", u.counter);
}

// Trivial measurement that just increments a counter; also serializable for round-trip checks.
struct trivial_meas {
    int count = 0;
    void measure() { ++count; }
};

template <serializer S>
void simplemc_save(S& s, const trivial_meas& m) {
    s.save_at("count", m.count);
}
template <serializer S>
void simplemc_load(const S& s, trivial_meas& m) {
    s.load_at("count", m.count);
}

// Serializable user-state object attached to a checkpoint via the writer's optional Config slot.
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

// Read all of `fp` (rewinding first) into a std::string for diagnostic assertions.
std::string read_tmpfile(std::FILE* fp) {
    std::fflush(fp);
    std::fseek(fp, 0, SEEK_SET);
    std::string out;
    std::array<char, 256> buf {};
    while (auto n = std::fread(buf.data(), 1, buf.size(), fp)) {
        out.append(buf.data(), n);
    }
    return out;
}

} // namespace

TEST(MCCallbacks, ProgressPrinterPrintsOnFirstCall) {
    std::FILE* fp = std::tmpfile();
    ASSERT_NE(fp, nullptr);
    progress_printer pp { .out = fp, .prefix = "test", .throttle_sec = 60.0, .max_steps = 100 };
    simulation_ctx x { .steps_done = 42 };
    pp(x);
    const auto text = read_tmpfile(fp);
    std::fclose(fp);
    EXPECT_NE(text.find("[test]"), std::string::npos);
    EXPECT_NE(text.find("42/100"), std::string::npos);
}

TEST(MCCallbacks, ProgressPrinterThrottlesSubsequentCalls) {
    std::FILE* fp = std::tmpfile();
    ASSERT_NE(fp, nullptr);
    // Very large throttle: after the first emit, no further line should appear within the test.
    progress_printer pp { .out = fp, .prefix = "tp", .throttle_sec = 60.0, .max_time = 1.0 };
    pp(simulation_ctx { .steps_done = 1 });
    pp(simulation_ctx { .steps_done = 2 });
    pp(simulation_ctx { .steps_done = 3 });
    const auto text = read_tmpfile(fp);
    std::fclose(fp);
    // Exactly one newline expected.
    EXPECT_EQ(std::count(text.begin(), text.end(), '\n'), 1);
}

TEST(MCCallbacks, ProgressPrinterRankGateSuppressesNonZero) {
    std::FILE* fp = std::tmpfile();
    ASSERT_NE(fp, nullptr);
    progress_printer pp { .out = fp, .prefix = "rg", .throttle_sec = 0.0,
        .rank = 3, .rank_zero_only = true };
    pp(simulation_ctx { .steps_done = 1 });
    pp(simulation_ctx { .steps_done = 2 });
    const auto text = read_tmpfile(fp);
    std::fclose(fp);
    EXPECT_TRUE(text.empty());
}

TEST(MCCallbacks, MakeStepProgressPrinterDefaults) {
    const auto pp = make_step_progress_printer(123);
    EXPECT_EQ(pp.max_steps, 123u);
    EXPECT_DOUBLE_EQ(pp.throttle_sec, 1.0);
    EXPECT_EQ(pp.rank, 0);
    EXPECT_FALSE(pp.rank_zero_only);
}

TEST(MCCallbacks, MakeTimeProgressPrinterDefaults) {
    const auto pp = make_time_progress_printer(2.5);
    EXPECT_DOUBLE_EQ(pp.max_time, 2.5);
    EXPECT_DOUBLE_EQ(pp.throttle_sec, 1.0);
}

TEST(MCCallbacks, JsonCheckpointWriterRoundTripsComponents) {
    const auto path = std::filesystem::temp_directory_path() / "simplemc_test_ckpt.json";
    std::filesystem::remove(path);

    // Assemble the components, populate cumulative stats, then write.
    xoshiro256ss rng;
    update_set<> updates;
    measurement_set<> meas;
    simulation_stats stats;
    updates.add({ trivial_update {}, "u", 1.0 });
    meas.add({ trivial_meas {}, "m" });
    stats.cumulative_steps = 1234;
    stats.cumulative_time = 5.5;

    const auto writer = make_json_checkpoint_writer(rng, updates, meas, stats, path);
    writer(simulation_ctx {});
    ASSERT_TRUE(std::filesystem::exists(path));

    // Fresh components with the same registration, load, verify.
    xoshiro256ss dst_rng;
    update_set<> dst_updates;
    measurement_set<> dst_meas;
    simulation_stats dst_stats;
    dst_updates.add({ trivial_update {}, "u", 1.0 });
    dst_meas.add({ trivial_meas {}, "m" });
    load_json_checkpoint(dst_rng, dst_updates, dst_meas, dst_stats, path);
    EXPECT_EQ(dst_stats.cumulative_steps, 1234u);
    EXPECT_DOUBLE_EQ(dst_stats.cumulative_time, 5.5);

    std::filesystem::remove(path);
}

TEST(MCCallbacks, JsonCheckpointWriterRespectsModeOptionAndConfig) {
    const auto path = std::filesystem::temp_directory_path() / "simplemc_test_ckpt.cbor";
    std::filesystem::remove(path);

    xoshiro256ss rng;
    update_set<> updates;
    measurement_set<> meas;
    simulation_stats stats;
    updates.add({ trivial_update {}, "u", 1.0 });
    stats.cumulative_steps = 7;
    run_config cfg { .seed = 5, .beta = 2.5 };

    json_io_options opts { .mode = json_file_mode::cbor };
    auto writer = make_json_checkpoint_writer(rng, updates, meas, stats, &cfg, path, opts);
    writer(simulation_ctx {});
    ASSERT_TRUE(std::filesystem::exists(path));

    xoshiro256ss dst_rng;
    update_set<> dst_updates;
    measurement_set<> dst_meas;
    simulation_stats dst_stats;
    run_config dst_cfg;
    dst_updates.add({ trivial_update {}, "u", 1.0 });
    load_json_checkpoint(dst_rng, dst_updates, dst_meas, dst_stats, &dst_cfg, path, opts);
    EXPECT_EQ(dst_stats.cumulative_steps, 7u);
    EXPECT_EQ(dst_cfg.seed, 5);
    EXPECT_DOUBLE_EQ(dst_cfg.beta, 2.5);

    std::filesystem::remove(path);
}

#ifdef SIMPLEMC_USE_HDF5

TEST(MCCallbacks, Hdf5CheckpointWriterRoundTripsComponentsAndConfig) {
    using traits_t = mc_traits<hdf5_serializer, json_serializer>;
    const auto path = std::filesystem::temp_directory_path() / "simplemc_test_ckpt.h5";
    std::filesystem::remove(path);

    typename traits_t::rng_type rng;
    update_set<traits_t> updates;
    measurement_set<traits_t> meas;
    simulation_stats stats;
    updates.add({ trivial_update {}, "u", 1.0 });
    meas.add({ trivial_meas {}, "m" });
    stats.cumulative_steps = 9999;
    stats.cumulative_time = 12.25;
    run_config cfg { .seed = 3, .beta = 1.5 };

    auto writer = make_hdf5_checkpoint_writer(rng, updates, meas, stats, &cfg, path);
    writer(simulation_ctx {});
    ASSERT_TRUE(std::filesystem::exists(path));

    typename traits_t::rng_type dst_rng;
    update_set<traits_t> dst_updates;
    measurement_set<traits_t> dst_meas;
    simulation_stats dst_stats;
    run_config dst_cfg;
    dst_updates.add({ trivial_update {}, "u", 1.0 });
    dst_meas.add({ trivial_meas {}, "m" });
    load_hdf5_checkpoint(dst_rng, dst_updates, dst_meas, dst_stats, &dst_cfg, path);
    EXPECT_EQ(dst_stats.cumulative_steps, 9999u);
    EXPECT_DOUBLE_EQ(dst_stats.cumulative_time, 12.25);
    EXPECT_EQ(dst_cfg.seed, 3);
    EXPECT_DOUBLE_EQ(dst_cfg.beta, 1.5);

    std::filesystem::remove(path);
}

#endif // SIMPLEMC_USE_HDF5
