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

TEST(MCCallbacks, JsonCheckpointWriterRoundTripsSimulation) {
    const auto path = std::filesystem::temp_directory_path() / "simplemc_test_ckpt.json";
    std::filesystem::remove(path);

    // Build a simulation, populate cumulative stats, then write.
    simulation<> src;
    src.add_update(trivial_update {}, "u", 1.0);
    src.add_measurement(trivial_meas {}, "m");
    {
        auto& mut_stats = const_cast<simulation_stats&>(src.stats()); // NOLINT
        mut_stats.cumulative_steps = 1234;
        mut_stats.cumulative_time = 5.5;
    }

    const auto writer = make_json_checkpoint_writer(src, path);
    writer(simulation_ctx {});
    ASSERT_TRUE(std::filesystem::exists(path));

    // Build a fresh simulation with the same registration, load, verify.
    simulation<> dst;
    dst.add_update(trivial_update {}, "u", 1.0);
    dst.add_measurement(trivial_meas {}, "m");
    load_json_checkpoint(dst, path);
    EXPECT_EQ(dst.stats().cumulative_steps, 1234u);
    EXPECT_DOUBLE_EQ(dst.stats().cumulative_time, 5.5);

    std::filesystem::remove(path);
}

TEST(MCCallbacks, JsonCheckpointWriterRespectsModeOption) {
    const auto path = std::filesystem::temp_directory_path() / "simplemc_test_ckpt.cbor";
    std::filesystem::remove(path);

    simulation<> src;
    src.add_update(trivial_update {}, "u", 1.0);
    {
        auto& mut_stats = const_cast<simulation_stats&>(src.stats()); // NOLINT
        mut_stats.cumulative_steps = 7;
    }

    json_io_options opts { .mode = json_file_mode::cbor };
    auto writer = make_json_checkpoint_writer(src, path, opts);
    writer(simulation_ctx {});
    ASSERT_TRUE(std::filesystem::exists(path));

    simulation<> dst;
    dst.add_update(trivial_update {}, "u", 1.0);
    load_json_checkpoint(dst, path, opts);
    EXPECT_EQ(dst.stats().cumulative_steps, 7u);

    std::filesystem::remove(path);
}

#ifdef SIMPLEMC_USE_HDF5

TEST(MCCallbacks, Hdf5CheckpointWriterRoundTripsSimulation) {
    const auto path = std::filesystem::temp_directory_path() / "simplemc_test_ckpt.h5";
    std::filesystem::remove(path);

    using sim_t = simulation<mc_traits<hdf5_serializer, json_serializer>>;
    sim_t src;
    src.add_update(trivial_update {}, "u", 1.0);
    src.add_measurement(trivial_meas {}, "m");
    {
        auto& mut_stats = const_cast<simulation_stats&>(src.stats()); // NOLINT
        mut_stats.cumulative_steps = 9999;
        mut_stats.cumulative_time = 12.25;
    }

    auto writer = make_hdf5_checkpoint_writer(src, path);
    writer(simulation_ctx {});
    ASSERT_TRUE(std::filesystem::exists(path));

    sim_t dst;
    dst.add_update(trivial_update {}, "u", 1.0);
    dst.add_measurement(trivial_meas {}, "m");
    load_hdf5_checkpoint(dst, path);
    EXPECT_EQ(dst.stats().cumulative_steps, 9999u);
    EXPECT_DOUBLE_EQ(dst.stats().cumulative_time, 12.25);

    std::filesystem::remove(path);
}

#endif // SIMPLEMC_MC_HAVE_HDF5_CALLBACKS
