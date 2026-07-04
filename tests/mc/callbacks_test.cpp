#include <simplemc/mc.hpp>
#include <simplemc/mc/checkpoint.hpp>
#include <simplemc/mc/progress_printer.hpp>
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

// Config whose save throws mid-checkpoint, used to verify that a failed write never corrupts an
// existing checkpoint file (atomic temp-file + rename semantics).
struct throwing_config {
    template <serializer S>
    friend void simplemc_save(S&, const throwing_config&) {
        throw simplemc_exception("intentional save failure");
    }
    template <serializer S>
    friend void simplemc_load(const S&, throwing_config&) {}
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
    progress_printer pp { { .prefix = "test", .max_steps = 100 }, false, true, 0.01, fp };
    simulation_ctx x { .steps_done = 42 };
    pp(x);
    const auto text = read_tmpfile(fp);
    std::fclose(fp);
    EXPECT_NE(text.find("test "), std::string::npos);
    EXPECT_NE(text.find("42/100"), std::string::npos);
}

TEST(MCCallbacks, ProgressPrinterBudgetCtor) {
    std::FILE* fp = std::tmpfile();
    ASSERT_NE(fp, nullptr);
    // Convenience ctor: budgets only, builds the progress_line internally.
    progress_printer pp { 100, 0.0, false, true, 0.01, fp };
    pp(simulation_ctx { .steps_done = 25 });
    const auto text = read_tmpfile(fp);
    std::fclose(fp);
    EXPECT_NE(text.find("steps 25/100"), std::string::npos);
    EXPECT_NE(text.find("25.0%"), std::string::npos);
}

TEST(MCCallbacks, ProgressPrinterThrottlesSubsequentCalls) {
    std::FILE* fp = std::tmpfile();
    ASSERT_NE(fp, nullptr);
    // Large progress throttle: after the first emit, the tiny per-call advance stays below it.
    progress_printer pp { { .prefix = "tp", .max_steps = 100 }, false, true, 0.5, fp };
    pp(simulation_ctx { .steps_done = 1 });
    pp(simulation_ctx { .steps_done = 2 });
    pp(simulation_ctx { .steps_done = 3 });
    const auto text = read_tmpfile(fp);
    std::fclose(fp);
    // Exactly one newline expected.
    EXPECT_EQ(std::count(text.begin(), text.end(), '\n'), 1);
}

TEST(MCCallbacks, ProgressPrinterEmitsAfterEnoughProgress) {
    std::FILE* fp = std::tmpfile();
    ASSERT_NE(fp, nullptr);
    // Gate on 25% step progress: first call (0%) prints, then nothing until the fraction advances.
    progress_printer pp { { .prefix = "ep", .max_steps = 100 }, false, true, 0.25, fp };
    pp(simulation_ctx { .steps_done = 0 });  // prints (first call)
    pp(simulation_ctx { .steps_done = 10 }); // +10% < 25% -> suppressed
    pp(simulation_ctx { .steps_done = 20 }); // +20% < 25% -> suppressed
    pp(simulation_ctx { .steps_done = 30 }); // +30% >= 25% -> prints
    const auto text = read_tmpfile(fp);
    std::fclose(fp);
    EXPECT_EQ(std::count(text.begin(), text.end(), '\n'), 2);
}

TEST(MCCallbacks, ProgressPrinterDisabledSuppressesOutput) {
    std::FILE* fp = std::tmpfile();
    ASSERT_NE(fp, nullptr);
    progress_printer pp { { .prefix = "rg", .max_steps = 100 }, false, /*enabled=*/false, 0.01, fp };
    pp(simulation_ctx { .steps_done = 1 });
    pp(simulation_ctx { .steps_done = 2 });
    const auto text = read_tmpfile(fp);
    std::fclose(fp);
    EXPECT_TRUE(text.empty());
}

TEST(MCCallbacks, ProgressPrinterShowsBarWhenEnabled) {
    std::FILE* fp = std::tmpfile();
    ASSERT_NE(fp, nullptr);
    progress_printer pp { { .prefix = "bar", .max_steps = 100, .show_bar = true, .bar_width = 10 }, false, true,
        0.01, fp };
    pp(simulation_ctx { .steps_done = 50 });
    const auto text = read_tmpfile(fp);
    std::fclose(fp);
    EXPECT_NE(text.find("[#####-----]"), std::string::npos);
    EXPECT_NE(text.find("50.0%"), std::string::npos);
}

TEST(MCCallbacks, ProgressLineCustomGlyphsAndRender) {
    // progress_line renders standalone; custom fill/empty glyphs appear in the bar.
    const progress_line line { .prefix = "g", .max_steps = 100, .show_bar = true, .bar_width = 10,
        .fill_char = '*', .empty_char = '.' };
    const auto text = line.render(simulation_ctx { .steps_done = 30 });
    EXPECT_NE(text.find("[***.......]"), std::string::npos);
    EXPECT_NE(text.find("steps 30/100"), std::string::npos);
    EXPECT_NE(text.find("30.0%"), std::string::npos);
    EXPECT_EQ(text.find('\n'), std::string::npos); // render() adds no terminator
}

TEST(MCCallbacks, ProgressPrinterBarUsesMaxOfBudgets) {
    std::FILE* fp = std::tmpfile();
    ASSERT_NE(fp, nullptr);
    // 20/100 steps -> 20%, but a 100 s time budget against an effectively zero runtime keeps time%
    // near 0; the step fraction is the binding (larger) one and should fill the bar.
    progress_printer pp { { .prefix = "b", .max_steps = 100, .max_time = 100.0, .show_bar = true, .bar_width = 10 },
        false, true, 0.01, fp };
    pp(simulation_ctx { .steps_done = 20 });
    const auto text = read_tmpfile(fp);
    std::fclose(fp);
    EXPECT_NE(text.find("[##--------]"), std::string::npos);
    EXPECT_NE(text.find("20.0%"), std::string::npos);
}

TEST(MCCallbacks, ProgressPrinterInplaceUsesCarriageReturn) {
    std::FILE* fp = std::tmpfile();
    ASSERT_NE(fp, nullptr);
    progress_printer pp { { .prefix = "ip", .max_steps = 100 }, /*in_place=*/true, true, 0.01, fp };
    pp(simulation_ctx { .steps_done = 10 });
    pp(simulation_ctx { .steps_done = 20 });
    const auto text = read_tmpfile(fp);
    std::fclose(fp);
    EXPECT_EQ(std::count(text.begin(), text.end(), '\n'), 0);
    EXPECT_GT(std::count(text.begin(), text.end(), '\r'), 0);
}

TEST(MCCallbacks, ProgressPrinterNoBarWithoutBudget) {
    std::FILE* fp = std::tmpfile();
    ASSERT_NE(fp, nullptr);
    progress_printer pp { { .prefix = "nb", .show_bar = true }, false, true, 0.01, fp };
    pp(simulation_ctx { .steps_done = 7 });
    const auto text = read_tmpfile(fp);
    std::fclose(fp);
    EXPECT_EQ(text.find('#'), std::string::npos);
    EXPECT_NE(text.find("steps 7"), std::string::npos);
}

TEST(MCCallbacks, ProgressPrinterAlwaysEmitsCompletion) {
    std::FILE* fp = std::tmpfile();
    ASSERT_NE(fp, nullptr);
    // Large throttle would normally suppress the 90% -> 100% step, but completion bypasses it.
    progress_printer pp { { .prefix = "c", .max_steps = 100 }, false, true, 0.5, fp };
    pp(simulation_ctx { .steps_done = 90 });  // first call prints (90%)
    pp(simulation_ctx { .steps_done = 100 }); // +10% < 50% but complete -> prints
    const auto text = read_tmpfile(fp);
    std::fclose(fp);
    EXPECT_EQ(std::count(text.begin(), text.end(), '\n'), 2);
    EXPECT_NE(text.find("100.0%"), std::string::npos);
}

TEST(MCCallbacks, ProgressPrinterStopsAfterCompletion) {
    std::FILE* fp = std::tmpfile();
    ASSERT_NE(fp, nullptr);
    progress_printer pp { { .prefix = "s", .max_steps = 100 }, false, true, 0.01, fp };
    pp(simulation_ctx { .steps_done = 100 }); // completes -> prints
    pp(simulation_ctx { .steps_done = 100 }); // already finished -> no-op
    pp(simulation_ctx { .steps_done = 120 }); // already finished -> no-op
    const auto text = read_tmpfile(fp);
    std::fclose(fp);
    EXPECT_EQ(std::count(text.begin(), text.end(), '\n'), 1);
}

TEST(MCCallbacks, ProgressPrinterInplaceCompletionCommitsWithNewline) {
    std::FILE* fp = std::tmpfile();
    ASSERT_NE(fp, nullptr);
    progress_printer pp { { .prefix = "ic", .max_steps = 100 }, /*in_place=*/true, true, 0.01, fp };
    pp(simulation_ctx { .steps_done = 50 });  // live redraw -> '\r'
    pp(simulation_ctx { .steps_done = 100 }); // completion -> committed with '\n'
    const auto text = read_tmpfile(fp);
    std::fclose(fp);
    EXPECT_EQ(std::count(text.begin(), text.end(), '\r'), 1);
    EXPECT_EQ(std::count(text.begin(), text.end(), '\n'), 1);
}

TEST(MCCallbacks, CheckpointRoundTripsComponents) {
    const auto path = std::filesystem::temp_directory_path() / "simplemc_test_ckpt.json";
    std::filesystem::remove(path);

    // Assemble the components, populate cumulative stats, then write (backend deduced from ".json").
    xoshiro256ss rng;
    update_set updates { update { trivial_update {}, "u", 1.0 } };
    measurement_set meas { measurement { trivial_meas {}, "m" } };
    simulation_stats stats;
    stats.cumulative_steps = 1234;
    stats.cumulative_time = 5.5;

    const auto writer = make_checkpoint_writer(rng, updates, meas, stats, path);
    writer(simulation_ctx {});
    ASSERT_TRUE(std::filesystem::exists(path));

    // Fresh components with the same registration, load, verify.
    xoshiro256ss dst_rng;
    update_set dst_updates { update { trivial_update {}, "u", 1.0 } };
    measurement_set dst_meas { measurement { trivial_meas {}, "m" } };
    simulation_stats dst_stats;
    load_checkpoint(path, dst_rng, dst_updates, dst_meas, dst_stats);
    EXPECT_EQ(dst_stats.cumulative_steps, 1234u);
    EXPECT_DOUBLE_EQ(dst_stats.cumulative_time, 5.5);

    std::filesystem::remove(path);
}

TEST(MCCallbacks, CheckpointDeducesOptionsFromExtension) {
    const auto json_mode = [](const char* name) {
        const auto opts = deduce_checkpoint_options(name);
        EXPECT_EQ(opts.backend, checkpoint_backend::json);
        return opts.json_mode;
    };
    EXPECT_EQ(json_mode("ck.json"), json_file_mode::text);
    EXPECT_EQ(json_mode("ck.bson"), json_file_mode::bson);
    EXPECT_EQ(json_mode("ck.cbor"), json_file_mode::cbor);
    EXPECT_EQ(json_mode("ck.msgpack"), json_file_mode::msgpack);
    EXPECT_EQ(json_mode("ck.mp"), json_file_mode::msgpack);
    EXPECT_EQ(json_mode("ck.ubjson"), json_file_mode::ubjson);
    EXPECT_EQ(json_mode("ck.ubj"), json_file_mode::ubjson);
    EXPECT_EQ(json_mode("ck.JSON"), json_file_mode::text); // case-insensitive
#ifdef SIMPLEMC_USE_HDF5
    EXPECT_EQ(deduce_checkpoint_options("ck.h5").backend, checkpoint_backend::hdf5);
    EXPECT_EQ(deduce_checkpoint_options("ck.hdf5").backend, checkpoint_backend::hdf5);
#endif
}

TEST(MCCallbacks, CheckpointUnknownExtensionThrows) {
    EXPECT_THROW((void)deduce_checkpoint_options("ck.dat"), simplemc_exception);
    EXPECT_THROW((void)deduce_checkpoint_options("ck"), simplemc_exception);

    // The factory resolves the options eagerly, so a bad path fails at setup time, not mid-run.
    const xoshiro256ss rng;
    const update_set<> updates;
    const measurement_set<> meas;
    const simulation_stats stats;
    EXPECT_THROW((void)make_checkpoint_writer(rng, updates, meas, stats, "ck.dat"), simplemc_exception);
}

TEST(MCCallbacks, CheckpointExplicitOptionsOverrideExtension) {
    // ".ckpt" is not deducible; the explicit options select the JSON backend in MessagePack mode.
    const auto path = std::filesystem::temp_directory_path() / "simplemc_test_ckpt.ckpt";
    std::filesystem::remove(path);
    const checkpoint_options opts { .backend = checkpoint_backend::json, .json_mode = json_file_mode::msgpack };

    xoshiro256ss rng;
    update_set updates { update { trivial_update {}, "u", 1.0 } };
    measurement_set<> meas;
    simulation_stats stats;
    stats.cumulative_steps = 42;

    save_checkpoint(path, rng, updates, meas, stats, opts);
    ASSERT_TRUE(std::filesystem::exists(path));

    xoshiro256ss dst_rng;
    update_set dst_updates { update { trivial_update {}, "u", 1.0 } };
    measurement_set<> dst_meas;
    simulation_stats dst_stats;
    load_checkpoint(path, dst_rng, dst_updates, dst_meas, dst_stats, opts);
    EXPECT_EQ(dst_stats.cumulative_steps, 42u);

    std::filesystem::remove(path);
}

TEST(MCCallbacks, CheckpointRoundTripsConfig) {
    // Deduced binary JSON mode (".cbor") plus the config channel of the save/load overloads.
    const auto path = std::filesystem::temp_directory_path() / "simplemc_test_ckpt.cbor";
    std::filesystem::remove(path);

    xoshiro256ss rng;
    update_set updates { update { trivial_update {}, "u", 1.0 } };
    measurement_set<> meas;
    simulation_stats stats;
    stats.cumulative_steps = 7;
    const run_config cfg { .seed = 5, .beta = 2.5 };

    save_checkpoint(path, rng, updates, meas, stats, cfg);
    ASSERT_TRUE(std::filesystem::exists(path));

    xoshiro256ss dst_rng;
    update_set dst_updates { update { trivial_update {}, "u", 1.0 } };
    measurement_set<> dst_meas;
    simulation_stats dst_stats;
    run_config dst_cfg;
    load_checkpoint(path, dst_rng, dst_updates, dst_meas, dst_stats, dst_cfg);
    EXPECT_EQ(dst_stats.cumulative_steps, 7u);
    EXPECT_EQ(dst_cfg.seed, 5);
    EXPECT_DOUBLE_EQ(dst_cfg.beta, 2.5);

    std::filesystem::remove(path);
}

TEST(MCCallbacks, CheckpointWriteIsAtomic) {
    const auto path = std::filesystem::temp_directory_path() / "simplemc_test_ckpt_atomic.json";
    const auto tmp = std::filesystem::path { path.string() + ".tmp" };
    std::filesystem::remove(path);
    std::filesystem::remove(tmp);

    xoshiro256ss rng;
    update_set updates { update { trivial_update {}, "u", 1.0 } };
    measurement_set<> meas;
    simulation_stats stats;
    stats.cumulative_steps = 111;
    save_checkpoint(path, rng, updates, meas, stats);
    ASSERT_TRUE(std::filesystem::exists(path));

    // A failing save must leave the previous checkpoint intact and clean up its temp file.
    stats.cumulative_steps = 222;
    EXPECT_THROW(save_checkpoint(path, rng, updates, meas, stats, throwing_config {}), simplemc_exception);
    EXPECT_FALSE(std::filesystem::exists(tmp));

    xoshiro256ss dst_rng;
    update_set dst_updates { update { trivial_update {}, "u", 1.0 } };
    measurement_set<> dst_meas;
    simulation_stats dst_stats;
    load_checkpoint(path, dst_rng, dst_updates, dst_meas, dst_stats);
    EXPECT_EQ(dst_stats.cumulative_steps, 111u);

    std::filesystem::remove(path);
}

#ifdef SIMPLEMC_USE_HDF5

TEST(MCCallbacks, Hdf5CheckpointRoundTripsComponentsAndConfig) {
    const auto path = std::filesystem::temp_directory_path() / "simplemc_test_ckpt.h5";
    std::filesystem::remove(path);

    xoshiro256ss rng;
    update_set updates { update { trivial_update {}, "u", 1.0 } };
    measurement_set meas { measurement { trivial_meas {}, "m" } };
    simulation_stats stats;
    stats.cumulative_steps = 9999;
    stats.cumulative_time = 12.25;
    run_config cfg { .seed = 3, .beta = 1.5 };

    // Backend deduced from ".h5"; the writer carries the resolved options.
    auto writer = make_checkpoint_writer(rng, updates, meas, stats, &cfg, path);
    writer(simulation_ctx {});
    ASSERT_TRUE(std::filesystem::exists(path));

    xoshiro256ss dst_rng;
    update_set dst_updates { update { trivial_update {}, "u", 1.0 } };
    measurement_set dst_meas { measurement { trivial_meas {}, "m" } };
    simulation_stats dst_stats;
    run_config dst_cfg;
    load_checkpoint(path, dst_rng, dst_updates, dst_meas, dst_stats, dst_cfg);
    EXPECT_EQ(dst_stats.cumulative_steps, 9999u);
    EXPECT_DOUBLE_EQ(dst_stats.cumulative_time, 12.25);
    EXPECT_EQ(dst_cfg.seed, 3);
    EXPECT_DOUBLE_EQ(dst_cfg.beta, 1.5);

    std::filesystem::remove(path);
}

TEST(MCCallbacks, Hdf5CheckpointWriteIsAtomic) {
    // Genuinely exercises the temp-file + rename path: HDF5 opens its file mid-save.
    const auto path = std::filesystem::temp_directory_path() / "simplemc_test_ckpt_atomic.h5";
    const auto tmp = std::filesystem::path { path.string() + ".tmp" };
    std::filesystem::remove(path);
    std::filesystem::remove(tmp);

    xoshiro256ss rng;
    update_set updates { update { trivial_update {}, "u", 1.0 } };
    measurement_set<> meas;
    simulation_stats stats;
    stats.cumulative_steps = 111;
    save_checkpoint(path, rng, updates, meas, stats);
    ASSERT_TRUE(std::filesystem::exists(path));

    stats.cumulative_steps = 222;
    EXPECT_THROW(save_checkpoint(path, rng, updates, meas, stats, throwing_config {}), simplemc_exception);
    EXPECT_FALSE(std::filesystem::exists(tmp));

    xoshiro256ss dst_rng;
    update_set dst_updates { update { trivial_update {}, "u", 1.0 } };
    measurement_set<> dst_meas;
    simulation_stats dst_stats;
    load_checkpoint(path, dst_rng, dst_updates, dst_meas, dst_stats);
    EXPECT_EQ(dst_stats.cumulative_steps, 111u);

    std::filesystem::remove(path);
}

#else // SIMPLEMC_USE_HDF5

TEST(MCCallbacks, CheckpointH5ExtensionThrowsWithoutHdf5) {
    const xoshiro256ss rng;
    const update_set<> updates;
    const measurement_set<> meas;
    const simulation_stats stats;
    EXPECT_THROW(save_checkpoint("ck.h5", rng, updates, meas, stats), simplemc_exception);
    EXPECT_THROW((void)deduce_checkpoint_options("ck.hdf5"), simplemc_exception);
}

#endif // SIMPLEMC_USE_HDF5
