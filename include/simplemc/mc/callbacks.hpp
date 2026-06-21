/**
 * @file
 * @brief Opt-in callback helpers for the simplemc::run loop.
 *
 * @details Plug-in callables that drop into simplemc::run_callbacks slots. Each is a strongly-typed
 * struct with a `operator()(const simulation_stats&)` so the compiler can inline it in place — no
 * `std::function` indirection. Bundled helpers:
 *
 * - simplemc::progress_printer — throttled status line with optional MPI rank gating, plus the
 *   convenience factories simplemc::make_step_progress_printer and
 *   simplemc::make_time_progress_printer.
 * - simplemc::json_checkpoint_writer / simplemc::load_json_checkpoint — write/read the persistent
 *   state of the run components (RNG, update_set, measurement_set, simulation_stats) plus an
 *   optional user-supplied config object, to/from a JSON file.
 * - When the library was built with `SIMPLEMC_USE_HDF5=ON`, simplemc::hdf5_checkpoint_writer /
 *   simplemc::load_hdf5_checkpoint — the same pair against the HDF5 backend.
 */

#ifndef SIMPLEMC_MC_CALLBACKS_HPP
#define SIMPLEMC_MC_CALLBACKS_HPP

#include <simplemc/config.hpp>
#include <simplemc/mc/run_callbacks.hpp>
#include <simplemc/mc/simulation_ctx.hpp>
#include <simplemc/mc/utils.hpp>
#include <simplemc/serialize/json/file_io.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/timer.hpp>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#ifdef SIMPLEMC_USE_HDF5
#include <simplemc/serialize/hdf5/hdf5_serializer.hpp>
#endif

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <type_traits>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-callbacks
 * @{
 */

/**
 * @brief Throttled progress-printer callback.
 *
 * @details A small callable that prints a one-line status to a `FILE*` no more often than
 * `throttle_sec` seconds of wall clock. It reports the live step count (`ctx.steps_done`) and live
 * wall-clock (`ctx.elapsed()`). If both `max_steps` and `max_time` are zero, it prints just the
 * running counters; otherwise it adds a percentage based on whichever budget is non-zero.
 *
 * Rank gating is **caller-side**: the user passes the local MPI rank via `rank`, and the printer
 * suppresses output on ranks other than `0` when `rank_zero_only` is true. This keeps the struct
 * independent of `<simplemc/mpi.hpp>`, so it works in single-process builds and in MPI builds alike.
 *
 * Throttle state and the wall-clock timer are `mutable` so a const-qualified `run_callbacks` (as
 * held inside the run loop) can still update them.
 */
struct progress_printer {
    /// Destination file handle (defaults to stdout). Must outlive this callable.
    std::FILE* out = stdout;

    /// Optional label prepended to each printed line.
    std::string prefix = "progress";

    /// Minimum wall-clock seconds between two consecutive prints.
    double throttle_sec = 1.0;

    /// Target number of steps for percentage reporting; `0` disables the steps percentage.
    std::uint64_t max_steps = 0;

    /// Target wall-clock seconds for percentage reporting; `0.0` disables the time percentage.
    double max_time = 0.0;

    /// Local MPI rank (default `0`, single-process).
    int rank = 0;

    /// If `true`, only rank `0` prints. Caller must set `rank` correctly for this to gate.
    bool rank_zero_only = false;

    /// Wall-clock timer used to throttle. Reset by emit() after a print.
    timer<> throttle_clk {};

    /// Whether the first call has been observed (so the first call always prints).
    bool started = false;

    /**
     * @brief Print one status line if rank-gated emission is enabled and the throttle has elapsed.
     */
    void operator()(const simulation_ctx& x) {
        if (rank_zero_only && rank != 0) {
            return;
        }
        const double now = throttle_clk.elapsed();
        if (started && now < throttle_sec) {
            return;
        }

        const std::uint64_t steps = x.steps_done;
        const double runtime = x.elapsed();
        if (max_steps > 0 && max_time > 0.0) {
            const double pct_steps = 100.0 * static_cast<double>(steps) / static_cast<double>(max_steps);
            const double pct_time = 100.0 * runtime / max_time;
            fmt::print(out, "[{}] steps {}/{} ({:.2f}%), time {:.2f}/{:.2f} s ({:.2f}%)\n", prefix, steps, max_steps,
                pct_steps, runtime, max_time, pct_time);
        } else if (max_steps > 0) {
            const double pct = 100.0 * static_cast<double>(steps) / static_cast<double>(max_steps);
            fmt::print(out, "[{}] steps {}/{} ({:.2f}%), time {:.2f} s\n", prefix, steps, max_steps, pct, runtime);
        } else if (max_time > 0.0) {
            const double pct = 100.0 * runtime / max_time;
            fmt::print(out, "[{}] steps {}, time {:.2f}/{:.2f} s ({:.2f}%)\n", prefix, steps, runtime, max_time, pct);
        } else {
            fmt::print(out, "[{}] steps {}, time {:.2f} s\n", prefix, steps, runtime);
        }
        std::fflush(out);

        throttle_clk.reset();
        started = true;
    }
};

/**
 * @brief Convenience factory: steps-based progress printer with sensible defaults.
 *
 * @param max_steps Target step count for percentage display.
 * @param throttle_sec Minimum wall-clock seconds between prints. Defaults to `1.0`.
 * @param rank Local MPI rank (defaults to `0`, single-process).
 * @param rank_zero_only Whether to suppress output on non-zero ranks. Defaults to `false`.
 * @return Configured simplemc::progress_printer.
 */
[[nodiscard]] inline progress_printer make_step_progress_printer(
    std::uint64_t max_steps, double throttle_sec = 1.0, int rank = 0, bool rank_zero_only = false) {
    return progress_printer {
        .throttle_sec = throttle_sec, .max_steps = max_steps, .rank = rank, .rank_zero_only = rank_zero_only
    };
}

/**
 * @brief Convenience factory: time-based progress printer with sensible defaults.
 *
 * @param max_time Target wall-clock seconds for percentage display.
 * @param throttle_sec Minimum wall-clock seconds between prints. Defaults to `1.0`.
 * @param rank Local MPI rank (defaults to `0`, single-process).
 * @param rank_zero_only Whether to suppress output on non-zero ranks. Defaults to `false`.
 * @return Configured simplemc::progress_printer.
 */
[[nodiscard]] inline progress_printer make_time_progress_printer(
    double max_time, double throttle_sec = 1.0, int rank = 0, bool rank_zero_only = false) {
    return progress_printer {
        .throttle_sec = throttle_sec, .max_time = max_time, .rank = rank, .rank_zero_only = rank_zero_only
    };
}

/**
 * @brief Checkpoint-writer callback that serializes the borrowed run components to a JSON file.
 *
 * @details Each invocation opens a fresh JSON document, calls the free
 * `simplemc_save(serializer, rng, updates, meas, stats)` over the four borrowed components, optionally
 * appends a user `config` object under `"config"`, and writes the document to `path` using the chosen
 * simplemc::json_io_options (text/cbor/bson/...). Typical usage: install as `on_checkpoint`.
 *
 * The borrowed objects must outlive the callable; they are held by pointer so the checkpoint always
 * reflects the live state at invocation time.
 *
 * @tparam RNG Random number generator type.
 * @tparam Config Optional extra user state to persist under `"config"`; `void` writes none. When
 *                non-`void` it must be writable by the JSON backend (simplemc::save_at_compatible).
 */
template <typename RNG, typename Config = void>
    requires(std::is_void_v<Config> || save_at_compatible<Config, json_serializer>)
struct json_checkpoint_writer {
    /// Borrowed RNG. Must outlive this callable.
    const RNG* rng = nullptr;

    /// Borrowed update set. Must outlive this callable.
    const update_set* updates = nullptr;

    /// Borrowed measurement set. Must outlive this callable.
    const measurement_set* meas = nullptr;

    /// Borrowed cumulative statistics. Must outlive this callable.
    const simulation_stats* stats = nullptr;

    /// Optional borrowed user state written under `"config"` (ignored when `Config` is `void`).
    const Config* config = nullptr;

    /// Destination path; opened with truncate semantics on every call.
    std::filesystem::path path {};

    /// JSON I/O options (mode, indent).
    json_io_options opts {};

    /**
     * @brief Snapshot the borrowed components (and optional config) into `path`.
     */
    void operator()(const simulation_ctx&) const {
        mc_serializer ser { json_serializer {} };
        simplemc_save(ser, *rng, *updates, *meas, *stats);
        auto& json = std::get<json_serializer>(ser.backend());
        if constexpr (!std::is_void_v<Config>) {
            json.save_at("config", *config);
        }
        write_json_file(json.root(), path, opts);
    }
};

/**
 * @brief Convenience factory: build a component-borrowing simplemc::json_checkpoint_writer.
 *
 * @tparam RNG Random number generator type, deduced from `rng`.
 * @param rng RNG to borrow.
 * @param updates Update set to borrow.
 * @param meas Measurement set to borrow.
 * @param stats Cumulative statistics to borrow.
 * @param path Destination filesystem path.
 * @param opts JSON I/O options.
 * @return Configured writer with no extra config object.
 */
template <typename RNG>
[[nodiscard]] json_checkpoint_writer<RNG> make_json_checkpoint_writer(const RNG& rng, const update_set& updates,
    const measurement_set& meas, const simulation_stats& stats, std::filesystem::path path, json_io_options opts = {}) {
    return json_checkpoint_writer<RNG> {
        .rng = &rng, .updates = &updates, .meas = &meas, .stats = &stats, .path = std::move(path), .opts = opts
    };
}

/**
 * @brief Convenience factory overload that also persists a borrowed user `config` object.
 *
 * @tparam RNG Random number generator type, deduced from `rng`.
 * @tparam Config User state type, deduced from `config`.
 * @param rng RNG to borrow.
 * @param updates Update set to borrow.
 * @param meas Measurement set to borrow.
 * @param stats Cumulative statistics to borrow.
 * @param config User state to borrow and persist under `"config"`.
 * @param path Destination filesystem path.
 * @param opts JSON I/O options.
 * @return Configured writer that also writes `*config`.
 */
template <typename RNG, typename Config>
[[nodiscard]] json_checkpoint_writer<RNG, Config> make_json_checkpoint_writer(const RNG& rng, const update_set& updates,
    const measurement_set& meas, const simulation_stats& stats, const Config* config, std::filesystem::path path,
    json_io_options opts = {}) {
    return json_checkpoint_writer<RNG, Config> { .rng = &rng,
        .updates = &updates,
        .meas = &meas,
        .stats = &stats,
        .config = config,
        .path = std::move(path),
        .opts = opts };
}

/**
 * @brief Free helper to load a previously-written JSON checkpoint into the borrowed components.
 *
 * @details Symmetric to simplemc::json_checkpoint_writer. Designed to be called once **before**
 * simplemc::run, so it is a free function and not part of `run_callbacks`.
 *
 * @tparam RNG Random number generator type, deduced from `rng`.
 * @param rng RNG to restore.
 * @param updates Update set to patch in place.
 * @param meas Measurement set to patch in place.
 * @param stats Cumulative statistics to restore.
 * @param path Source filesystem path.
 * @param opts JSON I/O options.
 */
template <typename RNG>
void load_json_checkpoint(RNG& rng, update_set& updates, measurement_set& meas, simulation_stats& stats,
    const std::filesystem::path& path, const json_io_options& opts = {}) {
    nlohmann::json doc;
    read_json_file(doc, path, opts);
    const mc_serializer ser { json_serializer { std::move(doc) } };
    simplemc_load(ser, rng, updates, meas, stats);
}

/**
 * @brief Overload that also restores a borrowed user `config` object from `"config"`.
 *
 * @tparam RNG Random number generator type, deduced from `rng`.
 * @tparam Config User state type, deduced from `config`; must be readable by the JSON backend.
 * @param rng RNG to restore.
 * @param updates Update set to patch in place.
 * @param meas Measurement set to patch in place.
 * @param stats Cumulative statistics to restore.
 * @param config User state to restore from `"config"`.
 * @param path Source filesystem path.
 * @param opts JSON I/O options.
 */
template <typename RNG, typename Config>
    requires load_at_compatible<Config, json_serializer>
void load_json_checkpoint(RNG& rng, update_set& updates, measurement_set& meas, simulation_stats& stats, Config* config,
    const std::filesystem::path& path, const json_io_options& opts = {}) {
    nlohmann::json doc;
    read_json_file(doc, path, opts);
    const mc_serializer ser { json_serializer { std::move(doc) } };
    simplemc_load(ser, rng, updates, meas, stats);
    std::get<json_serializer>(ser.backend()).load_at("config", *config);
}

#ifdef SIMPLEMC_USE_HDF5

/**
 * @brief Checkpoint-writer callback that serializes the borrowed run components to an HDF5 file.
 *
 * @details Mirrors simplemc::json_checkpoint_writer for the HDF5 backend. Opens the file with
 * `hdf5_file_mode::truncate` so each checkpoint replaces the previous content. Available only when
 * the HDF5 backend header is reachable (typically when the project was built with
 * `SIMPLEMC_USE_HDF5=ON`).
 *
 * @tparam RNG Random number generator type.
 * @tparam Config Optional extra user state to persist under `"config"`; `void` writes none. When
 *                non-`void` it must be writable by the HDF5 backend (simplemc::save_at_compatible).
 */
template <typename RNG, typename Config = void>
    requires(std::is_void_v<Config> || save_at_compatible<Config, hdf5_serializer>)
struct hdf5_checkpoint_writer {
    /// Borrowed RNG. Must outlive this callable.
    const RNG* rng = nullptr;

    /// Borrowed update set. Must outlive this callable.
    const update_set* updates = nullptr;

    /// Borrowed measurement set. Must outlive this callable.
    const measurement_set* meas = nullptr;

    /// Borrowed cumulative statistics. Must outlive this callable.
    const simulation_stats* stats = nullptr;

    /// Optional borrowed user state written under `"config"` (ignored when `Config` is `void`).
    const Config* config = nullptr;

    /// Destination path.
    std::filesystem::path path {};

    /// File open mode. Defaults to truncate (fresh write each call).
    hdf5_file_mode mode = hdf5_file_mode::truncate;

    /**
     * @brief Snapshot the borrowed components (and optional config) into `path`.
     */
    void operator()(const simulation_ctx&) const {
        mc_serializer ser { hdf5_serializer { path, mode } };
        simplemc_save(ser, *rng, *updates, *meas, *stats);
        if constexpr (!std::is_void_v<Config>) {
            std::get<hdf5_serializer>(ser.backend()).save_at("config", *config);
        }
    }
};

/**
 * @brief Convenience factory: build a component-borrowing simplemc::hdf5_checkpoint_writer.
 */
template <typename RNG>
[[nodiscard]] hdf5_checkpoint_writer<RNG> make_hdf5_checkpoint_writer(const RNG& rng, const update_set& updates,
    const measurement_set& meas, const simulation_stats& stats, std::filesystem::path path,
    hdf5_file_mode mode = hdf5_file_mode::truncate) {
    return hdf5_checkpoint_writer<RNG> {
        .rng = &rng, .updates = &updates, .meas = &meas, .stats = &stats, .path = std::move(path), .mode = mode
    };
}

/**
 * @brief Convenience factory overload that also persists a borrowed user `config` object.
 */
template <typename RNG, typename Config>
[[nodiscard]] hdf5_checkpoint_writer<RNG, Config> make_hdf5_checkpoint_writer(const RNG& rng, const update_set& updates,
    const measurement_set& meas, const simulation_stats& stats, const Config* config, std::filesystem::path path,
    hdf5_file_mode mode = hdf5_file_mode::truncate) {
    return hdf5_checkpoint_writer<RNG, Config> { .rng = &rng,
        .updates = &updates,
        .meas = &meas,
        .stats = &stats,
        .config = config,
        .path = std::move(path),
        .mode = mode };
}

/**
 * @brief Free helper to load a previously-written HDF5 checkpoint into the borrowed components.
 *
 * @tparam RNG Random number generator type, deduced from `rng`.
 * @param rng RNG to restore.
 * @param updates Update set to patch in place.
 * @param meas Measurement set to patch in place.
 * @param stats Cumulative statistics to restore.
 * @param path Source filesystem path.
 */
template <typename RNG>
void load_hdf5_checkpoint(
    RNG& rng, update_set& updates, measurement_set& meas, simulation_stats& stats, const std::filesystem::path& path) {
    const mc_serializer ser { hdf5_serializer { path, hdf5_file_mode::read } };
    simplemc_load(ser, rng, updates, meas, stats);
}

/**
 * @brief Overload that also restores a borrowed user `config` object from `"config"`.
 */
template <typename RNG, typename Config>
    requires load_at_compatible<Config, hdf5_serializer>
void load_hdf5_checkpoint(RNG& rng, update_set& updates, measurement_set& meas, simulation_stats& stats, Config* config,
    const std::filesystem::path& path) {
    const mc_serializer ser { hdf5_serializer { path, hdf5_file_mode::read } };
    simplemc_load(ser, rng, updates, meas, stats);
    std::get<hdf5_serializer>(ser.backend()).load_at("config", *config);
}

#endif // SIMPLEMC_USE_HDF5

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_CALLBACKS_HPP
