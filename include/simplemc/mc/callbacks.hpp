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
 *   state of any simplemc::simulation (or any other type with `simplemc_save`/`simplemc_load`
 *   overloads for json_serializer) to/from a JSON file.
 * - When the library was built with `SIMPLEMC_USE_HDF5=ON`, simplemc::hdf5_checkpoint_writer /
 *   simplemc::load_hdf5_checkpoint — the same pair against the HDF5 backend.
 */

#ifndef SIMPLEMC_MC_CALLBACKS_HPP
#define SIMPLEMC_MC_CALLBACKS_HPP

#include <simplemc/config.hpp>
#include <simplemc/mc/simulation_ctx.hpp>
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
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-mc
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
    mutable timer<> throttle_clk {};

    /// Whether the first call has been observed (so the first call always prints).
    mutable bool started = false;

    /**
     * @brief Print one status line if rank-gated emission is enabled and the throttle has elapsed.
     */
    void operator()(const simulation_ctx& x) const {
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
            fmt::print(out, "[{}] steps {}/{} ({:.2f}%), time {:.2f}/{:.2f} s ({:.2f}%)\n", prefix, steps,
                max_steps, pct_steps, runtime, max_time, pct_time);
        } else if (max_steps > 0) {
            const double pct = 100.0 * static_cast<double>(steps) / static_cast<double>(max_steps);
            fmt::print(out, "[{}] steps {}/{} ({:.2f}%), time {:.2f} s\n", prefix, steps, max_steps, pct,
                runtime);
        } else if (max_time > 0.0) {
            const double pct = 100.0 * runtime / max_time;
            fmt::print(out, "[{}] steps {}, time {:.2f}/{:.2f} s ({:.2f}%)\n", prefix, steps, runtime,
                max_time, pct);
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
 * @brief Checkpoint-writer callback that serializes a borrowed object to a JSON file.
 *
 * @details Each invocation opens a fresh JSON document, calls `simplemc_save(serializer, *sim)`
 * (ADL on the borrowed object's type), and writes it to `path` using the chosen
 * simplemc::json_io_options (text/cbor/bson/...). Typical usage: install as `on_checkpoint`.
 *
 * The borrowed object must outlive the callable. simplemc::simulation is non-copyable, so this
 * helper takes a pointer rather than a value.
 *
 * @tparam Sim Type to serialize (typically simplemc::simulation<...>).
 */
template <class Sim>
struct json_checkpoint_writer {
    /// Borrowed object to serialize. Must outlive this callable.
    const Sim* sim = nullptr;

    /// Destination path; opened with truncate semantics on every call.
    std::filesystem::path path {};

    /// JSON I/O options (mode, indent).
    json_io_options opts {};

    /**
     * @brief Snapshot `*sim` into `path`.
     */
    void operator()(const simulation_ctx&) const {
        json_serializer ser;
        simplemc_save(ser, *sim);
        write_json_file(ser.root(), path, opts);
    }
};

/**
 * @brief Convenience factory: build a simplemc::json_checkpoint_writer.
 *
 * @tparam Sim Type to serialize.
 * @param sim Borrowed object whose persistent state will be written.
 * @param path Destination filesystem path.
 * @param opts JSON I/O options.
 * @return Configured simplemc::json_checkpoint_writer<Sim>.
 */
template <class Sim>
[[nodiscard]] json_checkpoint_writer<Sim> make_json_checkpoint_writer(
    const Sim& sim, std::filesystem::path path, json_io_options opts = {}) {
    return json_checkpoint_writer<Sim> { .sim = &sim, .path = std::move(path), .opts = opts };
}

/**
 * @brief Free helper to load a previously-written JSON checkpoint into a borrowed object.
 *
 * @details Symmetric to simplemc::json_checkpoint_writer. Designed to be called once **before**
 * simplemc::simulation::run, so it is a free function and not part of `run_callbacks`.
 *
 * @tparam Sim Object type to deserialize into.
 * @param sim Destination object.
 * @param path Source filesystem path.
 * @param opts JSON I/O options.
 */
template <class Sim>
void load_json_checkpoint(Sim& sim, const std::filesystem::path& path, const json_io_options& opts = {}) {
    nlohmann::json doc;
    read_json_file(doc, path, opts);
    const json_serializer ser { std::move(doc) };
    simplemc_load(ser, sim);
}

#ifdef SIMPLEMC_USE_HDF5

/**
 * @brief Checkpoint-writer callback that serializes a borrowed object to an HDF5 file.
 *
 * @details Mirrors simplemc::json_checkpoint_writer for the HDF5 backend. Opens the file with
 * `hdf5_file_mode::truncate` so each checkpoint replaces the previous content. Available only when
 * the HDF5 backend header is reachable (typically when the project was built with
 * `SIMPLEMC_USE_HDF5=ON`).
 *
 * @tparam Sim Type to serialize.
 */
template <class Sim>
struct hdf5_checkpoint_writer {
    /// Borrowed object to serialize.
    const Sim* sim = nullptr;

    /// Destination path.
    std::filesystem::path path {};

    /// File open mode. Defaults to truncate (fresh write each call).
    hdf5_file_mode mode = hdf5_file_mode::truncate;

    /**
     * @brief Snapshot `*sim` into `path`.
     */
    void operator()(const simulation_ctx&) const {
        hdf5_serializer ser { path, mode };
        simplemc_save(ser, *sim);
    }
};

/**
 * @brief Convenience factory: build a simplemc::hdf5_checkpoint_writer.
 */
template <class Sim>
[[nodiscard]] hdf5_checkpoint_writer<Sim> make_hdf5_checkpoint_writer(
    const Sim& sim, std::filesystem::path path, hdf5_file_mode mode = hdf5_file_mode::truncate) {
    return hdf5_checkpoint_writer<Sim> { .sim = &sim, .path = std::move(path), .mode = mode };
}

/**
 * @brief Free helper to load a previously-written HDF5 checkpoint into a borrowed object.
 *
 * @tparam Sim Object type to deserialize into.
 * @param sim Destination object.
 * @param path Source filesystem path.
 */
template <class Sim>
void load_hdf5_checkpoint(Sim& sim, const std::filesystem::path& path) {
    const hdf5_serializer ser { path, hdf5_file_mode::read };
    simplemc_load(ser, sim);
}

#endif // SIMPLEMC_USE_HDF5

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_CALLBACKS_HPP
