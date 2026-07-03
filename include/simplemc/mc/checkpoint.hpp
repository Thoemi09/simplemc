/**
 * @file
 * @brief Backend-agnostic checkpoint callback for the simplemc::run loop.
 */

#ifndef SIMPLEMC_MC_CHECKPOINT_HPP
#define SIMPLEMC_MC_CHECKPOINT_HPP

#include <simplemc/config.hpp>
#include <simplemc/mc/simulation_ctx.hpp>
#include <simplemc/mc/utils.hpp>
#include <simplemc/serialize/json/file_io.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>
#include <fmt/std.h>
#include <nlohmann/json.hpp>

#include <cctype>
#include <filesystem>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

#ifdef SIMPLEMC_USE_HDF5
#include <simplemc/serialize/hdf5/hdf5_serializer.hpp>
#endif // SIMPLEMC_USE_HDF5

namespace simplemc {

/**
 * @addtogroup simplemc-mc-callbacks
 * @{
 */

#ifdef SIMPLEMC_USE_HDF5

/**
 * @brief Options for the HDF5 checkpoint backend.
 */
struct hdf5_io_options {
    /// File open mode for saving. `read` is invalid on the save side and ignored (forced to `read`)
    /// when loading.
    hdf5_file_mode mode = hdf5_file_mode::truncate;
};

/**
 * @brief Runtime backend selector for checkpoint I/O; one alternative per simplemc::mc_serializer
 * backend.
 *
 * @details Holding a simplemc::json_io_options selects the JSON backend (text or one of the binary
 * modes); holding a simplemc::hdf5_io_options selects the HDF5 backend. The HDF5 alternative exists
 * only when the library is built with `SIMPLEMC_USE_HDF5=ON`.
 */
using checkpoint_options = std::variant<json_io_options, hdf5_io_options>;

#else

/**
 * @brief Runtime backend selector for checkpoint I/O; one alternative per simplemc::mc_serializer
 * backend.
 *
 * @details Holding a simplemc::json_io_options selects the JSON backend (text or one of the binary
 * modes); holding a simplemc::hdf5_io_options selects the HDF5 backend. The HDF5 alternative exists
 * only when the library is built with `SIMPLEMC_USE_HDF5=ON`.
 */
using checkpoint_options = std::variant<json_io_options>;

#endif // SIMPLEMC_USE_HDF5

/**
 * @brief Deduce the checkpoint backend and format from a file extension.
 *
 * @details The extension is matched case-insensitively:
 *
 * - `.json` — JSON, human-readable text
 * - `.bson` — JSON, BSON binary mode
 * - `.cbor` — JSON, CBOR binary mode
 * - `.msgpack` / `.mp` — JSON, MessagePack binary mode
 * - `.ubjson` / `.ubj` — JSON, UBJSON binary mode
 * - `.h5` / `.hdf5` — HDF5
 *
 * It throws a simplemc::simplemc_exception on an unrecognized extension, and on `.h5` / `.hdf5` when
 * the library was built without HDF5 support (`SIMPLEMC_USE_HDF5=OFF`).
 *
 * @param path Checkpoint file path whose extension selects the backend.
 * @return Options selecting the deduced backend, with default settings otherwise.
 */
[[nodiscard]] inline checkpoint_options deduce_checkpoint_options(const std::filesystem::path& path) {
    auto ext = path.extension().string();
    for (auto& c : ext) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (ext == ".json") {
        return json_io_options { .mode = json_file_mode::text };
    }
    if (ext == ".bson") {
        return json_io_options { .mode = json_file_mode::bson };
    }
    if (ext == ".cbor") {
        return json_io_options { .mode = json_file_mode::cbor };
    }
    if (ext == ".msgpack" || ext == ".mp") {
        return json_io_options { .mode = json_file_mode::msgpack };
    }
    if (ext == ".ubjson" || ext == ".ubj") {
        return json_io_options { .mode = json_file_mode::ubjson };
    }
    if (ext == ".h5" || ext == ".hdf5") {
#ifdef SIMPLEMC_USE_HDF5
        return hdf5_io_options {};
#else
        throw simplemc_exception(
            fmt::format("Checkpoint file {} requires the HDF5 backend, but the library was built without HDF5 "
                        "support; rebuild with SIMPLEMC_USE_HDF5=ON",
                path));
#endif
    }
    throw simplemc_exception(fmt::format("Cannot deduce a checkpoint backend from the extension of {}", path));
}

namespace detail {

// Excludes the options types from binding as the deduced Config of the save/load overloads.
template <typename T>
concept not_checkpoint_options = !std::same_as<std::remove_cvref_t<T>, checkpoint_options> &&
    !std::same_as<std::remove_cvref_t<T>, std::optional<checkpoint_options>>;

// Sibling temp file (same directory, hence same filesystem) so the final rename is atomic.
inline std::filesystem::path checkpoint_tmp_path(const std::filesystem::path& path) {
    auto tmp = path;
    tmp += ".tmp";
    return tmp;
}

// Shared save implementation: serialize into a sibling temp file via the backend selected by `opts`,
// then atomically rename over `path`. `Config = void` (with `config == nullptr`) skips the config
// channel; on any failure the temp file is removed and the exception rethrown.
template <typename Config, typename RNG>
void save_checkpoint_impl(const std::filesystem::path& path, const checkpoint_options& opts, const RNG& rng,
    const update_set& updates, const measurement_set& meas, const simulation_stats& stats, const Config* config) {
    const auto tmp = checkpoint_tmp_path(path);
    try {
        std::visit(
            [&](const auto& o) {
                using opts_t = std::remove_cvref_t<decltype(o)>;
                if constexpr (std::is_same_v<opts_t, json_io_options>) {
                    mc_serializer ser { json_serializer {} };
                    simplemc_save(ser, rng, updates, meas, stats);
                    if constexpr (!std::is_void_v<Config>) {
                        ser.save_at("config", *config);
                    }
                    write_json_file(std::get<json_serializer>(ser.backend()).root(), tmp, o);
                }
#ifdef SIMPLEMC_USE_HDF5
                else if constexpr (std::is_same_v<opts_t, hdf5_io_options>) {
                    if (o.mode == hdf5_file_mode::read) {
                        throw simplemc_exception("Cannot save a checkpoint with hdf5_file_mode::read");
                    }
                    // Appending must start from the previous checkpoint's content.
                    if (o.mode == hdf5_file_mode::append && std::filesystem::exists(path)) {
                        std::filesystem::copy_file(path, tmp, std::filesystem::copy_options::overwrite_existing);
                    }
                    {
                        mc_serializer ser { hdf5_serializer { tmp, o.mode } };
                        simplemc_save(ser, rng, updates, meas, stats);
                        if constexpr (!std::is_void_v<Config>) {
                            ser.save_at("config", *config);
                        }
                    } // drop the last file handle so the HDF5 file is closed before the rename
                }
#endif
            },
            opts);
        std::filesystem::rename(tmp, path);
    } catch (...) {
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        throw;
    }
}

// Shared load implementation: read `path` via the backend selected by `opts`. `Config = void` (with
// `config == nullptr`) skips the config channel. The HDF5 file is always opened read-only.
template <typename Config, typename RNG>
void load_checkpoint_impl(const std::filesystem::path& path, const checkpoint_options& opts, RNG& rng,
    update_set& updates, measurement_set& meas, simulation_stats& stats, Config* config) {
    std::visit(
        [&](const auto& o) {
            using opts_t = std::remove_cvref_t<decltype(o)>;
            if constexpr (std::is_same_v<opts_t, json_io_options>) {
                nlohmann::json doc;
                read_json_file(doc, path, o);
                const mc_serializer ser { json_serializer { std::move(doc) } };
                simplemc_load(ser, rng, updates, meas, stats);
                if constexpr (!std::is_void_v<Config>) {
                    ser.load_at("config", *config);
                }
            }
#ifdef SIMPLEMC_USE_HDF5
            else if constexpr (std::is_same_v<opts_t, hdf5_io_options>) {
                const mc_serializer ser { hdf5_serializer { path, hdf5_file_mode::read } };
                simplemc_load(ser, rng, updates, meas, stats);
                if constexpr (!std::is_void_v<Config>) {
                    ser.load_at("config", *config);
                }
            }
#endif
        },
        opts);
}

} // namespace detail

/**
 * @brief Save a checkpoint of the run components to `path`, choosing the backend at runtime.
 *
 * @details The backend and format come from `opts`, or are deduced from the file extension via
 * simplemc::deduce_checkpoint_options when `opts` is empty. The components are serialized with the
 * free `simplemc_save(serializer, rng, updates, meas, stats)`.
 *
 * The write is atomic: the checkpoint is first written to a sibling `<path>.tmp` file, which is then
 * renamed over `path`. A failure mid-write never corrupts an existing checkpoint at `path`.
 *
 * @tparam RNG Random number generator type, deduced from `rng`.
 * @param path Destination filesystem path.
 * @param rng RNG to save.
 * @param updates Update set to save.
 * @param meas Measurement set to save.
 * @param stats Cumulative statistics to save.
 * @param opts Backend/format options; empty deduces them from the extension of `path`.
 */
template <typename RNG>
void save_checkpoint(const std::filesystem::path& path, const RNG& rng, const update_set& updates,
    const measurement_set& meas, const simulation_stats& stats, std::optional<checkpoint_options> opts = std::nullopt) {
    detail::save_checkpoint_impl<void>(
        path, opts ? *opts : deduce_checkpoint_options(path), rng, updates, meas, stats, nullptr);
}

/**
 * @brief Overload that also persists a user `config` object under `"config"`.
 *
 * @tparam RNG Random number generator type, deduced from `rng`.
 * @tparam Config User state type, deduced from `config`; must be writable through
 *                simplemc::mc_serializer (i.e. by at least one backend). Saving through a backend
 *                that cannot handle it throws a simplemc::simplemc_exception at runtime.
 * @param path Destination filesystem path.
 * @param rng RNG to save.
 * @param updates Update set to save.
 * @param meas Measurement set to save.
 * @param stats Cumulative statistics to save.
 * @param config User state to persist under `"config"`.
 * @param opts Backend/format options; empty deduces them from the extension of `path`.
 */
template <typename RNG, typename Config>
    requires(save_at_all<Config, mc_serializer> && detail::not_checkpoint_options<Config>)
void save_checkpoint(const std::filesystem::path& path, const RNG& rng, const update_set& updates,
    const measurement_set& meas, const simulation_stats& stats, const Config& config,
    std::optional<checkpoint_options> opts = std::nullopt) {
    detail::save_checkpoint_impl<Config>(
        path, opts ? *opts : deduce_checkpoint_options(path), rng, updates, meas, stats, &config);
}

/**
 * @brief Load a checkpoint from `path` into the run components, choosing the backend at runtime.
 *
 * @details Symmetric to simplemc::save_checkpoint. The backend and format come from `opts`, or are
 * deduced from the file extension via simplemc::deduce_checkpoint_options when `opts` is empty; an
 * HDF5 checkpoint is always opened read-only (any simplemc::hdf5_io_options mode is ignored).
 *
 * Update and measurement sets are patched in place, so they must be registered (with the same names)
 * before loading. Designed to be called once **before** simplemc::run, so it is a free function and
 * not part of simplemc::run_callbacks.
 *
 * @tparam RNG Random number generator type, deduced from `rng`.
 * @param path Source filesystem path.
 * @param rng RNG to restore.
 * @param updates Update set to patch in place.
 * @param meas Measurement set to patch in place.
 * @param stats Cumulative statistics to restore.
 * @param opts Backend/format options; empty deduces them from the extension of `path`.
 */
template <typename RNG>
void load_checkpoint(const std::filesystem::path& path, RNG& rng, update_set& updates, measurement_set& meas,
    simulation_stats& stats, std::optional<checkpoint_options> opts = std::nullopt) {
    detail::load_checkpoint_impl<void>(
        path, opts ? *opts : deduce_checkpoint_options(path), rng, updates, meas, stats, nullptr);
}

/**
 * @brief Overload that also restores a user `config` object from `"config"`.
 *
 * @tparam RNG Random number generator type, deduced from `rng`.
 * @tparam Config User state type, deduced from `config`; must be readable through
 *                simplemc::mc_serializer (i.e. by at least one backend). Loading through a backend
 *                that cannot handle it throws a simplemc::simplemc_exception at runtime.
 * @param path Source filesystem path.
 * @param rng RNG to restore.
 * @param updates Update set to patch in place.
 * @param meas Measurement set to patch in place.
 * @param stats Cumulative statistics to restore.
 * @param config User state to restore from `"config"`.
 * @param opts Backend/format options; empty deduces them from the extension of `path`.
 */
template <typename RNG, typename Config>
    requires(load_at_all<Config, mc_serializer> && detail::not_checkpoint_options<Config>)
void load_checkpoint(const std::filesystem::path& path, RNG& rng, update_set& updates, measurement_set& meas,
    simulation_stats& stats, Config& config, std::optional<checkpoint_options> opts = std::nullopt) {
    detail::load_checkpoint_impl<Config>(
        path, opts ? *opts : deduce_checkpoint_options(path), rng, updates, meas, stats, &config);
}

/**
 * @brief Checkpoint-writer callback that serializes the borrowed run components to a file.
 *
 * @details Each invocation forwards to simplemc::save_checkpoint with the stored (already resolved)
 * simplemc::checkpoint_options, so one writer type covers every backend; the backend is chosen at
 * runtime, by the factory, from the file extension or an explicit options argument. Writes are
 * atomic (temp file + rename). Typical usage: install as `on_checkpoint` via
 * simplemc::make_checkpoint_writer.
 *
 * The borrowed objects must outlive the callable; they are held by pointer so the checkpoint always
 * reflects the live state at invocation time.
 *
 * @tparam RNG Random number generator type.
 * @tparam Config Optional extra user state to persist under `"config"`; `void` writes none. When
 *                non-`void` it must be writable through simplemc::mc_serializer.
 */
template <typename RNG, typename Config = void>
    requires(std::is_void_v<Config> || save_at_all<Config, mc_serializer>)
struct checkpoint_writer {
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

    /// Destination path; atomically replaced on every call.
    std::filesystem::path path {};

    /// Resolved backend/format options (the factories deduce them from `path` upfront).
    checkpoint_options opts { json_io_options {} };

    /**
     * @brief Snapshot the borrowed components (and optional config) into `path`.
     */
    void operator()(const simulation_ctx&) const {
        if constexpr (std::is_void_v<Config>) {
            save_checkpoint(path, *rng, *updates, *meas, *stats, opts);
        } else {
            save_checkpoint(path, *rng, *updates, *meas, *stats, *config, opts);
        }
    }
};

/**
 * @brief Convenience factory: build a component-borrowing simplemc::checkpoint_writer.
 *
 * @details The backend/format options are resolved eagerly — deduced from the extension of `path`
 * when `opts` is empty — so an unusable path fails here, at setup time, rather than mid-run inside
 * the `on_checkpoint` callback.
 *
 * @tparam RNG Random number generator type, deduced from `rng`.
 * @param rng RNG to borrow.
 * @param updates Update set to borrow.
 * @param meas Measurement set to borrow.
 * @param stats Cumulative statistics to borrow.
 * @param path Destination filesystem path.
 * @param opts Backend/format options; empty deduces them from the extension of `path`.
 * @return Configured writer with no extra config object.
 */
template <typename RNG>
[[nodiscard]] checkpoint_writer<RNG> make_checkpoint_writer(const RNG& rng, const update_set& updates,
    const measurement_set& meas, const simulation_stats& stats, std::filesystem::path path,
    std::optional<checkpoint_options> opts = std::nullopt) {
    auto resolved = opts ? *opts : deduce_checkpoint_options(path);
    return checkpoint_writer<RNG> {
        .rng = &rng, .updates = &updates, .meas = &meas, .stats = &stats, .path = std::move(path), .opts = resolved
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
 * @param opts Backend/format options; empty deduces them from the extension of `path`.
 * @return Configured writer that also writes `*config`.
 */
template <typename RNG, typename Config>
[[nodiscard]] checkpoint_writer<RNG, Config> make_checkpoint_writer(const RNG& rng, const update_set& updates,
    const measurement_set& meas, const simulation_stats& stats, const Config* config, std::filesystem::path path,
    std::optional<checkpoint_options> opts = std::nullopt) {
    auto resolved = opts ? *opts : deduce_checkpoint_options(path);
    return checkpoint_writer<RNG, Config> { .rng = &rng,
        .updates = &updates,
        .meas = &meas,
        .stats = &stats,
        .config = config,
        .path = std::move(path),
        .opts = resolved };
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_CHECKPOINT_HPP
