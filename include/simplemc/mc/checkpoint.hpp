/**
 * @file
 * @brief Backend-agnostic checkpoint I/O and callback for the simplemc::run loop.
 */

#ifndef SIMPLEMC_MC_CHECKPOINT_HPP
#define SIMPLEMC_MC_CHECKPOINT_HPP

#include <simplemc/config.hpp>
#include <simplemc/mc/concepts.hpp>
#include <simplemc/mc/measurement_set.hpp>
#include <simplemc/mc/simulation_ctx.hpp>
#include <simplemc/mc/simulation_stats.hpp>
#include <simplemc/mc/update_set.hpp>
#include <simplemc/mc/utils.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/serialize/json/file_io.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>
#include <fmt/std.h>
#include <nlohmann/json.hpp>

#include <cctype>
#include <filesystem>
#include <optional>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unistd.h>
#include <utility>

#ifdef SIMPLEMC_USE_HDF5
#include <simplemc/serialize/hdf5/hdf5_serializer.hpp>
#endif // SIMPLEMC_USE_HDF5

namespace simplemc {

/**
 * @addtogroup simplemc-mc-callbacks
 * @{
 */

/**
 * @brief Storage backend for a checkpoint.
 */
enum class checkpoint_backend {
    json, ///< JSON backend (text or one of the binary modes).
    hdf5  ///< HDF5 backend (only when built with `SIMPLEMC_USE_HDF5=ON`).
};

/**
 * @brief Runtime backend/format selector for checkpoint I/O.
 *
 * @details A plain value type (no type erasure): `backend` chooses the backend, `json_mode` the JSON
 * on-disk format, and `hdf5_mode` (only when built with HDF5) the HDF5 file open mode.
 */
struct checkpoint_options {
    /// Selected backend.
    checkpoint_backend backend = checkpoint_backend::json;

    /// JSON on-disk format (used when `backend == json`).
    json_file_mode json_mode = json_file_mode::text;

#ifdef SIMPLEMC_USE_HDF5
    /// HDF5 file open mode (used when `backend == hdf5`).
    hdf5_file_mode hdf5_mode = hdf5_file_mode::truncate;
#endif
};

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
        return { .backend = checkpoint_backend::json, .json_mode = json_file_mode::text };
    }
    if (ext == ".bson") {
        return { .backend = checkpoint_backend::json, .json_mode = json_file_mode::bson };
    }
    if (ext == ".cbor") {
        return { .backend = checkpoint_backend::json, .json_mode = json_file_mode::cbor };
    }
    if (ext == ".msgpack" || ext == ".mp") {
        return { .backend = checkpoint_backend::json, .json_mode = json_file_mode::msgpack };
    }
    if (ext == ".ubjson" || ext == ".ubj") {
        return { .backend = checkpoint_backend::json, .json_mode = json_file_mode::ubjson };
    }
    if (ext == ".h5" || ext == ".hdf5") {
#ifdef SIMPLEMC_USE_HDF5
        return { .backend = checkpoint_backend::hdf5 };
#else
        throw simplemc_exception(
            fmt::format("Checkpoint file {} requires the HDF5 backend, but the library was built without HDF5 "
                        "support; rebuild with SIMPLEMC_USE_HDF5=ON",
                path));
#endif
    }
    throw simplemc_exception(fmt::format("Cannot deduce a checkpoint backend from the extension of {}", path));
}

/**
 * @brief Whether a user config type is serializable by at least one checkpoint backend.
 *
 * @tparam C Config type.
 */
template <typename C>
concept checkpoint_config = save_at_all<C, json_serializer>
#ifdef SIMPLEMC_USE_HDF5
    || save_at_all<C, hdf5_serializer>
#endif
    ;

namespace detail {

// Excludes the mode/option types from binding as the deduced Config of the checkpoint overloads.
template <typename T>
concept not_checkpoint_mode = !std::same_as<std::remove_cvref_t<T>, json_file_mode> &&
    !std::same_as<std::remove_cvref_t<T>, checkpoint_options>
#ifdef SIMPLEMC_USE_HDF5
    && !std::same_as<std::remove_cvref_t<T>, hdf5_file_mode>
#endif
    ;

// Sibling temp file (same directory, hence same filesystem) so the final rename is atomic. The PID
// suffix keeps concurrent writers (e.g. several MPI ranks checkpointing to the same path) from
// clobbering each other's temp file.
inline std::filesystem::path checkpoint_tmp_path(const std::filesystem::path& path) {
    auto tmp = path;
    tmp += fmt::format(".tmp.{}", ::getpid());
    return tmp;
}

// Shared config channel of the save impls: persist the optional user config under "config". A void
// `Config` (or a null `config`) skips the channel; a config the backend cannot serialize throws.
template <typename Config, serializer S>
void save_config_at(S& s, const Config* config, std::string_view backend) {
    if constexpr (!std::is_void_v<Config>) {
        if (config != nullptr) {
            if constexpr (save_at_all<Config, S>) {
                s.save_at("config", *config);
            } else {
                throw simplemc_exception(fmt::format("config type is not serializable by the {} backend", backend));
            }
        }
    }
}

// Shared config channel of the load impls: mirror of save_config_at.
template <typename Config, serializer S>
void load_config_at(const S& s, Config* config, std::string_view backend) {
    if constexpr (!std::is_void_v<Config>) {
        if (config != nullptr) {
            if constexpr (load_at_all<Config, S>) {
                s.load_at("config", *config);
            } else {
                throw simplemc_exception(fmt::format("config type is not deserializable by the {} backend", backend));
            }
        }
    }
}

// JSON save: serialize into a sibling temp file, then atomically rename over `path`. `Config = void`
// (with `config == nullptr`) skips the config channel. On failure the temp file is removed.
template <typename Config, typename RNG, mc_update... Us, mc_measurement... Ms>
void save_json_impl(const std::filesystem::path& path, const RNG& rng, const update_set<Us...>& updates,
    const measurement_set<Ms...>& meas, const simulation_stats& stats, const Config* config, json_file_mode mode) {
    const auto tmp = checkpoint_tmp_path(path);
    try {
        json_serializer s;
        simplemc_save(s, rng, updates, meas, stats);
        save_config_at(s, config, "JSON");
        write_json_file(s.root(), tmp, json_io_options { .mode = mode });
        std::filesystem::rename(tmp, path);
    } catch (...) {
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        throw;
    }
}

// JSON load: read `path`, deserialize into the components. `Config = void` skips the config channel.
template <typename Config, typename RNG, mc_update... Us, mc_measurement... Ms>
void load_json_impl(const std::filesystem::path& path, RNG& rng, update_set<Us...>& updates,
    measurement_set<Ms...>& meas, simulation_stats& stats, Config* config, json_file_mode mode) {
    nlohmann::json doc;
    read_json_file(doc, path, json_io_options { .mode = mode });
    const json_serializer s { std::move(doc) };
    simplemc_load(s, rng, updates, meas, stats);
    load_config_at(s, config, "JSON");
}

#ifdef SIMPLEMC_USE_HDF5

// HDF5 save: mirror of save_json_impl. Appending starts from the previous checkpoint's content.
template <typename Config, typename RNG, mc_update... Us, mc_measurement... Ms>
void save_hdf5_impl(const std::filesystem::path& path, const RNG& rng, const update_set<Us...>& updates,
    const measurement_set<Ms...>& meas, const simulation_stats& stats, const Config* config, hdf5_file_mode mode) {
    if (mode == hdf5_file_mode::read) {
        throw simplemc_exception("Cannot save a checkpoint with hdf5_file_mode::read");
    }
    const auto tmp = checkpoint_tmp_path(path);
    try {
        if (mode == hdf5_file_mode::append && std::filesystem::exists(path)) {
            std::filesystem::copy_file(path, tmp, std::filesystem::copy_options::overwrite_existing);
        }
        {
            hdf5_serializer s { tmp, mode };
            simplemc_save(s, rng, updates, meas, stats);
            save_config_at(s, config, "HDF5");
        } // drop the file handle so the HDF5 file is closed before the rename
        std::filesystem::rename(tmp, path);
    } catch (...) {
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        throw;
    }
}

// HDF5 load: the file is always opened read-only.
template <typename Config, typename RNG, mc_update... Us, mc_measurement... Ms>
void load_hdf5_impl(const std::filesystem::path& path, RNG& rng, update_set<Us...>& updates,
    measurement_set<Ms...>& meas, simulation_stats& stats, Config* config) {
    const hdf5_serializer s { path, hdf5_file_mode::read };
    simplemc_load(s, rng, updates, meas, stats);
    load_config_at(s, config, "HDF5");
}

#endif // SIMPLEMC_USE_HDF5

} // namespace detail

// ---------------------------------------------------------------------------------------------------
// Explicit per-backend primitives
// ---------------------------------------------------------------------------------------------------

/**
 * @brief Write a JSON checkpoint of the run components to `path` (atomic temp-file + rename).
 *
 * @tparam RNG Random number generator type.
 * @tparam Us User update types.
 * @tparam Ms User measurement types.
 */
template <typename RNG, mc_update... Us, mc_measurement... Ms>
void write_json_checkpoint(const std::filesystem::path& path, const RNG& rng, const update_set<Us...>& updates,
    const measurement_set<Ms...>& meas, const simulation_stats& stats, json_file_mode mode = json_file_mode::text) {
    detail::save_json_impl<void>(path, rng, updates, meas, stats, nullptr, mode);
}

/**
 * @brief JSON write overload that also persists a user `config` object under `"config"`.
 */
template <typename RNG, typename Config, mc_update... Us, mc_measurement... Ms>
    requires(save_at_all<Config, json_serializer> && detail::not_checkpoint_mode<Config>)
void write_json_checkpoint(const std::filesystem::path& path, const RNG& rng, const update_set<Us...>& updates,
    const measurement_set<Ms...>& meas, const simulation_stats& stats, const Config& config,
    json_file_mode mode = json_file_mode::text) {
    detail::save_json_impl<Config>(path, rng, updates, meas, stats, &config, mode);
}

/**
 * @brief Read a JSON checkpoint from `path` into the run components (patched in place by name).
 */
template <typename RNG, mc_update... Us, mc_measurement... Ms>
void read_json_checkpoint(const std::filesystem::path& path, RNG& rng, update_set<Us...>& updates,
    measurement_set<Ms...>& meas, simulation_stats& stats, json_file_mode mode = json_file_mode::text) {
    detail::load_json_impl<void>(path, rng, updates, meas, stats, nullptr, mode);
}

/**
 * @brief JSON read overload that also restores a user `config` object from `"config"`.
 */
template <typename RNG, typename Config, mc_update... Us, mc_measurement... Ms>
    requires(load_at_all<Config, json_serializer> && detail::not_checkpoint_mode<Config>)
void read_json_checkpoint(const std::filesystem::path& path, RNG& rng, update_set<Us...>& updates,
    measurement_set<Ms...>& meas, simulation_stats& stats, Config& config,
    json_file_mode mode = json_file_mode::text) {
    detail::load_json_impl<Config>(path, rng, updates, meas, stats, &config, mode);
}

#ifdef SIMPLEMC_USE_HDF5

/**
 * @brief Write an HDF5 checkpoint of the run components to `path` (atomic temp-file + rename).
 */
template <typename RNG, mc_update... Us, mc_measurement... Ms>
void write_hdf5_checkpoint(const std::filesystem::path& path, const RNG& rng, const update_set<Us...>& updates,
    const measurement_set<Ms...>& meas, const simulation_stats& stats,
    hdf5_file_mode mode = hdf5_file_mode::truncate) {
    detail::save_hdf5_impl<void>(path, rng, updates, meas, stats, nullptr, mode);
}

/**
 * @brief HDF5 write overload that also persists a user `config` object under `"config"`.
 */
template <typename RNG, typename Config, mc_update... Us, mc_measurement... Ms>
    requires(save_at_all<Config, hdf5_serializer> && detail::not_checkpoint_mode<Config>)
void write_hdf5_checkpoint(const std::filesystem::path& path, const RNG& rng, const update_set<Us...>& updates,
    const measurement_set<Ms...>& meas, const simulation_stats& stats, const Config& config,
    hdf5_file_mode mode = hdf5_file_mode::truncate) {
    detail::save_hdf5_impl<Config>(path, rng, updates, meas, stats, &config, mode);
}

/**
 * @brief Read an HDF5 checkpoint from `path` into the run components (patched in place by name).
 */
template <typename RNG, mc_update... Us, mc_measurement... Ms>
void read_hdf5_checkpoint(const std::filesystem::path& path, RNG& rng, update_set<Us...>& updates,
    measurement_set<Ms...>& meas, simulation_stats& stats) {
    detail::load_hdf5_impl<void>(path, rng, updates, meas, stats, nullptr);
}

/**
 * @brief HDF5 read overload that also restores a user `config` object from `"config"`.
 */
template <typename RNG, typename Config, mc_update... Us, mc_measurement... Ms>
    requires(load_at_all<Config, hdf5_serializer> && detail::not_checkpoint_mode<Config>)
void read_hdf5_checkpoint(const std::filesystem::path& path, RNG& rng, update_set<Us...>& updates,
    measurement_set<Ms...>& meas, simulation_stats& stats, Config& config) {
    detail::load_hdf5_impl<Config>(path, rng, updates, meas, stats, &config);
}

#endif // SIMPLEMC_USE_HDF5

// ---------------------------------------------------------------------------------------------------
// Convenience dispatcher — chooses the backend at runtime (from `opts` or the file extension)
// ---------------------------------------------------------------------------------------------------

/**
 * @brief Save a checkpoint of the run components to `path`, choosing the backend at runtime.
 *
 * @details The backend and format come from `opts`, or are deduced from the file extension via
 * simplemc::deduce_checkpoint_options when `opts` is empty. Forwards to the matching per-backend
 * primitive; writes are atomic.
 */
template <typename RNG, mc_update... Us, mc_measurement... Ms>
void save_checkpoint(const std::filesystem::path& path, const RNG& rng, const update_set<Us...>& updates,
    const measurement_set<Ms...>& meas, const simulation_stats& stats,
    std::optional<checkpoint_options> opts = std::nullopt) {
    const auto o = opts ? *opts : deduce_checkpoint_options(path);
    if (o.backend == checkpoint_backend::json) {
        detail::save_json_impl<void>(path, rng, updates, meas, stats, nullptr, o.json_mode);
    }
#ifdef SIMPLEMC_USE_HDF5
    else {
        detail::save_hdf5_impl<void>(path, rng, updates, meas, stats, nullptr, o.hdf5_mode);
    }
#else
    else {
        throw simplemc_exception("The HDF5 checkpoint backend was requested, but the library was built without "
                                 "HDF5 support; rebuild with SIMPLEMC_USE_HDF5=ON");
    }
#endif
}

/**
 * @brief Dispatcher overload that also persists a user `config` object under `"config"`.
 *
 * @details `Config` need only be serializable by *some* backend; saving through a backend that cannot
 * handle it throws a simplemc::simplemc_exception at runtime.
 */
template <typename RNG, typename Config, mc_update... Us, mc_measurement... Ms>
    requires(checkpoint_config<Config> && detail::not_checkpoint_mode<Config>)
void save_checkpoint(const std::filesystem::path& path, const RNG& rng, const update_set<Us...>& updates,
    const measurement_set<Ms...>& meas, const simulation_stats& stats, const Config& config,
    std::optional<checkpoint_options> opts = std::nullopt) {
    const auto o = opts ? *opts : deduce_checkpoint_options(path);
    if (o.backend == checkpoint_backend::json) {
        detail::save_json_impl<Config>(path, rng, updates, meas, stats, &config, o.json_mode);
    }
#ifdef SIMPLEMC_USE_HDF5
    else {
        detail::save_hdf5_impl<Config>(path, rng, updates, meas, stats, &config, o.hdf5_mode);
    }
#else
    else {
        throw simplemc_exception("The HDF5 checkpoint backend was requested, but the library was built without "
                                 "HDF5 support; rebuild with SIMPLEMC_USE_HDF5=ON");
    }
#endif
}

/**
 * @brief Load a checkpoint from `path` into the run components, choosing the backend at runtime.
 *
 * @details Symmetric to simplemc::save_checkpoint. Update and measurement sets are patched in place, so
 * they must be constructed with the same names before loading. Designed to be called once **before**
 * simplemc::run.
 */
template <typename RNG, mc_update... Us, mc_measurement... Ms>
void load_checkpoint(const std::filesystem::path& path, RNG& rng, update_set<Us...>& updates,
    measurement_set<Ms...>& meas, simulation_stats& stats, std::optional<checkpoint_options> opts = std::nullopt) {
    const auto o = opts ? *opts : deduce_checkpoint_options(path);
    if (o.backend == checkpoint_backend::json) {
        detail::load_json_impl<void>(path, rng, updates, meas, stats, nullptr, o.json_mode);
    }
#ifdef SIMPLEMC_USE_HDF5
    else {
        detail::load_hdf5_impl<void>(path, rng, updates, meas, stats, nullptr);
    }
#else
    else {
        throw simplemc_exception("The HDF5 checkpoint backend was requested, but the library was built without "
                                 "HDF5 support; rebuild with SIMPLEMC_USE_HDF5=ON");
    }
#endif
}

/**
 * @brief Dispatcher overload that also restores a user `config` object from `"config"`.
 */
template <typename RNG, typename Config, mc_update... Us, mc_measurement... Ms>
    requires(checkpoint_config<Config> && detail::not_checkpoint_mode<Config>)
void load_checkpoint(const std::filesystem::path& path, RNG& rng, update_set<Us...>& updates,
    measurement_set<Ms...>& meas, simulation_stats& stats, Config& config,
    std::optional<checkpoint_options> opts = std::nullopt) {
    const auto o = opts ? *opts : deduce_checkpoint_options(path);
    if (o.backend == checkpoint_backend::json) {
        detail::load_json_impl<Config>(path, rng, updates, meas, stats, &config, o.json_mode);
    }
#ifdef SIMPLEMC_USE_HDF5
    else {
        detail::load_hdf5_impl<Config>(path, rng, updates, meas, stats, &config);
    }
#else
    else {
        throw simplemc_exception("The HDF5 checkpoint backend was requested, but the library was built without "
                                 "HDF5 support; rebuild with SIMPLEMC_USE_HDF5=ON");
    }
#endif
}

// ---------------------------------------------------------------------------------------------------
// Checkpoint-writer callback
// ---------------------------------------------------------------------------------------------------

/**
 * @brief Checkpoint-writer callback that serializes the borrowed run components to a file.
 *
 * @details Each invocation forwards to simplemc::save_checkpoint with the stored (already resolved)
 * simplemc::checkpoint_options. The borrowed objects must outlive the callable; they are held by
 * pointer so the checkpoint always reflects the live state at invocation time. Typical usage: install
 * as `on_checkpoint` via simplemc::make_checkpoint_writer.
 *
 * @tparam RNG Random number generator type.
 * @tparam UpdateSet Concrete update-set type.
 * @tparam MeasSet Concrete measurement-set type.
 * @tparam Config Optional extra user state to persist under `"config"`; `void` writes none.
 */
template <typename RNG, typename UpdateSet, typename MeasSet, typename Config = void>
struct checkpoint_writer {
    /// Borrowed RNG. Must outlive this callable.
    const RNG* rng = nullptr;

    /// Borrowed update set. Must outlive this callable.
    const UpdateSet* updates = nullptr;

    /// Borrowed measurement set. Must outlive this callable.
    const MeasSet* meas = nullptr;

    /// Borrowed cumulative statistics. Must outlive this callable.
    const simulation_stats* stats = nullptr;

    /// Optional borrowed user state written under `"config"` (ignored when `Config` is `void`).
    const Config* config = nullptr;

    /// Destination path; atomically replaced on every call.
    std::filesystem::path path {};

    /// Resolved backend/format options (the factories deduce them from `path` upfront).
    checkpoint_options opts {};

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
 * when `opts` is empty — so an unusable path fails here, at setup time, rather than mid-run.
 */
template <typename RNG, typename UpdateSet, typename MeasSet>
[[nodiscard]] checkpoint_writer<RNG, UpdateSet, MeasSet> make_checkpoint_writer(const RNG& rng,
    const UpdateSet& updates, const MeasSet& meas, const simulation_stats& stats, std::filesystem::path path,
    std::optional<checkpoint_options> opts = std::nullopt) {
    auto resolved = opts ? *opts : deduce_checkpoint_options(path);
    return checkpoint_writer<RNG, UpdateSet, MeasSet> {
        .rng = &rng, .updates = &updates, .meas = &meas, .stats = &stats, .path = std::move(path), .opts = resolved
    };
}

/**
 * @brief Convenience factory overload that also persists a borrowed user `config` object.
 */
template <typename RNG, typename UpdateSet, typename MeasSet, typename Config>
    requires(checkpoint_config<Config> && detail::not_checkpoint_mode<Config>)
[[nodiscard]] checkpoint_writer<RNG, UpdateSet, MeasSet, Config> make_checkpoint_writer(const RNG& rng,
    const UpdateSet& updates, const MeasSet& meas, const simulation_stats& stats, const Config* config,
    std::filesystem::path path, std::optional<checkpoint_options> opts = std::nullopt) {
    auto resolved = opts ? *opts : deduce_checkpoint_options(path);
    return checkpoint_writer<RNG, UpdateSet, MeasSet, Config> { .rng = &rng,
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
