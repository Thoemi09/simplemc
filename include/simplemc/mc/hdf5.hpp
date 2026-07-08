/**
 * @file
 * @brief HDF5 file I/O for **simplemc-mc**.
 */

#ifndef SIMPLEMC_MC_HDF5_HPP
#define SIMPLEMC_MC_HDF5_HPP

#include <simplemc/mc/concepts.hpp>
#include <simplemc/mc/measurement_set.hpp>
#include <simplemc/mc/simulation_stats.hpp>
#include <simplemc/mc/update_set.hpp>
#include <simplemc/mc/utils.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/serialize/hdf5/hdf5_serializer.hpp>
#include <simplemc/utils/file_io.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <concepts>
#include <filesystem>
#include <functional>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-utils
 * @{
 */

/**
 * @brief No-op callback for the HDF5 file helpers.
 */
struct no_op_hdf5 {
    /**
     * @brief Do nothing.
     */
    constexpr void operator()(const hdf5_serializer&) const noexcept {}
};

/**
 * @brief Write an HDF5 file through a user callback.
 *
 * @details It creates a simplemc::hdf5_serializer with the given (writable) @ref hdf5_file_mode and
 * passes it to the callback which is responsible for serializing actual content.
 *
 * To avoid corrupting an existing file, it uses simplemc::atomic_file_write.
 *
 * It throws a simplemc::simplemc_exception if the file mode is `hdf5_file_mode::read`.
 *
 * @tparam F Callback type.
 * @param path Path to the file.
 * @param f Callback that fills the document via the serializer handle (invocable with an lvalue
 * reference to a simplemc::hdf5_serializer).
 * @param mode File open mode.
 */
template <typename F>
    requires std::invocable<F&, hdf5_serializer&>
void save_hdf5_file(const std::filesystem::path& path, F&& f, hdf5_file_mode mode = hdf5_file_mode::truncate) {
    if (mode == hdf5_file_mode::read) {
        throw simplemc_exception("save_hdf5_file requires a writable file mode");
    }
    atomic_file_write(
        path,
        [&](const std::filesystem::path& tmp) {
            hdf5_serializer s { tmp, mode };
            std::invoke(std::forward<F>(f), s);
        },
        /* copy_existing = */ mode == hdf5_file_mode::append);
}

/**
 * @brief Read an HDF5 file through a user callback.
 *
 * @details It creates a simplemc::hdf5_serializer in read mode (see @ref hdf5_file_mode) and passes
 * it to the callback which is responsible for reading the file's content.
 *
 * @tparam F Callback type.
 * @param path Path to the file.
 * @param f Callback that reads the document via the serializer handle (invocable with a const lvalue
 * reference to a simplemc::hdf5_serializer).
 */
template <typename F>
    requires std::invocable<F&, const hdf5_serializer&>
void load_hdf5_file(const std::filesystem::path& path, F&& f) {
    const hdf5_serializer s { path, hdf5_file_mode::read };
    std::invoke(std::forward<F>(f), s);
}

/**
 * @brief Save an MC checkpoint to an HDF5 file.
 *
 * @details This is a convenience wrapper around simplemc::save_hdf5_file that saves the most common
 * MC components by calling simplemc::simplemc_save(S, const RNG&, const update_set<Us...>&,
 * const measurement_set<Ms...>&, const simulation_stats&).
 *
 * The additional callable can be used to checkpoint other components.
 *
 * @tparam RNG Random number generator type.
 * @tparam Us User update types.
 * @tparam Ms User measurement types.
 * @tparam F Callback type.
 * @param path Path to the checkpoint file.
 * @param rng Random number generator to serialize.
 * @param us Update set to serialize.
 * @param ms Measurement set to serialize.
 * @param stats Simulation statistics to serialize.
 * @param f Optional callback (invocable with an lvalue reference to a simplemc::hdf5_serializer).
 */
template <typename RNG, mc_update... Us, mc_measurement... Ms, typename F = no_op_hdf5>
    requires std::invocable<F&, hdf5_serializer&>
void save_hdf5_checkpoint(const std::filesystem::path& path, const RNG& rng, const update_set<Us...>& us,
    const measurement_set<Ms...>& ms, const simulation_stats& stats, F&& f = {}) {
    save_hdf5_file(path, [&](hdf5_serializer& s) {
        simplemc_save(s, rng, us, ms, stats);
        std::invoke(std::forward<F>(f), s);
    });
}

/**
 * @brief Save an MC checkpoint including an MC configuration component to an HDF5 file.
 *
 * @details Same as simplemc::save_hdf5_checkpoint, but additionally saves an MC configuration to
 * file.
 *
 * @tparam RNG Random number generator type.
 * @tparam Us User update types.
 * @tparam Ms User measurement types.
 * @tparam Cfg MC configuration type.
 * @tparam F Callback type.
 * @param path Path to the checkpoint file.
 * @param rng Random number generator to serialize.
 * @param us Update set to serialize.
 * @param ms Measurement set to serialize.
 * @param stats Simulation statistics to serialize.
 * @param cfg MC configuration to serialize.
 * @param f Optional callback (invocable with an lvalue reference to a simplemc::hdf5_serializer).
 */
template <typename RNG, mc_update... Us, mc_measurement... Ms, typename Cfg, typename F = no_op_hdf5>
    requires save_at_all<Cfg, hdf5_serializer> && std::invocable<F&, hdf5_serializer&>
void save_hdf5_checkpoint(const std::filesystem::path& path, const RNG& rng, const update_set<Us...>& us,
    const measurement_set<Ms...>& ms, const simulation_stats& stats, const Cfg& cfg, F&& f = {}) {
    save_hdf5_checkpoint(path, rng, us, ms, stats, [&](hdf5_serializer& s) {
        s.save_at("config", cfg);
        std::invoke(std::forward<F>(f), s);
    });
}

/**
 * @brief Load an MC checkpoint from an HDF5 file.
 *
 * @details This is a convenience wrapper around simplemc::load_hdf5_file that loads the most common
 * MC components by calling simplemc::simplemc_load(const S&, RNG&, update_set<Us...>&,
 * measurement_set<Ms...>&, simulation_stats&).
 *
 * The additional callable can be used to load other components.
 *
 * @tparam RNG Random number generator type.
 * @tparam Us User update types.
 * @tparam Ms User measurement types.
 * @tparam F Callback type.
 * @param path Path to the checkpoint file.
 * @param rng Random number generator to deserialize into.
 * @param us Update set to deserialize into.
 * @param ms Measurement set to deserialize into.
 * @param stats Simulation statistics to deserialize into.
 * @param f Optional callback (invocable with a const lvalue reference to a
 * simplemc::hdf5_serializer).
 */
template <typename RNG, mc_update... Us, mc_measurement... Ms, typename F = no_op_hdf5>
    requires std::invocable<F&, const hdf5_serializer&>
void load_hdf5_checkpoint(const std::filesystem::path& path, RNG& rng, update_set<Us...>& us,
    measurement_set<Ms...>& ms, simulation_stats& stats, F&& f = {}) {
    load_hdf5_file(path, [&](const hdf5_serializer& s) {
        simplemc_load(s, rng, us, ms, stats);
        std::invoke(std::forward<F>(f), s);
    });
}

/**
 * @brief Load an MC checkpoint including an MC configuration component from an HDF5 file.
 *
 * @details Same as simplemc::load_hdf5_checkpoint, but additionally loads an MC configuration from
 * file.
 *
 * @tparam RNG Random number generator type.
 * @tparam Us User update types.
 * @tparam Ms User measurement types.
 * @tparam Cfg MC configuration type.
 * @tparam F Callback type.
 * @param path Path to the checkpoint file.
 * @param rng Random number generator to deserialize into.
 * @param us Update set to deserialize into.
 * @param ms Measurement set to deserialize into.
 * @param stats Simulation statistics to deserialize into.
 * @param cfg MC configuration to deserialize into.
 * @param f Optional callback (invocable with a const lvalue reference to a
 * simplemc::hdf5_serializer).
 */
template <typename RNG, mc_update... Us, mc_measurement... Ms, typename Cfg, typename F = no_op_hdf5>
    requires load_at_all<Cfg, hdf5_serializer> && std::invocable<F&, const hdf5_serializer&>
void load_hdf5_checkpoint(const std::filesystem::path& path, RNG& rng, update_set<Us...>& us,
    measurement_set<Ms...>& ms, simulation_stats& stats, Cfg& cfg, F&& f = {}) {
    load_hdf5_checkpoint(path, rng, us, ms, stats, [&](const hdf5_serializer& s) {
        s.load_at("config", cfg);
        std::invoke(std::forward<F>(f), s);
    });
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_HDF5_HPP
