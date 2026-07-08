/**
 * @file
 * @brief JSON file I/O for **simplemc-mc**.
 */

#ifndef SIMPLEMC_MC_JSON_HPP
#define SIMPLEMC_MC_JSON_HPP

#include <simplemc/mc/concepts.hpp>
#include <simplemc/mc/measurement_set.hpp>
#include <simplemc/mc/simulation_params.hpp>
#include <simplemc/mc/simulation_stats.hpp>
#include <simplemc/mc/update_set.hpp>
#include <simplemc/mc/utils.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/serialize/json.hpp>
#include <simplemc/utils/file_io.hpp>

#include <nlohmann/json.hpp>

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
 * @brief No-op callback for the JSON file helpers.
 */
struct no_op_json {
    /**
     * @brief Do nothing.
     */
    constexpr void operator()(const json_serializer&) const noexcept {}
};

/**
 * @brief Write a JSON file through a user callback.
 *
 * @details It creates a simplemc::json_serializer and passes it to the callback which is responsible
 * for serializing actual content. The resulting `nlohmann::json` tree is written to file via
 * simplemc::write_json_file using the given @ref json_io_options.
 *
 * To avoid corrupting an existing file, it uses simplemc::atomic_file_write.
 *
 * @tparam F Callback type.
 * @param path Path to the file.
 * @param f Callback that fills the document via the serializer handle (invocable with an lvalue
 * reference to a simplemc::json_serializer).
 * @param opts IO options.
 */
template <typename F>
    requires std::invocable<F&, json_serializer&>
void save_json_file(const std::filesystem::path& path, F&& f, const json_io_options& opts = {}) {
    atomic_file_write(path, [&](const std::filesystem::path& tmp) {
        json_serializer s;
        std::invoke(std::forward<F>(f), s);
        write_json_file(s.root(), tmp, opts);
    });
}

/**
 * @brief Read a JSON file through a user callback.
 *
 * @details It first reads the file into an `nlohmann::json` tree via simplemc::read_json_file, then
 * moves the tree into a simplemc::json_serializer. The serializer handle is passed to the callback
 * which is responsible for deserializing the content.
 *
 * @tparam F Callback type.
 * @param path Path to the file.
 * @param f Callback that reads the content via the serializer handle (invocable with a const lvalue
 * reference to a simplemc::json_serializer).
 * @param opts IO options.
 */
template <typename F>
    requires std::invocable<F&, const json_serializer&>
void load_json_file(const std::filesystem::path& path, F&& f, const json_io_options& opts = {}) {
    json_serializer s;
    read_json_file(s.root(), path, opts);
    std::invoke(std::forward<F>(f), s);
}

/**
 * @brief Save an MC checkpoint to a JSON file.
 *
 * @details This is a convenience wrapper around simplemc::save_json_file that saves the most common
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
 * @param f Optional callback (invocable with an lvalue reference to a simplemc::json_serializer).
 * @param opts IO options.
 */
template <typename RNG, mc_update... Us, mc_measurement... Ms, typename F = no_op_json>
    requires std::invocable<F&, json_serializer&>
void save_json_checkpoint(const std::filesystem::path& path, const RNG& rng, const update_set<Us...>& us,
    const measurement_set<Ms...>& ms, const simulation_stats& stats, F&& f = {}, const json_io_options& opts = {}) {
    save_json_file(
        path,
        [&](json_serializer& s) {
            simplemc_save(s, rng, us, ms, stats);
            std::invoke(std::forward<F>(f), s);
        },
        opts);
}

/**
 * @brief Save an MC checkpoint including an MC configuration component to a JSON file.
 *
 * @details Same as simplemc::save_json_checkpoint, but additionally saves an MC configuration to
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
 * @param f Optional callback (invocable with an lvalue reference to a simplemc::json_serializer).
 * @param opts IO options.
 */
template <typename RNG, mc_update... Us, mc_measurement... Ms, typename Cfg, typename F = no_op_json>
    requires save_at_all<Cfg, json_serializer> && std::invocable<F&, json_serializer&>
void save_json_checkpoint(const std::filesystem::path& path, const RNG& rng, const update_set<Us...>& us,
    const measurement_set<Ms...>& ms, const simulation_stats& stats, const Cfg& cfg, F&& f = {},
    const json_io_options& opts = {}) {
    save_json_checkpoint(
        path, rng, us, ms, stats,
        [&](json_serializer& s) {
            s.save_at("config", cfg);
            std::invoke(std::forward<F>(f), s);
        },
        opts);
}

/**
 * @brief Load an MC checkpoint from a JSON file.
 *
 * @details This is a convenience wrapper around simplemc::load_json_file that loads the most common
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
 * simplemc::json_serializer).
 * @param opts IO options.
 */
template <typename RNG, mc_update... Us, mc_measurement... Ms, typename F = no_op_json>
    requires std::invocable<F&, const json_serializer&>
void load_json_checkpoint(const std::filesystem::path& path, RNG& rng, update_set<Us...>& us,
    measurement_set<Ms...>& ms, simulation_stats& stats, F&& f = {}, const json_io_options& opts = {}) {
    load_json_file(
        path,
        [&](const json_serializer& s) {
            simplemc_load(s, rng, us, ms, stats);
            std::invoke(std::forward<F>(f), s);
        },
        opts);
}

/**
 * @brief Load an MC checkpoint including an MC configuration component from a JSON file.
 *
 * @details Same as simplemc::load_json_checkpoint, but additionally loads an MC configuration from
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
 * simplemc::json_serializer).
 * @param opts IO options.
 */
template <typename RNG, mc_update... Us, mc_measurement... Ms, typename Cfg, typename F = no_op_json>
    requires load_at_all<Cfg, json_serializer> && std::invocable<F&, const json_serializer&>
void load_json_checkpoint(const std::filesystem::path& path, RNG& rng, update_set<Us...>& us,
    measurement_set<Ms...>& ms, simulation_stats& stats, Cfg& cfg, F&& f = {}, const json_io_options& opts = {}) {
    load_json_checkpoint(
        path, rng, us, ms, stats,
        [&](const json_serializer& s) {
            s.load_at("config", cfg);
            std::invoke(std::forward<F>(f), s);
        },
        opts);
}

/**
 * @brief Save an MC input config to a JSON file.
 *
 * @details This is a convenience wrapper around simplemc::save_json_file that saves the input config
 * of the most common MC components by calling simplemc::simplemc_save_input_config(S,
 * const simulation_params&, const update_set<Us...>&, const measurement_set<Ms...>&).
 *
 * The additional callable can be used to write other components.
 *
 * Input-config files are meant to be edited by hand, so they are always written as indented text
 * JSON.
 *
 * @tparam Us User update types.
 * @tparam Ms User measurement types.
 * @tparam F Callback type.
 * @param path Path to the input-config file.
 * @param params Simulation parameters to serialize.
 * @param us Update set to serialize.
 * @param ms Measurement set to serialize.
 * @param f Optional callback (invocable with an lvalue reference to a simplemc::json_serializer).
 * @param indent Indentation width.
 */
template <mc_update... Us, mc_measurement... Ms, typename F = no_op_json>
    requires std::invocable<F&, json_serializer&>
void save_json_input_config(const std::filesystem::path& path, const simulation_params& params,
    const update_set<Us...>& us, const measurement_set<Ms...>& ms, F&& f = {}, int indent = 4) {
    save_json_file(path,
        [&](json_serializer& s) {
            simplemc_save_input_config(s, params, us, ms);
            std::invoke(std::forward<F>(f), s);
        },
        { .indent = indent });
}

/**
 * @brief Save an MC input config including an MC configuration component to a JSON file.
 *
 * @details Same as simplemc::save_json_input_config, but additionally saves an MC configuration to
 * file through its input-config channel, i.e. the type is required to satisfy
 * simplemc::has_simplemc_save_input_config.
 *
 * @tparam Us User update types.
 * @tparam Ms User measurement types.
 * @tparam Cfg MC configuration type.
 * @tparam F Callback type.
 * @param path Path to the input-config file.
 * @param params Simulation parameters to serialize.
 * @param us Update set to serialize.
 * @param ms Measurement set to serialize.
 * @param cfg MC configuration to serialize.
 * @param f Optional callback (invocable with an lvalue reference to a simplemc::json_serializer).
 * @param indent Indentation width.
 */
template <mc_update... Us, mc_measurement... Ms, typename Cfg, typename F = no_op_json>
    requires has_simplemc_save_input_config<Cfg, json_serializer> && std::invocable<F&, json_serializer&>
void save_json_input_config(const std::filesystem::path& path, const simulation_params& params,
    const update_set<Us...>& us, const measurement_set<Ms...>& ms, const Cfg& cfg, F&& f = {}, int indent = 4) {
    save_json_input_config(
        path, params, us, ms,
        [&](json_serializer& s) {
            simplemc_save_input_config(s["config"], cfg);
            std::invoke(std::forward<F>(f), s);
        },
        indent);
}

/**
 * @brief Load an MC input config from a JSON file.
 *
 * @details This is a convenience wrapper around simplemc::load_json_file that loads the input config
 * of the most common MC components by calling simplemc::simplemc_load_input_config(const S&,
 * simulation_params&, update_set<Us...>&, measurement_set<Ms...>&).
 *
 * The additional callable can be used to load other components.
 *
 * @tparam Us User update types.
 * @tparam Ms User measurement types.
 * @tparam F Callback type.
 * @param path Path to the input-config file.
 * @param params Simulation parameters to deserialize into.
 * @param us Update set to deserialize into.
 * @param ms Measurement set to deserialize into.
 * @param f Optional callback (invocable with a const lvalue reference to a
 * simplemc::json_serializer).
 */
template <mc_update... Us, mc_measurement... Ms, typename F = no_op_json>
    requires std::invocable<F&, const json_serializer&>
void load_json_input_config(const std::filesystem::path& path, simulation_params& params, update_set<Us...>& us,
    measurement_set<Ms...>& ms, F&& f = {}) {
    load_json_file(path, [&](const json_serializer& s) {
        simplemc_load_input_config(s, params, us, ms);
        std::invoke(std::forward<F>(f), s);
    });
}

/**
 * @brief Load an MC input config including an MC configuration component from a JSON file.
 *
 * @details Same as simplemc::load_json_input_config, but additionally tries to load an MC
 * configuration from file through its input-config channel, i.e. the type is required to satisfy
 * simplemc::has_simplemc_load_input_config.
 *
 * @tparam Us User update types.
 * @tparam Ms User measurement types.
 * @tparam Cfg MC configuration type.
 * @tparam F Callback type.
 * @param path Path to the input-config file.
 * @param params Simulation parameters to deserialize into.
 * @param us Update set to deserialize into.
 * @param ms Measurement set to deserialize into.
 * @param cfg MC configuration to deserialize into.
 * @param f Optional callback (invocable with a const lvalue reference to a
 * simplemc::json_serializer).
 */
template <mc_update... Us, mc_measurement... Ms, typename Cfg, typename F = no_op_json>
    requires has_simplemc_load_input_config<Cfg, json_serializer> && std::invocable<F&, const json_serializer&>
void load_json_input_config(const std::filesystem::path& path, simulation_params& params, update_set<Us...>& us,
    measurement_set<Ms...>& ms, Cfg& cfg, F&& f = {}) {
    load_json_input_config(path, params, us, ms, [&](const json_serializer& s) {
        if (s.has("config")) {
            simplemc_load_input_config(s["config"], cfg);
        }
        std::invoke(std::forward<F>(f), s);
    });
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_JSON_HPP
