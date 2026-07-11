// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Concepts for simplemc-serialize-json.
 */

#ifndef SIMPLEMC_SERIALIZE_JSON_CONCEPTS_HPP
#define SIMPLEMC_SERIALIZE_JSON_CONCEPTS_HPP

#include <simplemc/serialize/concepts.hpp>

#include <nlohmann/json.hpp>

namespace simplemc {

/**
 * @addtogroup simplemc-serialize-json
 * @{
 */

// Forward declaration.
class json_serializer;

/**
 * @brief Concept describing value types that can be written with simplemc::json_serializer::save_at.
 *
 * @details This is satisfied when either
 *
 * - type `T` opts into ADL serialization (simplemc::has_simplemc_save with a
 * simplemc::json_serializer, or
 * - type `T` is writable by nlohmann's internal machinery.
 *
 * The second clause inherits nlohmann's own SFINAE constraints.
 *
 * @tparam T Type to check.
 */
template <typename T>
concept json_savable = requires(nlohmann::json& j, const T& v) { j = v; } || has_simplemc_save<T, json_serializer>;

/**
 * @brief Concept describing value types that can be read with simplemc::json_serializer::load_at.
 *
 * @details This is satisfied when either
 *
 * - type `T` opts into ADL serialization (simplemc::has_simplemc_load with a
 * simplemc::json_serializer), or
 * - type `T` is readable by nlohmann's internal machinery.
 *
 * The second clause inherits nlohmann's own SFINAE constraints.
 *
 * @tparam T Type to check.
 */
template <typename T>
concept json_loadable =
    requires(const nlohmann::json& j, T& v) { j.get_to(v); } || has_simplemc_load<T, json_serializer>;

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_SERIALIZE_JSON_CONCEPTS_HPP
