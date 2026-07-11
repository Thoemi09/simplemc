// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Concepts for simplemc-serialize-hdf5.
 */

#ifndef SIMPLEMC_SERIALIZE_HDF5_CONCEPTS_HPP
#define SIMPLEMC_SERIALIZE_HDF5_CONCEPTS_HPP

#include <simplemc/serialize/concepts.hpp>
#include <simplemc/utils/traits.hpp>

#include <highfive/highfive.hpp>

#include <concepts>
#include <cstddef>
#include <string>
#include <type_traits>

namespace simplemc {

/**
 * @addtogroup simplemc-serialize-hdf5
 * @{
 */

// Forward declaration.
class hdf5_serializer;

/**
 * @brief Concept describing scalar types HighFive can represent as a datatype.
 *
 * @details HighFive supports arithmetic types, `std::complex`, `std::string`, `std::byte`, and
 * `bool`.
 *
 * @tparam T Type to check.
 */
template <typename T>
concept hdf5_atomic_base = std::is_arithmetic_v<T> || is_std_complex_v<T> || std::same_as<T, std::string> ||
    std::same_as<T, std::byte> || std::same_as<T, HighFive::Reference> || std::same_as<T, HighFive::details::Boolean>;

/**
 * @brief Concept describing value types HighFive can serialize directly.
 *
 * @details HighFive's inspector defines `base_type` recursively for every container it supports
 * (`std::vector`, `std::array`, C-arrays, pointers, and `Eigen::Matrix` / `Eigen::Array` / `
 * Eigen::Map`).
 *
 * @note This couples to HighFive internals via `HighFive::details::inspector` and
 * `details::Boolean`).
 *
 * @tparam T Type to check.
 */
template <typename T>
concept hdf5_native = requires { typename HighFive::details::inspector<std::remove_cvref_t<T>>::base_type; } &&
    hdf5_atomic_base<typename HighFive::details::inspector<std::remove_cvref_t<T>>::base_type>;

/**
 * @brief Concept describing value types that can be written with simplemc::hdf5_serializer::save_at.
 *
 * @details This is satisfied when either
 *
 * - type `T` opts into ADL serialization (simplemc::has_simplemc_save with a 
 * simplemc::hdf5_serializer), or
 * - type `T` is natively supported by HighFive (simplemc::hdf5_native).
 *
 * @tparam T Value type.
 */
template <typename T>
concept hdf5_savable = hdf5_native<T> || has_simplemc_save<T, hdf5_serializer>;

/**
 * @brief Concept describing value types that can be read with simplemc::hdf5_serializer::load_at.
 *
 * @details This is satisfied when either
 *
 * - type `T` opts into ADL serialization (simplemc::has_simplemc_load with a
 * simplemc::hdf5_serializer), or
 * - type `T` is natively supported by HighFive (simplemc::hdf5_native).
 *
 * @tparam T Value type.
 */
template <typename T>
concept hdf5_loadable = hdf5_native<T> || has_simplemc_load<T, hdf5_serializer>;

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_SERIALIZE_HDF5_CONCEPTS_HPP
