/**
 * @file
 * @brief Library-wide serializer type used throughout the **simplemc-mc** sublibrary.
 */

#ifndef SIMPLEMC_MC_SERIALIZER_HPP
#define SIMPLEMC_MC_SERIALIZER_HPP

#include <simplemc/config.hpp>

#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/serialize/variant/variant_serializer.hpp>

#ifdef SIMPLEMC_USE_HDF5
#include <simplemc/serialize/hdf5.hpp>
#endif

namespace simplemc {

/**
 * @addtogroup simplemc-mc-utils
 * @{
 */

/**
 * @brief Serializer type all **simplemc-mc** types use for checkpoint and input-config serialization.
 *
 * @details A simplemc::variant_serializer over the following backends:
 *
 * - JSON (see simplemc::json_serializer),
 * - HDF5 (see simplemc::hdf5_serializer).
 *
 * HDF5 serialization is only supported when the library is built with `SIMPLEMC_USE_HDF5=ON`.
 */
#ifdef SIMPLEMC_USE_HDF5
using mc_serializer = variant_serializer<json_serializer, hdf5_serializer>;
#else
using mc_serializer = variant_serializer<json_serializer>;
#endif

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_SERIALIZER_HPP
