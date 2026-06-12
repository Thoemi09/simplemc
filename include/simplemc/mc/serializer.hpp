/**
 * @file
 * @brief Library-wide serializer type used throughout the simplemc-mc sublibrary.
 */

#ifndef SIMPLEMC_MC_SERIALIZER_HPP
#define SIMPLEMC_MC_SERIALIZER_HPP

#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/serialize/variant/variant_serializer.hpp>

#ifdef SIMPLEMC_USE_HDF5
#include <simplemc/serialize/hdf5.hpp> // umbrella: pulls in highfive/eigen.hpp so Eigen-backed
                                       // payloads (e.g. accumulators) are HDF5-writable
#endif

namespace simplemc {

/**
 * @addtogroup simplemc-mc-serialize
 * @{
 */

/**
 * @brief The serializer all simplemc-mc types use for checkpoint and input-config serialization.
 *
 * @details A simplemc::variant_serializer over the closed set of shipped backends: JSON always, plus
 * HDF5 when the library is built with `SIMPLEMC_USE_HDF5=ON`. Because it is a single concrete type
 * that erases its backend at runtime, the simplemc-mc class templates no longer need a serializer
 * template parameter — the backend is chosen at run time by constructing the serializer from a
 * concrete handle (e.g. `mc_serializer{ json_serializer{} }`).
 *
 * The same type drives both the checkpoint and the input-config channels; the two stay independent
 * because they are distinct serializer instances and distinct calls, not distinct types.
 *
 * @note Capability is the **intersection** of the variant's backends (see
 * simplemc::variant_serializer): with HDF5 enabled, a wrapped payload must be serializable by *both*
 * JSON and HDF5 for the type-erased save/load to engage; otherwise the wrapper's serialization is a
 * silent no-op.
 */
#ifdef SIMPLEMC_USE_HDF5
using mc_serializer = variant_serializer<json_serializer, hdf5_serializer>;
#else
using mc_serializer = variant_serializer<json_serializer>;
#endif

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_SERIALIZER_HPP
