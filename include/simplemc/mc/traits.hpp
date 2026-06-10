/**
 * @file
 * @brief Traits bundle of backend types threaded through the simplemc-mc class templates.
 */

#ifndef SIMPLEMC_MC_TRAITS_HPP
#define SIMPLEMC_MC_TRAITS_HPP

#include <simplemc/mc/concepts.hpp>
#include <simplemc/random/xoshiro256.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-sim
 * @{
 */

/**
 * @brief Bundle of backend types threaded through every simplemc-mc class template.
 *
 * @details All simplemc-mc class templates (simplemc::basic_update, simplemc::update_set,
 * simplemc::simulation, ...) take a single traits parameter rather than spelling each backend type
 * individually. This keeps their signatures stable as the set of mc-wide backend types grows: new
 * typedefs are added here, not bolted on as extra template parameters. The aliases are:
 *
 * - `checkpoint_serializer_type` — serializer used for checkpoint (state) serialization.
 * - `input_config_serializer_type` — serializer used for input-config serialization.
 * - `rng_type` — random number generator type.
 *
 * Any user struct exposing the same three aliases satisfies simplemc::mc_traits_like and can be used
 * in place of this template; `mc_traits` simply provides the defaults and the common spelling, e.g.
 * `simplemc::simulation<simplemc::mc_traits<hdf5_serializer>>` keeps checkpoints in HDF5 while the
 * input config stays in JSON.
 *
 * @tparam S1 Serializer type used for checkpoint serialization.
 * @tparam S2 Serializer type used for input-config serialization.
 * @tparam RNG Random number generator type.
 */
template <serializer S1 = json_serializer, serializer S2 = json_serializer, class RNG = xoshiro256ss>
struct mc_traits {
    /**
     * @brief Serializer type used for checkpoint serialization.
     */
    using checkpoint_serializer_type = S1;

    /**
     * @brief Serializer type used for input-config serialization.
     */
    using input_config_serializer_type = S2;

    /**
     * @brief Random number generator type.
     */
    using rng_type = RNG;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_TRAITS_HPP
