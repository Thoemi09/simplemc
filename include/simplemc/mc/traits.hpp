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
 * @addtogroup simplemc-mc-concepts
 * @{
 */

/**
 * @brief Default MC traits class satisfying simplemc::mc_traits_like.
 *
 * @details It comes with the following defaults:
 *
 * - `checkpoint_serializer_type` is set to simplemc::json_serializer.
 * - `input_config_serializer_type` is set to simplemc::json_serializer.
 * - `rng_type` is set to simplemc::xoshiro256ss.
 *
 * Users can either define their own traits struct satisfying simplemc::mc_traits_like or simply
 * change the template parameters of this one. For example,
 *
 * ```cpp
 * using my_traits = simplemc::mc_traits<hdf5_serializer, json_serializer, std::mt19937>;
 * ```
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
