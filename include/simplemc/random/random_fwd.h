/**
 * @file
 * @brief Forward declarations for simplemc-random.
 */

#ifndef SIMPLEMC_RANDOM_RANDOM_FWD_HPP
#define SIMPLEMC_RANDOM_RANDOM_FWD_HPP

#include <simplemc/utils/concepts.hpp>

namespace simplemc {

// Distribution parameter types.
namespace detail {

template <integer_only T>
class dad_param_type;

template <integer_only T>
class dd_param_type;

class urd_param_type;

} // namespace detail

// Distributions.
template <integer_only T>
class discrete_alias_distribution;

template <integer_only T>
class discrete_distribution;

class uniform_real_distribution;

// RNGs.
class splitmix64;

enum class xoshiro256_type;

template <xoshiro256_type>
class xoshiro256;

} // namespace simplemc

#endif // SIMPLEMC_RANDOM_RANDOM_FWD_HPP
