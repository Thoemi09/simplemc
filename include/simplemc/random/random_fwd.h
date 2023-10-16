/**
 * @file random_fwd.hpp
 * @brief Forward declarations for simplemc-random.
 */

#ifndef SIMPLEMC_RANDOM_RANDOM_FWD_HPP
#define SIMPLEMC_RANDOM_RANDOM_FWD_HPP

namespace simplemc {

class discrete_alias_distribution;
class discrete_distribution;
class splitmix64;
class uniform_real_distribution;
enum class xoshiro256_type;
template <xoshiro256_type>
class xoshiro256;

} // namespace simplemc

#endif // SIMPLEMC_RANDOM_RANDOM_FWD_HPP
