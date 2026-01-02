/**
 * @file
 * @brief Forward declarations for simplemc-random.
 */

#ifndef SIMPLEMC_RANDOM_RANDOM_FWD_HPP
#define SIMPLEMC_RANDOM_RANDOM_FWD_HPP

#include <simplemc/utils/concepts.hpp>

namespace simplemc {

class splitmix64;
enum class xoshiro256_type;
template <xoshiro256_type>
class xoshiro256;

} // namespace simplemc

#endif // SIMPLEMC_RANDOM_RANDOM_FWD_HPP
