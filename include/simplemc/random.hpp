/**
 * @file random.hpp
 * @brief Include all relevant header files from simpelmc-random.
 */

#ifndef SIMPLEMC_RANDOM_HPP
#define SIMPLEMC_RANDOM_HPP

// RNGs
#include <simplemc/random/splitmix64.hpp>
#include <simplemc/random/xoshiro256.hpp>

// Distributions
#include <simplemc/random/discrete_alias_distribution.hpp>
#include <simplemc/random/discrete_distribution.hpp>
#include <simplemc/random/uniform_real_distribution.hpp>

// Samples
#include <simplemc/random/samples.hpp>

// Seed RNGs
#include <simplemc/random/seed_rng.hpp>

#endif // SIMPLEMC_RANDOM_HPP
