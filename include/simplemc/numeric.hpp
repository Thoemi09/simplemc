/**
 * @file numeric.hpp
 * @brief Include all relevant header files from simplemc-numeric.
 */

#ifndef SIMPLEMC_NUMERIC_HPP
#define SIMPLEMC_NUMERIC_HPP

// utils + eigen
#include <simplemc/numeric/utils.hpp>
#include <simplemc/numeric/eigen.hpp>

// Bravais lattice
#include <simplemc/numeric/bravais_lattice.hpp>

// interpolation
#include <simplemc/numeric/cubic_spline_interpolation.hpp>
#include <simplemc/numeric/linear_interpolation.hpp>
#include <simplemc/numeric/polynomial_interpolation.hpp>

// quadrature
#include <simplemc/numeric/quadrature.hpp>

// special functions
#include <simplemc/numeric/linear_map.hpp>
#include <simplemc/numeric/orthogonal_polynomials.hpp>

#endif // SIMPLEMC_NUMERIC_HPP
