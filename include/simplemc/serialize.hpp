/**
 * @file
 * @brief Include all relevant header files from simplemc-serialize.
 */

#ifndef SIMPLEMC_SERIALIZE_HPP
#define SIMPLEMC_SERIALIZE_HPP

#include <simplemc/config.hpp>

// core
#include <simplemc/serialize/concepts.hpp>

// JSON backend
#include <simplemc/serialize/json.hpp>

// backend-erasing variant serializer
#include <simplemc/serialize/variant.hpp>

// HDF5 backend (opt-in via SIMPLEMC_USE_HDF5=ON)
#ifdef SIMPLEMC_USE_HDF5
#include <simplemc/serialize/hdf5.hpp>
#endif

#endif // SIMPLEMC_SERIALIZE_HPP
