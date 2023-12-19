/**
 * @file utils.cpp
 * @brief Utility functions for simplemc-mpi.
 */

#include <simplemc/mpi/utils.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>
#include <mpi.h>

namespace simplemc::mpi {

void check_mpi_call(int errcode, const std::string& mpi_routine) {
    if (errcode != MPI_SUCCESS) {
        throw simplemc_exception(fmt::format("MPI error code: {}", errcode), mpi_routine);
    }
}

} // namespace simplemc::mpi
