/**
 * @file utils.hpp
 * @brief Utility function for simplemc-mpi.
 */

#ifndef SIMPLEMC_MPI_UTILS_HPP
#define SIMPLEMC_MPI_UTILS_HPP

#include <string>

namespace simplemc::mpi {

/**
 * @brief Check the success of a MPI call.
 *
 * @details Throws an mpi_exception if the error code is != MPI_SUCCESS.
 *
 * @param errcode Error code returned by MPI routine.
 * @param mpi_routine Name of the MPI routine.
 */
void check_mpi_call(int errcode, const std::string& mpi_routine = "");

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_UTILS_HPP
