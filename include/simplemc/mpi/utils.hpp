/**
 * @file
 * @brief Utility functions for simplemc-mpi.
 */

#ifndef SIMPLEMC_MPI_UTILS_HPP
#define SIMPLEMC_MPI_UTILS_HPP

#include <string>

namespace simplemc::mpi {

/**
 * @brief Check the success of an MPI call.
 *
 * @details Throws a simplemc::simplemc_exception if the error code is != MPI_SUCCESS.
 *
 * @param errcode Error code returned by the MPI routine.
 * @param mpi_routine Name of the MPI routine for the error message in the exception.
 */
void check_mpi_call(int errcode, const std::string& mpi_routine = "");

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_UTILS_HPP
