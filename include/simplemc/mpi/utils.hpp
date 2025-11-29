/**
 * @file
 * @brief Utility functions for simplemc-mpi.
 */

#ifndef SIMPLEMC_MPI_UTILS_HPP
#define SIMPLEMC_MPI_UTILS_HPP

#include <string_view>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-utils
 * @{
 */

/**
 * @brief Check the success of an MPI call.
 *
 * @details Throws a simplemc::simplemc_exception if the error code is `!= MPI_SUCCESS`.
 *
 * @code{.cpp}
 * int my_rank {};
 * check_mpi_call(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank), "MPI_Comm_rank");
 * @endcode
 *
 * @param errcode Error code returned by the MPI routine.
 * @param mpi_routine Name of the MPI routine that returned the error code.
 */
void check_mpi_call(int errcode, std::string_view mpi_routine = "");

/**
 * @brief Policy that indicates if a wrapped MPI resource (communicator, group, etc) should be freed
 * when going out of scope.
 *
 * @details The policy is used in the constructors of wrapper classes around MPI objects (see e.g.
 * simplemc::mpi::communicator and simplemc::mpi::group).
 *
 * The following options are available:
 * - `take_ownership`: The wrapper is responsible for managing the lifetime of the MPI object, i.e.
 * for releasing any resources.
 * - `attach`: The lifetime of the MPI object is managed somewhere else and the wrapper just uses it.
 */
enum class resource_policy { take_ownership, attach };

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_UTILS_HPP
