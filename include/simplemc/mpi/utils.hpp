/**
 * @file
 * @brief Utility functions for simplemc-mpi.
 */

#ifndef SIMPLEMC_MPI_UTILS_HPP
#define SIMPLEMC_MPI_UTILS_HPP

#include <mpi.h>

#include <string_view>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-utils
 * @{
 */

/**
 * @brief Check the success of an MPI call.
 *
 * @details It throws a simplemc::simplemc_exception if the error code is `!= MPI_SUCCESS`.
 *
 * @code{.cpp}
 * int my_rank {};
 * check_mpi_call(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank), "MPI_Comm_rank");
 * @endcode
 *
 * @note This only works if the error handler in MPI is set to `MPI_ERRORS_RETURN`. By default, MPI
 * sets it to `MPI_ERRORS_ARE_FATAL`, which causes the program to abort all connected MPI processes
 * in case of an error (see e.g. simplemc::mpi::comm_set_errhandler).
 *
 * @param errcode Error code returned by the MPI routine.
 * @param mpi_routine Name of the MPI routine that returned the error code.
 */
void check_mpi_call(int errcode, std::string_view mpi_routine = "");

/**
 * @brief Get the address of an object as an `MPI_Aint`.
 *
 * @details It wraps `MPI_Get_address` to obtain the absolute address of a given object. The returned
 * address can be used in datatype constructors and other MPI routines that require addresses.
 *
 * @tparam T Type of the object.
 * @param obj Object to get the address of.
 * @return Absolute address of the object as an `MPI_Aint`.
 */
template <typename T>
[[nodiscard]] MPI_Aint get_address(T const& obj) {
    MPI_Aint addr {};
    check_mpi_call(MPI_Get_address(&obj, &addr), "MPI_Get_address");
    return addr;
}

/**
 * @brief Get the displacement between two objects as an `MPI_Aint`.
 *
 * @details It computes the displacement (difference in addresses) between `obj` and `base` using
 * `MPI_Aint_diff`. This is useful for computing displacements for structured MPI datatypes.
 *
 * @tparam T Type of the object.
 * @tparam U Type of the base object.
 * @param obj Object to compute the displacement for.
 * @param base Base object from which the displacement is measured.
 * @return Displacement of the object relative to the base as an `MPI_Aint`.
 */
template <typename T, typename U>
[[nodiscard]] MPI_Aint get_displacement(T const& obj, U const& base) {
    return MPI_Aint_diff(get_address(obj), get_address(base));
}

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
