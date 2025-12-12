/**
 * @file
 * @brief Broadcast communications between MPI processes.
 */

#ifndef SIMPLEMC_MPI_BROADCAST_HPP
#define SIMPLEMC_MPI_BROADCAST_HPP

#include <simplemc/mpi/all_reduce.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/mpi/mpi_type.hpp>
#include <simplemc/mpi/utils.hpp>
#include <simplemc/utils/ranges.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <mpi.h>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-coll
 * @{
 */

/**
 * @brief Broadcast a given number of values.
 *
 * @details It calls `MPI_Bcast` if `count > 0`, otherwise it does nothing.
 *
 * If the MPI call fails or if `root` is not a valid process ID, a simplemc::simplemc_exception is
 * thrown.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param comm simplemc::mpi::communicator object.
 * @param values Pointer to the memory to be broadcasted from/into.
 * @param count Number of values to be broadcasted.
 * @param root Root process.
 */
template <mpi_compatible T>
void broadcast(const communicator& comm, T* values, int count, int root) {
    if (count <= 0) {
        return;
    }
    if (root < 0 || root >= comm.size()) {
        throw simplemc_exception("Root process is out of bounds", "mpi::broadcast");
    }
    check_mpi_call(MPI_Bcast(values, count, mpi_type<T>::get(), root, comm), "MPI_Bcast");
}

/**
 * @brief Broadcast a single value.
 *
 * @details It calls simplemc::mpi::broadcast with a count of 1.
 *
 * @code{.cpp}
 * // broadcast a single value from the root process
 * simplemc::mpi::communicator comm {};
 * int value {}, root {};
 *
 * // set root and value on root...
 *
 * simplemc::mpi::broadcast(comm, value, root);
 * @endcode
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param comm simplemc::mpi::communicator object.
 * @param value Value to be broadcasted from/into.
 * @param root Root process.
 */
template <mpi_compatible T>
void broadcast(const communicator& comm, T& value, int root) {
    broadcast(comm, &value, 1, root);
}

/**
 * @brief Broadcast a range.
 *
 * @details It calls simplemc::mpi::broadcast.
 *
 * The ranges must have the same size across all processes, otherwise a simplemc::simplemc_exception
 * is thrown.
 *
 * @code{.cpp}
 * // broadcast a vector of integers from the root process
 * simplemc::mpi::communicator comm {};
 * std::vector<int> data {};
 * int root {};
 *
 * // set data and root...
 *
 * simplemc::mpi::broadcast(comm, data, root);
 * @endcode
 *
 * @tparam R Input/Output range.
 * @param comm simplemc::mpi::communicator object.
 * @param rg Range to be broadcasted from/into.
 * @param root Root process.
 */
template <mpi_range R>
void broadcast(const communicator& comm, R&& rg, int root) { // NOLINT
    if (!all_equal(comm, ranges::size(rg))) {
        throw simplemc_exception("Range sizes are not equal across all processes", "mpi::broadcast");
    }
    broadcast(comm, ranges::data(rg), ranges::size(rg), root);
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_BROADCAST_HPP
