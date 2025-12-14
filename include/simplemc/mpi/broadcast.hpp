/**
 * @file
 * @brief Broadcast communications between MPI processes.
 */

#ifndef SIMPLEMC_MPI_BROADCAST_HPP
#define SIMPLEMC_MPI_BROADCAST_HPP

#include <simplemc/mpi/all_equal.hpp>
#include <simplemc/mpi/all_reduce.hpp>
#include <simplemc/mpi/mpi_type.hpp>
#include <simplemc/mpi/utils.hpp>
#include <simplemc/utils/ranges.hpp>

#include <mpi.h>

#include <cassert>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-coll
 * @{
 */

/**
 * @brief Broadcast data (low-level).
 *
 * @details Thin wrapper around `MPI_Bcast`. Does nothing if `count <= 0`.
 *
 * This overload accepts an explicit `MPI_Datatype`, allowing users to broadcast custom or
 * user-defined MPI datatypes. The caller is responsible for ensuring that `buffer` points to valid
 * memory of the correct type and size.
 *
 * It asserts that `count` and `root` is the same on all processes and that `root` is a valid rank.
 *
 * @param buf Pointer to the buffer to broadcast from (on root) or receive into (on other ranks).
 * @param count Number of elements to broadcast.
 * @param datatype MPI datatype of the elements.
 * @param root Rank of the broadcasting process.
 * @param comm MPI communicator.
 */
inline void broadcast(void* buf, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
    assert(all_equal(count, comm));
    assert(all_equal(root, comm));
    if (count <= 0) {
        return;
    }
    check_mpi_call(MPI_Bcast(buf, count, datatype, root, comm), "MPI_Bcast");
}

/**
 * @brief Broadcast a contiguous array of values.
 *
 * @details It simply calls simplemc::mpi::broadcast with the deduced `MPI_Datatype` from the C++ type
 * `T` (see simplemc::mpi::mpi_type).
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param buf Pointer to the buffer to broadcast from (on root) or receive into (on other ranks).
 * @param count Number of elements to broadcast.
 * @param root Rank of the broadcasting process.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void broadcast(T* buf, int count, int root, MPI_Comm comm) {
    broadcast(static_cast<void*>(buf), count, mpi_type<T>::get(), root, comm);
}

/**
 * @brief Broadcast a single value.
 *
 * @details It simply calls simplemc::mpi::broadcast(T*, int, int, MPI_Comm) with a count of 1.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param value Reference to the value to broadcast from (on root) or receive into (on other ranks).
 * @param root Rank of the broadcasting process.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void broadcast(T& value, int root, MPI_Comm comm) {
    broadcast(&value, 1, root, comm);
}

/**
 * @brief Broadcast a contiguous range.
 *
 * @details  It broadcasts all elements in the input range from root to all other processes by calling 
 * simplemc::mpi::broadcast(T*, int, int, MPI_Comm).
 *
 * @tparam R simplemc::mpi::mpi_range type.
 * @param rg Range to broadcast from (on root) or receive into (on other ranks).
 * @param root Rank of the broadcasting process.
 * @param comm MPI communicator.
 */
template <mpi_range R>
void broadcast(R&& rg, int root, MPI_Comm comm) { // NOLINT (forwarding ref used as in/out param)
    broadcast(ranges::data(rg), static_cast<int>(ranges::size(rg)), root, comm);
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_BROADCAST_HPP
