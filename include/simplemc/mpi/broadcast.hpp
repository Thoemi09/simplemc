// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Wrapper functions for `MPI_Bcast`.
 */

#ifndef SIMPLEMC_MPI_BROADCAST_HPP
#define SIMPLEMC_MPI_BROADCAST_HPP

#include <simplemc/mpi/mpi_type.hpp>
#include <simplemc/mpi/utils.hpp>
#include <simplemc/utils/ranges.hpp>

#include <mpi.h>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-coll-bcast
 * @{
 */

/**
 * @brief Broadcast data (low-level).
 *
 * @details Thin wrapper around `MPI_Bcast`. See the official MPI documentation for more details, e.g.
 * <a href="https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Bcast.3.html">Open MPI manual</a>.
 *
 * @param buf Pointer to the buffer to broadcast from (on root) or into (on other ranks).
 * @param count Number of elements to broadcast.
 * @param datatype MPI datatype of the elements.
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
inline void broadcast(void* buf, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
    check_mpi_call(MPI_Bcast(buf, count, datatype, root, comm), "MPI_Bcast");
}

/**
 * @brief Broadcast a contiguous array of values.
 *
 * @details It calls simplemc::mpi::broadcast with the deduced `MPI_Datatype` from the C++ type `T`
 * (see simplemc::mpi::mpi_type).
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param buf Pointer to the buffer to broadcast from (on root) or into (on other ranks).
 * @param count Number of elements to broadcast.
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void broadcast(T* buf, int count, int root, MPI_Comm comm) {
    broadcast(buf, count, mpi_type<T>::get(), root, comm);
}

/**
 * @brief Broadcast a single value.
 *
 * @details It calls simplemc::mpi::broadcast(T*, int, int, MPI_Comm) with a count of 1.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param value Reference to the value to broadcast from (on root) or into (on other ranks).
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void broadcast(T& value, int root, MPI_Comm comm) {
    broadcast(&value, 1, root, comm);
}

/**
 * @brief Broadcast a contiguous range.
 *
 * @details It broadcasts all elements in the input range from root to all other processes by calling
 * simplemc::mpi::broadcast(T*, int, int, MPI_Comm).
 *
 * The range size is used as the count of elements to broadcast.
 *
 * @tparam R simplemc::mpi::mpi_range type.
 * @param rg Range to broadcast from (on root) or into (on other ranks).
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_range R>
void broadcast(R&& rg, int root, MPI_Comm comm) { // NOLINT (forwarding ref as in/out)
    broadcast(ranges::data(rg), ranges::size(rg), root, comm);
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_BROADCAST_HPP
