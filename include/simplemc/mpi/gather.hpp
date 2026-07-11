// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Wrapper functions for `MPI_Gather`.
 */

#ifndef SIMPLEMC_MPI_GATHER_HPP
#define SIMPLEMC_MPI_GATHER_HPP

#include <simplemc/mpi/communicator.hpp>
#include <simplemc/mpi/mpi_type.hpp>
#include <simplemc/mpi/utils.hpp>
#include <simplemc/utils/ranges.hpp>

#include <mpi.h>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-coll-gather
 * @{
 */

/**
 * @brief Gather data on the root process (low-level).
 *
 * @details Thin wrapper around `MPI_Gather`. See the official MPI documentation for more details,
 * e.g. <a href="https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Gather.3.html">Open MPI manual
 * </a>.
 *
 * @param sendbuf Pointer to the send buffer.
 * @param sendcount Number of elements to send.
 * @param sendtype MPI datatype of the send elements.
 * @param recvbuf Pointer to the receive buffer.
 * @param recvcount Number of elements to receive from each process.
 * @param recvtype MPI datatype of the receive elements.
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
inline void gather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
    MPI_Datatype recvtype, int root, MPI_Comm comm) {
    check_mpi_call(MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm), "MPI_Gather");
}

/**
 * @brief Gather a contiguous array of values on the root process.
 *
 * @details It calls simplemc::mpi::gather with the deduced `MPI_Datatype` from the C++ type `T` (see
 * simplemc::mpi::mpi_type).
 *
 * The given count is used for both send and receive counts.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param sendbuf Pointer to the send buffer.
 * @param recvbuf Pointer to the receive buffer.
 * @param count Number of elements to send/receive per process.
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void gather(const T* sendbuf, T* recvbuf, int count, int root, MPI_Comm comm) {
    gather(sendbuf, count, mpi_type<T>::get(), recvbuf, count, mpi_type<T>::get(), root, comm);
}

/**
 * @brief Gather a contiguous array of values in place on the root process.
 *
 * @details It calls simplemc::mpi::gather with the deduced `MPI_Datatype` from the C++ type `T` (see
 * simplemc::mpi::mpi_type). On the root process, `MPI_IN_PLACE` is used as the send buffer.
 *
 * The given count is used for both send and receive counts.
 *
 * @note On the root process, the elements to be sent must already be stored at their correct final
 * position in the buffer before calling this function.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param buf Pointer to the buffer (send on all processes and receive on root).
 * @param count Number of elements to send/receive per process.
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void gather_in_place(T* buf, int count, int root, MPI_Comm comm) {
    if (comm_rank(comm) == root) {
        gather(MPI_IN_PLACE, count, mpi_type<T>::get(), buf, count, mpi_type<T>::get(), root, comm);
    } else {
        gather(buf, count, mpi_type<T>::get(), buf, count, mpi_type<T>::get(), root, comm);
    }
}

/**
 * @brief Gather a single value on the root process.
 *
 * @details It gathers exactly one element from each process by calling
 * simplemc::mpi::gather(const T*, T*, int, int, MPI_Comm) with a count of 1.
 *
 * @tparam R simplemc::mpi::mpi_range type.
 * @param in_value Input value to gather.
 * @param out_rg Output range to gather into (only valid on root).
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_range R>
void gather(const ranges::range_value_t<R>& in_value, R&& out_rg, int root, MPI_Comm comm) { // NOLINT
    gather(&in_value, ranges::data(out_rg), 1, root, comm);
}

/**
 * @brief Gather a contiguous range on the root process.
 *
 * @details It gathers all elements in the input range and stores the result in the output range by
 * calling simplemc::mpi::gather(const T*, T*, int, int, MPI_Comm).
 *
 * The input range size is used as the count of elements to send/receive per process.
 *
 * @tparam R1 simplemc::mpi::mpi_range type.
 * @tparam R2 simplemc::mpi::mpi_range type.
 * @param in_rg Input range to gather.
 * @param out_rg Output range to gather into (only valid on root).
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_range R1, mpi_range R2>
void gather(R1&& in_rg, R2&& out_rg, int root, MPI_Comm comm) { // NOLINT (ranges need not be forwarded)
    gather(ranges::data(in_rg), ranges::data(out_rg), ranges::size(in_rg), root, comm);
}

/**
 * @brief Gather a contiguous range in place on the root process.
 *
 * @details It gathers all elements in the range in place on the root process such that elements from
 * rank 0 can be found at the beginning of the range, then elements from rank 1, and so on. The
 * function calls simplemc::mpi::gather_in_place(T*, int, int, MPI_Comm).
 *
 * The number of elements to receive per process is taken as the size of the range (on root, it is
 * divided by the number of processes in the communicator).
 *
 * @tparam R simplemc::mpi::mpi_range type.
 * @param rg Range to gather (input on all processes, output on root).
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_range R>
void gather_in_place(R&& rg, int root, MPI_Comm comm) { // NOLINT (ranges need not be forwarded)
    int count = ranges::size(rg);
    if (comm_rank(comm) == root) {
        count /= comm_size(comm);
    }
    gather_in_place(ranges::data(rg), count, root, comm);
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_GATHER_HPP
