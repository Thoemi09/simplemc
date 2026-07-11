// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Wrapper functions for `MPI_Scatter`.
 */

#ifndef SIMPLEMC_MPI_SCATTER_HPP
#define SIMPLEMC_MPI_SCATTER_HPP

#include <simplemc/mpi/communicator.hpp>
#include <simplemc/mpi/mpi_type.hpp>
#include <simplemc/mpi/utils.hpp>
#include <simplemc/utils/ranges.hpp>

#include <mpi.h>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-coll-scatter
 * @{
 */

/**
 * @brief Scatter data from the root process (low-level).
 *
 * @details Thin wrapper around `MPI_Scatter`. See the official MPI documentation for more details,
 * e.g. <a href="https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Scatter.3.html">Open MPI
 * manual</a>.
 *
 * @param sendbuf Pointer to the send buffer.
 * @param sendcount Number of elements to send to each process.
 * @param sendtype MPI datatype of the send elements.
 * @param recvbuf Pointer to the receive buffer.
 * @param recvcount Number of elements to receive.
 * @param recvtype MPI datatype of the receive elements.
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
inline void scatter(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
    MPI_Datatype recvtype, int root, MPI_Comm comm) {
    check_mpi_call(MPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm), "MPI_Scatter");
}

/**
 * @brief Scatter a contiguous array of values from the root process.
 *
 * @details It calls simplemc::mpi::scatter with the deduced `MPI_Datatype` from the C++ type `T` (see
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
void scatter(const T* sendbuf, T* recvbuf, int count, int root, MPI_Comm comm) {
    scatter(sendbuf, count, mpi_type<T>::get(), recvbuf, count, mpi_type<T>::get(), root, comm);
}

/**
 * @brief Scatter a contiguous array of values in place from the root process.
 *
 * @details It calls simplemc::mpi::scatter with the deduced `MPI_Datatype` from the C++ type `T` (see
 * simplemc::mpi::mpi_type). On the root process, `MPI_IN_PLACE` is used as the receive buffer.
 *
 * The given count is used for both send and receive counts.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param buf Pointer to the buffer (send on root and receive on all processes).
 * @param count Number of elements to send/receive per process.
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void scatter_in_place(T* buf, int count, int root, MPI_Comm comm) {
    if (comm_rank(comm) == root) {
        scatter(buf, count, mpi_type<T>::get(), MPI_IN_PLACE, count, mpi_type<T>::get(), root, comm);
    } else {
        scatter(buf, count, mpi_type<T>::get(), buf, count, mpi_type<T>::get(), root, comm);
    }
}

/**
 * @brief Scatter a range (from the root process) into a single value.
 *
 * @details It scatters exactly one element to each process by calling simplemc::mpi::scatter(const
 * T*, T*, int, int, MPI_Comm) with a count of 1.
 *
 * @tparam R simplemc::mpi::mpi_range type.
 * @param in_rg Input range to scatter (only valid on root).
 * @param out_value Output value to scatter into.
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_range R>
void scatter(R&& in_rg, ranges::range_value_t<R>& out_value, int root, MPI_Comm comm) { // NOLINT
    scatter(ranges::data(in_rg), &out_value, 1, root, comm);
}

/**
 * @brief Scatter a range (from the root process) into a range.
 *
 * @details It scatters all elements from the input range on the root process and stores the result in
 * the output range by calling simplemc::mpi::scatter(const T*, T*, int, int, MPI_Comm).
 *
 * The input range on the root process is divided evenly among all processes in the communicator. The
 * output range size is used as the count of elements to receive per process.
 *
 * @tparam R1 simplemc::mpi::mpi_range type.
 * @tparam R2 simplemc::mpi::mpi_range type.
 * @param in_rg Input range to be scattered (only valid on root).
 * @param out_rg Output range (result of scatter).
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_range R1, mpi_range R2>
void scatter(R1&& in_rg, R2&& out_rg, int root, MPI_Comm comm) { // NOLINT (ranges need not be forwarded)
    scatter(ranges::data(in_rg), ranges::data(out_rg), ranges::size(out_rg), root, comm);
}

/**
 * @brief Scatter a range (from the root process) into a range in place.
 *
 * @details It scatters all elements from the range on the root process in place by calling
 * simplemc::mpi::scatter_in_place(T*, int, int, MPI_Comm).
 *
 * The range on the root process is divided evenly among all processes in the communicator. The root
 * process uses the range size divided by the number of processes in the communicator as the count of
 * elements to send per process, while all other processes use the full range size as the count of
 * elements to receive.
 *
 * @tparam R simplemc::mpi::mpi_range type.
 * @param rg Range to scatter (output on all processes, input on root).
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_range R>
void scatter_in_place(R&& rg, int root, MPI_Comm comm) { // NOLINT (ranges need not be forwarded)
    int count = ranges::size(rg);
    if (comm_rank(comm) == root) {
        count /= comm_size(comm);
    }
    scatter_in_place(ranges::data(rg), count, root, comm);
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_SCATTER_HPP
