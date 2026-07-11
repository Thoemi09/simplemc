// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Wrapper functions for `MPI_Allgather`.
 */

#ifndef SIMPLEMC_MPI_ALL_GATHER_HPP
#define SIMPLEMC_MPI_ALL_GATHER_HPP

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
 * @brief All-gather data on all processes (low-level).
 *
 * @details Thin wrapper around `MPI_Allgather`. See the official MPI documentation for more details,
 * e.g. <a href="https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Allgather.3.html">Open MPI
 * manual</a>.
 *
 * @param sendbuf Pointer to the send buffer.
 * @param sendcount Number of elements to send.
 * @param sendtype MPI datatype of the send elements.
 * @param recvbuf Pointer to the receive buffer.
 * @param recvcount Number of elements to receive from each process.
 * @param recvtype MPI datatype of the receive elements.
 * @param comm MPI communicator.
 */
inline void all_gather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
    MPI_Datatype recvtype, MPI_Comm comm) {
    check_mpi_call(MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm), "MPI_Allgather");
}

/**
 * @brief All-gather a contiguous array of values on all processes.
 *
 * @details It calls simplemc::mpi::all_gather with the deduced `MPI_Datatype` from the C++ type `T`
 * (see simplemc::mpi::mpi_type).
 *
 * The given count is used for both send and receive counts.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param sendbuf Pointer to the send buffer.
 * @param recvbuf Pointer to the receive buffer.
 * @param count Number of elements to send/receive per process.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void all_gather(const T* sendbuf, T* recvbuf, int count, MPI_Comm comm) {
    all_gather(sendbuf, count, mpi_type<T>::get(), recvbuf, count, mpi_type<T>::get(), comm);
}

/**
 * @brief All-gather a contiguous array of values in place on all processes.
 *
 * @details It calls simplemc::mpi::all_gather with the deduced `MPI_Datatype` from the C++ type `T`
 * (see simplemc::mpi::mpi_type). `MPI_IN_PLACE` is used as the send buffer.
 *
 * The given count is used for both send and receive counts.
 *
 * @note The elements to be sent must already be stored at their correct final position in the buffer
 * before calling this function.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param buf Pointer to the buffer (send and receive).
 * @param count Number of elements to send/receive per process.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void all_gather_in_place(T* buf, int count, MPI_Comm comm) {
    all_gather(MPI_IN_PLACE, count, mpi_type<T>::get(), buf, count, mpi_type<T>::get(), comm);
}

/**
 * @brief All-gather a single value on all processes.
 *
 * @details It gathers exactly one element from each process by calling
 * simplemc::mpi::all_gather(const T*, T*, int, MPI_Comm) with a count of 1.
 *
 * @tparam R simplemc::mpi::mpi_range type.
 * @param in_value Input value to gather.
 * @param out_rg Output range to gather into.
 * @param comm MPI communicator.
 */
template <mpi_range R>
void all_gather(
    const ranges::range_value_t<R>& in_value, R&& out_rg, MPI_Comm comm) { // NOLINT (ranges need not be forwarded)
    all_gather(&in_value, ranges::data(out_rg), 1, comm);
}

/**
 * @brief All-gather a contiguous range on all processes.
 *
 * @details It gathers all elements in the input range and stores the result in the output range by
 * calling simplemc::mpi::all_gather(const T*, T*, int, MPI_Comm).
 *
 * The input range size is used as the count of elements to send/receive per process.
 *
 * @tparam R1 simplemc::mpi::mpi_range type.
 * @tparam R2 simplemc::mpi::mpi_range type.
 * @param in_rg Input range to gather.
 * @param out_rg Output range to gather into.
 * @param comm MPI communicator.
 */
template <mpi_range R1, mpi_range R2>
void all_gather(R1&& in_rg, R2&& out_rg, MPI_Comm comm) { // NOLINT (ranges need not be forwarded)
    all_gather(ranges::data(in_rg), ranges::data(out_rg), ranges::size(in_rg), comm);
}

/**
 * @brief All-gather a contiguous range in place on all processes.
 *
 * @details It gathers all elements in the range in place such that elements from rank 0 can be found
 * at the beginning of the range, then elements from rank 1, and so on. The function calls
 * simplemc::mpi::all_gather_in_place(T*, int, MPI_Comm).
 *
 * The count is taken as the size of the range divided by the number of processes in the communicator.
 *
 * @tparam R simplemc::mpi::mpi_range type.
 * @param rg Input/Output range to gather (into).
 * @param comm MPI communicator.
 */
template <mpi_range R>
void all_gather_in_place(R&& rg, MPI_Comm comm) { // NOLINT (ranges need not be forwarded)
    all_gather_in_place(ranges::data(rg), ranges::size(rg) / comm_size(comm), comm);
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_ALL_GATHER_HPP
