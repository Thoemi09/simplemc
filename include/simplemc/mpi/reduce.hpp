// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Wrapper functions for `MPI_Reduce`.
 */

#ifndef SIMPLEMC_MPI_REDUCE_HPP
#define SIMPLEMC_MPI_REDUCE_HPP

#include <simplemc/mpi/communicator.hpp>
#include <simplemc/mpi/mpi_type.hpp>
#include <simplemc/mpi/utils.hpp>
#include <simplemc/utils/ranges.hpp>

#include <mpi.h>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-coll-reduce
 * @{
 */

/**
 * @brief Reduce data on the root process (low-level).
 *
 * @details Thin wrapper around `MPI_Reduce`. See the official MPI documentation for more details,
 * e.g. <a href="https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Reduce.3.html">Open MPI manual
 * </a>.
 *
 * @param sendbuf Pointer to the send buffer.
 * @param recvbuf Pointer to the receive buffer.
 * @param count Number of elements to reduce.
 * @param datatype MPI datatype of the elements.
 * @param op MPI reduction operation (e.g., `MPI_SUM`, `MPI_MAX`, `MPI_MIN`).
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
inline void reduce(
    const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
    check_mpi_call(MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm), "MPI_Reduce");
}

/**
 * @brief Reduce a contiguous array of values on the root process.
 *
 * @details It calls simplemc::mpi::reduce with the deduced `MPI_Datatype` from the C++ type `T` (see
 * simplemc::mpi::mpi_type).
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param sendbuf Pointer to the send buffer.
 * @param recvbuf Pointer to the receive buffer.
 * @param count Number of elements to reduce.
 * @param op MPI reduction operation (e.g., `MPI_SUM`, `MPI_MAX`, `MPI_MIN`).
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void reduce(const T* sendbuf, T* recvbuf, int count, MPI_Op op, int root, MPI_Comm comm) {
    reduce(sendbuf, recvbuf, count, mpi_type<T>::get(), op, root, comm);
}

/**
 * @brief Reduce a contiguous array of values in place on the root process.
 *
 * @details It calls simplemc::mpi::reduce with the deduced `MPI_Datatype` from the C++ type `T` (see
 * simplemc::mpi::mpi_type). On the root process, `MPI_IN_PLACE` is used as the send buffer.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param buf Pointer to the buffer (send on all processes and receive on root).
 * @param count Number of elements to reduce.
 * @param op MPI reduction operation (e.g., `MPI_SUM`, `MPI_MAX`, `MPI_MIN`).
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void reduce_in_place(T* buf, int count, MPI_Op op, int root, MPI_Comm comm) {
    if (comm_rank(comm) == root) {
        reduce(MPI_IN_PLACE, buf, count, mpi_type<T>::get(), op, root, comm);
    } else {
        reduce(buf, buf, count, mpi_type<T>::get(), op, root, comm);
    }
}

/**
 * @brief Reduce a single value on the root process.
 *
 * @details It reduces exactly one element by calling simplemc::mpi::reduce(const T*, T*, int, MPI_Op,
 * int, MPI_Comm) with a count of 1.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param in_value Input value to reduce.
 * @param out_value Output value to reduce into (only valid on root).
 * @param op MPI reduction operation (e.g., `MPI_SUM`, `MPI_MAX`, `MPI_MIN`).
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void reduce(const T& in_value, T& out_value, MPI_Op op, int root, MPI_Comm comm) {
    reduce(&in_value, &out_value, 1, op, root, comm);
}

/**
 * @brief Reduce a single value in place on the root process.
 *
 * @details It reduces exactly one element by calling simplemc::mpi::reduce_in_place(T*, int, MPI_Op,
 * int, MPI_Comm) with a count of 1.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param value Value to reduce (input on all processes, output on root).
 * @param op MPI reduction operation (e.g., `MPI_SUM`, `MPI_MAX`, `MPI_MIN`).
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void reduce_in_place(T& value, MPI_Op op, int root, MPI_Comm comm) {
    reduce_in_place(&value, 1, op, root, comm);
}

/**
 * @brief Reduce a contiguous range on the root process.
 *
 * @details It reduces all elements in the input range and stores the result in the output range by
 * calling simplemc::mpi::reduce(const T*, T*, int, MPI_Op, int, MPI_Comm).
 *
 * The input range size is used as the count of elements to reduce.
 *
 * @tparam R1 simplemc::mpi::mpi_range type.
 * @tparam R2 simplemc::mpi::mpi_range type.
 * @param in_rg Input range to reduce.
 * @param out_rg Output range to reduce into (only valid on root).
 * @param op MPI reduction operation (e.g., `MPI_SUM`, `MPI_MAX`, `MPI_MIN`).
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_range R1, mpi_range R2>
void reduce(R1&& in_rg, R2&& out_rg, MPI_Op op, int root, MPI_Comm comm) { // NOLINT (forwarding ref as in/out)
    reduce(ranges::data(in_rg), ranges::data(out_rg), ranges::size(in_rg), op, root, comm);
}

/**
 * @brief Reduce a contiguous range in place on the root process.
 *
 * @details It reduces all elements in the range in place by calling simplemc::mpi::reduce_in_place(
 * T*, int, MPI_Op, int, MPI_Comm).
 *
 * The range size is used as the count of elements to reduce.
 *
 * @tparam R simplemc::mpi::mpi_range type.
 * @param rg Range to reduce (input on all processes, output on root).
 * @param op MPI reduction operation (e.g., `MPI_SUM`, `MPI_MAX`, `MPI_MIN`).
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_range R>
void reduce_in_place(R&& rg, MPI_Op op, int root, MPI_Comm comm) { // NOLINT (forwarding ref as in/out)
    reduce_in_place(ranges::data(rg), ranges::size(rg), op, root, comm);
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_REDUCE_HPP
