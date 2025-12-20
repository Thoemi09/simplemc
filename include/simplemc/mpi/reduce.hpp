/**
 * @file
 * @brief Reduce communications between MPI processes.
 */

#ifndef SIMPLEMC_MPI_REDUCE_HPP
#define SIMPLEMC_MPI_REDUCE_HPP

#include <simplemc/mpi/all_equal.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/mpi/mpi_type.hpp>
#include <simplemc/mpi/utils.hpp>
#include <simplemc/utils/ranges.hpp>

#include <mpi.h>

#include <cassert>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-coll-reduce
 * @{
 */

/**
 * @brief Reduce data to the root process (low-level).
 *
 * @details Thin wrapper around `MPI_Reduce` that accepts an explicit `MPI_Datatype`, allowing
 * users to reduce custom or user-defined MPI datatypes. The caller is responsible for ensuring that
 * `sendbuf` and `recvbuf` point to valid memory of the correct type and size.
 *
 * It asserts that `count` is the same on all processes, non-negative, and that `root` is a valid
 * rank.
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
    assert(all_equal(count, comm));
    assert(count >= 0);
    assert(root >= 0 && root < comm_size(comm));
    check_mpi_call(MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm), "MPI_Reduce");
}

/**
 * @brief Reduce data in place to the root process (low-level).
 *
 * @details Thin wrapper around `MPI_Reduce` with `MPI_IN_PLACE` that accepts an explicit
 * `MPI_Datatype`, allowing users to reduce custom or user-defined MPI datatypes. The caller is
 * responsible for ensuring that `buf` points to valid memory of the correct type and size.
 *
 * It calls simplemc::mpi::reduce with the given arguments on all processes, except that on the root
 * process it uses `MPI_IN_PLACE` for the send buffer.
 *
 * @param buf Pointer to the buffer (send on all processes and receive on root).
 * @param count Number of elements to reduce.
 * @param datatype MPI datatype of the elements.
 * @param op MPI reduction operation (e.g., `MPI_SUM`, `MPI_MAX`, `MPI_MIN`).
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
inline void reduce_in_place(void* buf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
    if (comm_rank(comm) == root) {
        reduce(MPI_IN_PLACE, buf, count, datatype, op, root, comm);
    } else {
        reduce(buf, buf, count, datatype, op, root, comm);
    }
}

/**
 * @brief Reduce a contiguous array of values to the root process.
 *
 * @details It simply calls simplemc::mpi::reduce with the deduced `MPI_Datatype` from the C++ type
 * `T` (see simplemc::mpi::mpi_type).
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
    reduce(static_cast<const void*>(sendbuf), static_cast<void*>(recvbuf), count, mpi_type<T>::get(), op, root, comm);
}

/**
 * @brief Reduce a contiguous array of values in place to the root process.
 *
 * @details It simply calls simplemc::mpi::reduce_in_place with the deduced `MPI_Datatype` from the
 * C++ type `T` (see simplemc::mpi::mpi_type).
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
    reduce_in_place(static_cast<void*>(buf), count, mpi_type<T>::get(), op, root, comm);
}

/**
 * @brief Reduce a single value to the root process.
 *
 * @details It reduces exactly one element by calling simplemc::mpi::reduce(const T*, T*, int, MPI_Op,
 * int, MPI_Comm) with a count of 1.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param in_value Input value to reduce.
 * @param out_value Output value (result of reduction, only valid on root).
 * @param op MPI reduction operation (e.g., `MPI_SUM`, `MPI_MAX`, `MPI_MIN`).
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void reduce(const T& in_value, T& out_value, MPI_Op op, int root, MPI_Comm comm) {
    reduce(&in_value, &out_value, 1, op, root, comm);
}

/**
 * @brief Reduce a single value in place to the root process.
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
 * @brief Reduce a contiguous range to the root process.
 *
 * @details It reduces all elements in the input range and stores the result in the output range by
 * calling simplemc::mpi::reduce(const T*, T*, int, MPI_Op, int, MPI_Comm).
 *
 * It asserts that the output range size is at least the input range size on the root process.
 *
 * @tparam R1 simplemc::mpi::mpi_range type.
 * @tparam R2 simplemc::mpi::mpi_range type.
 * @param in_rg Input range to reduce.
 * @param out_rg Output range (result of reduction, only valid on root).
 * @param op MPI reduction operation (e.g., `MPI_SUM`, `MPI_MAX`, `MPI_MIN`).
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_range R1, mpi_range R2>
void reduce(R1&& in_rg, R2&& out_rg, MPI_Op op, int root, MPI_Comm comm) { // NOLINT (forwarding ref as in/out)
    assert((comm_rank(comm) != root) || (ranges::size(in_rg) <= ranges::size(out_rg)));
    reduce(ranges::data(in_rg), ranges::data(out_rg), static_cast<int>(ranges::size(in_rg)), op, root, comm);
}

/**
 * @brief Reduce a contiguous range in place to the root process.
 *
 * @details It reduces all elements in the input range in place by calling
 * simplemc::mpi::reduce_in_place(T*, int, MPI_Op, int, MPI_Comm).
 *
 * @tparam R simplemc::mpi::mpi_range type.
 * @param rg Range to reduce (input on all processes, output on root).
 * @param op MPI reduction operation (e.g., `MPI_SUM`, `MPI_MAX`, `MPI_MIN`).
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_range R>
void reduce_in_place(R&& rg, MPI_Op op, int root, MPI_Comm comm) { // NOLINT (forwarding ref as in/out)
    reduce_in_place(ranges::data(rg), static_cast<int>(ranges::size(rg)), op, root, comm);
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_REDUCE_HPP
