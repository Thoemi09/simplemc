/**
 * @file
 * @brief All-reduce communications between MPI processes.
 */

#ifndef SIMPLEMC_MPI_ALL_REDUCE_HPP
#define SIMPLEMC_MPI_ALL_REDUCE_HPP

#include <simplemc/mpi/all_equal.hpp>
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
 * @brief All-reduce data across all processes (low-level).
 *
 * @details Thin wrapper around `MPI_Allreduce` that accepts an explicit `MPI_Datatype`, allowing
 * users to reduce custom or user-defined MPI datatypes. The caller is responsible for ensuring that
 * `sendbuf` and `recvbuf` point to valid memory of the correct type and size.
 *
 * It asserts that `count` is the same on all processes and non-negative.
 *
 * @param sendbuf Pointer to the send buffer.
 * @param recvbuf Pointer to the receive buffer.
 * @param count Number of elements to reduce.
 * @param datatype MPI datatype of the elements.
 * @param op MPI reduction operation (e.g., `MPI_SUM`, `MPI_MAX`, `MPI_MIN`).
 * @param comm MPI communicator.
 */
inline void all_reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    assert(all_equal(count, comm));
    assert(count >= 0);
    check_mpi_call(MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm), "MPI_Allreduce");
}

/**
 * @brief All-reduce data in place across all processes (low-level).
 *
 * @details Thin wrapper around `MPI_Allreduce` with `MPI_IN_PLACE` that accepts an explicit
 * `MPI_Datatype`, allowing users to reduce custom or user-defined MPI datatypes. The caller is
 * responsible for ensuring that `buf` points to valid memory of the correct type and size.
 *
 * It asserts that `count` is the same on all processes and non-negative.
 *
 * @param buf Pointer to the buffer (used as both send and receive).
 * @param count Number of elements to reduce.
 * @param datatype MPI datatype of the elements.
 * @param op MPI reduction operation (e.g., `MPI_SUM`, `MPI_MAX`, `MPI_MIN`).
 * @param comm MPI communicator.
 */
inline void all_reduce_in_place(void* buf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    assert(all_equal(count, comm));
    assert(count >= 0);
    check_mpi_call(MPI_Allreduce(MPI_IN_PLACE, buf, count, datatype, op, comm), "MPI_Allreduce");
}

/**
 * @brief All-reduce a contiguous array of values across all processes.
 *
 * @details It simply calls simplemc::mpi::all_reduce with the deduced `MPI_Datatype` from the C++
 * type `T` (see simplemc::mpi::mpi_type).
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param sendbuf Pointer to the send buffer.
 * @param recvbuf Pointer to the receive buffer.
 * @param count Number of elements to reduce.
 * @param op MPI reduction operation (e.g., `MPI_SUM`, `MPI_MAX`, `MPI_MIN`).
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void all_reduce(const T* sendbuf, T* recvbuf, int count, MPI_Op op, MPI_Comm comm) {
    all_reduce(static_cast<const void*>(sendbuf), static_cast<void*>(recvbuf), count, mpi_type<T>::get(), op, comm);
}

/**
 * @brief All-reduce a contiguous array of values in place across all processes.
 *
 * @details It simply calls simplemc::mpi::all_reduce_in_place with the deduced `MPI_Datatype` from
 * the C++ type `T` (see simplemc::mpi::mpi_type).
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param buf Pointer to the buffer (used as both input and output).
 * @param count Number of elements to reduce.
 * @param op MPI reduction operation (e.g., `MPI_SUM`, `MPI_MAX`, `MPI_MIN`).
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void all_reduce_in_place(T* buf, int count, MPI_Op op, MPI_Comm comm) {
    all_reduce_in_place(static_cast<void*>(buf), count, mpi_type<T>::get(), op, comm);
}

/**
 * @brief All-reduce a single value across all processes.
 *
 * @details It reduces exactly one element by calling simplemc::mpi::all_reduce(const T*, T*, int,
 * MPI_Op, MPI_Comm) with a count of 1.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param in_value Input value to reduce.
 * @param out_value Output value (result of reduction).
 * @param op MPI reduction operation (e.g., `MPI_SUM`, `MPI_MAX`, `MPI_MIN`).
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void all_reduce(const T& in_value, T& out_value, MPI_Op op, MPI_Comm comm) {
    all_reduce(&in_value, &out_value, 1, op, comm);
}

/**
 * @brief All-reduce a single value in place across all processes.
 *
 * @details It reduces exactly one element by calling simplemc::mpi::all_reduce_in_place(T*, int,
 * MPI_Op, MPI_Comm) with a count of 1.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param in_out_value Value to reduce (input and output).
 * @param op MPI reduction operation (e.g., `MPI_SUM`, `MPI_MAX`, `MPI_MIN`).
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void all_reduce_in_place(T& in_out_value, MPI_Op op, MPI_Comm comm) {
    all_reduce_in_place(&in_out_value, 1, op, comm);
}

/**
 * @brief All-reduce a contiguous range across all processes.
 *
 * @details It reduces all elements in the input range and stores the result in the output range by
 * calling simplemc::mpi::all_reduce(const T*, T*, int, MPI_Op, MPI_Comm).
 *
 * It asserts that the output range size is at least the input range size.
 *
 * @tparam R1 simplemc::mpi::mpi_range type.
 * @tparam R2 simplemc::mpi::mpi_range type.
 * @param in_values Input range to reduce.
 * @param out_values Output range (result of reduction).
 * @param op MPI reduction operation (e.g., `MPI_SUM`, `MPI_MAX`, `MPI_MIN`).
 * @param comm MPI communicator.
 */
template <mpi_range R1, mpi_range R2>
void all_reduce(R1&& in_values, R2&& out_values, MPI_Op op, MPI_Comm comm) { // NOLINT (forwarding ref as in/out)
    assert(ranges::size(in_values) <= ranges::size(out_values));
    all_reduce(ranges::data(in_values), ranges::data(out_values), static_cast<int>(ranges::size(in_values)), op, comm);
}

/**
 * @brief All-reduce a contiguous range in place across all processes.
 *
 * @details It reduces all elements in the input range in place by calling
 * simplemc::mpi::all_reduce_in_place(T*, int, MPI_Op, MPI_Comm).
 *
 * @tparam R simplemc::mpi::mpi_range type.
 * @param in_out_values Range to reduce (input and output).
 * @param op MPI reduction operation (e.g., `MPI_SUM`, `MPI_MAX`, `MPI_MIN`).
 * @param comm MPI communicator.
 */
template <mpi_range R>
void all_reduce_in_place(R&& in_out_values, MPI_Op op, MPI_Comm comm) { // NOLINT (forwarding ref as in/out)
    all_reduce_in_place(ranges::data(in_out_values), static_cast<int>(ranges::size(in_out_values)), op, comm);
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_ALL_REDUCE_HPP
