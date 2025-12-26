/**
 * @file
 * @brief Wrapper functions for `MPI_Scatter`.
 */

#ifndef SIMPLEMC_MPI_SCATTER_HPP
#define SIMPLEMC_MPI_SCATTER_HPP

#include <simplemc/mpi/all_equal.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/mpi/mpi_type.hpp>
#include <simplemc/mpi/utils.hpp>
#include <simplemc/utils/ranges.hpp>

#include <mpi.h>

#include <cassert>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-coll-scatter
 * @{
 */

/**
 * @brief Scatter data from the root process (low-level).
 *
 * @details Thin wrapper around `MPI_Scatter` that accepts an explicit `MPI_Datatype`, allowing users
 * to scatter custom or user-defined MPI datatypes. The caller is responsible for ensuring that send
 * and receive buffers point to valid memory of the correct type and size.
 *
 * It asserts that the send and receive counts are the same across all processes and non-negative, and
 * that the root rank is a valid rank.
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
    assert(all_equal(sendcount, comm));
    assert(all_equal(recvcount, comm));
    assert(sendcount >= 0);
    assert(recvcount >= 0);
    assert(root >= 0 && root < comm_size(comm));
    check_mpi_call(MPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm), "MPI_Scatter");
}

/**
 * @brief Scatter data in place from the root process (low-level).
 *
 * @details Thin wrapper around `MPI_Scatter` with `MPI_IN_PLACE` that accepts an explicit
 * `MPI_Datatype`, allowing users to scatter custom or user-defined MPI datatypes. The caller is
 * responsible for ensuring that the buffer points to valid memory of the correct type and size.
 *
 * It calls simplemc::mpi::scatter with the given arguments on all processes, except that on the root
 * process it uses `MPI_IN_PLACE` for the receive buffer.
 *
 * @note The root process does not scatter anything to itself.
 *
 * @param buf Pointer to the buffer (send on root and receive on all processes).
 * @param count Number of elements to send/receive per process.
 * @param datatype MPI datatype of the elements.
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
inline void scatter_in_place(void* buf, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
    if (comm_rank(comm) == root) {
        scatter(buf, count, datatype, MPI_IN_PLACE, count, datatype, root, comm);
    } else {
        scatter(buf, count, datatype, buf, count, datatype, root, comm);
    }
}

/**
 * @brief Scatter a contiguous array of values from the root process.
 *
 * @details It simply calls simplemc::mpi::scatter with the deduced `MPI_Datatype` from the C++ type
 * `T` (see simplemc::mpi::mpi_type) for both send and receive buffers.
 *
 * @note Since the MPI datatype for both send and receive buffers is the same, the number of elements
 * sent and received per process must also be the same.
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
 * @details It simply calls simplemc::mpi::scatter_in_place with the deduced `MPI_Datatype` from the
 * C++ type `T` (see simplemc::mpi::mpi_type).
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param buf Pointer to the buffer (send on root and receive on all processes).
 * @param count Number of elements to send/receive per process.
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void scatter_in_place(T* buf, int count, int root, MPI_Comm comm) {
    scatter_in_place(buf, count, mpi_type<T>::get(), root, comm);
}

/**
 * @brief Scatter a range (from the root process) into a single value.
 *
 * @details It scatters exactly one element to each process by calling simplemc::mpi::scatter(const
 * T*, T*, int, int, MPI_Comm) with a count of 1.
 *
 * It asserts that the input range size on the root process is equal the number of processes in the
 * communicator.
 *
 * @tparam R simplemc::mpi::mpi_range type.
 * @param in_rg Input range to scatter (only on root).
 * @param out_value Output value to scatter into.
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_range R>
void scatter(R&& in_rg, ranges::range_value_t<R>& out_value, int root, MPI_Comm comm) { // NOLINT
    assert((comm_rank(comm) != root) || (comm_size(comm) == static_cast<int>(ranges::size(in_rg))));
    scatter(ranges::data(in_rg), &out_value, 1, root, comm);
}

/**
 * @brief Scatter a range (from the root process) into a range.
 *
 * @details It scatters all elements from the input range on the root process and stores the result in
 * the output range by calling simplemc::mpi::scatter(const T*, T*, int, int, MPI_Comm). The input
 * range on the root process is divided evenly among all processes in the communicator.
 *
 * It asserts that the input range size on the root process is a multiple of the communicator size and
 * that the output range size on each process is equal to that multiple.
 *
 * @tparam R1 simplemc::mpi::mpi_range type.
 * @tparam R2 simplemc::mpi::mpi_range type.
 * @param in_rg Input range to be scattered (only on root).
 * @param out_rg Output range (result of scatter).
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_range R1, mpi_range R2>
void scatter(R1&& in_rg, R2&& out_rg, int root, MPI_Comm comm) { // NOLINT (ranges need not be forwarded)
    auto const count = ranges::size(out_rg);
    assert(all_equal(count, comm));
    assert((comm_rank(comm) != root) || (comm_size(comm) * count == ranges::size(in_rg)));
    scatter(ranges::data(in_rg), ranges::data(out_rg), count, root, comm);
}

/**
 * @brief Scatter a range (from the root process) into a range in place.
 *
 * @details It scatters all elements from the range on the root process in place by calling
 * simplemc::mpi::scatter_in_place(T*, int, int, MPI_Comm). The range on the root process is divided
 * evenly among all processes in the communicator.
 *
 * It asserts that the range size on the root process is a multiple of the communicator size and on
 * each other process it is equal to that multiple.
 *
 * @note The root process does not scatter anything to itself.
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
    assert(all_equal(count, comm));
    scatter_in_place(ranges::data(rg), count, root, comm);
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_SCATTER_HPP
