/**
 * @file
 * @brief Wrapper functions for `MPI_Gather`.
 */

#ifndef SIMPLEMC_MPI_GATHER_HPP
#define SIMPLEMC_MPI_GATHER_HPP

#include <simplemc/mpi/all_equal.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/mpi/mpi_type.hpp>
#include <simplemc/mpi/utils.hpp>
#include <simplemc/utils/ranges.hpp>

#include <mpi.h>

#include <cassert>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-coll-gather
 * @{
 */

/**
 * @brief Gather data on the root process (low-level).
 *
 * @details Thin wrapper around `MPI_Gather` that accepts an explicit `MPI_Datatype`, allowing users
 * to gather custom or user-defined MPI datatypes. The caller is responsible for ensuring that send
 * and receive buffers point to valid memory of the correct type and size.
 *
 * It asserts that the send and receive counts are the same arcross all processes and non-negative,
 * and that the root rank is a valid rank.
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
    assert(all_equal(sendcount, comm));
    assert(all_equal(recvcount, comm));
    assert(sendcount >= 0);
    assert(recvcount >= 0);
    assert(root >= 0 && root < comm_size(comm));
    check_mpi_call(MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm), "MPI_Gather");
}

/**
 * @brief Gather data in place on the root process (low-level).
 *
 * @details Thin wrapper around `MPI_Gather` with `MPI_IN_PLACE` that accepts an explicit
 * `MPI_Datatype`, allowing users to gather custom or user-defined MPI datatypes. The caller is
 * responsible for ensuring that the buffer points to valid memory of the correct type and size.
 *
 * It calls simplemc::mpi::gather with the given arguments on all processes, except that on the root
 * process it uses `MPI_IN_PLACE` for the send buffer.
 *
 * @note On root, the elements to be sent must already be stored at their correct final position in
 * the buffer before calling this function.
 *
 * @param buf Pointer to the buffer (send on all processes and receive on root).
 * @param count Number of elements to send/receive per process.
 * @param datatype MPI datatype of the elements.
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
inline void gather_in_place(void* buf, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
    if (comm_rank(comm) == root) {
        gather(MPI_IN_PLACE, count, datatype, buf, count, datatype, root, comm);
    } else {
        gather(buf, count, datatype, buf, count, datatype, root, comm);
    }
}

/**
 * @brief Gather a contiguous array of values on the root process.
 *
 * @details It simply calls simplemc::mpi::gather with the deduced `MPI_Datatype` from the C++ type
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
void gather(const T* sendbuf, T* recvbuf, int count, int root, MPI_Comm comm) {
    gather(sendbuf, count, mpi_type<T>::get(), recvbuf, count, mpi_type<T>::get(), root, comm);
}

/**
 * @brief Gather a contiguous array of values in place on the root process.
 *
 * @details It simply calls simplemc::mpi::gather_in_place with the deduced `MPI_Datatype` from the
 * C++ type `T` (see simplemc::mpi::mpi_type).
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param buf Pointer to the buffer (send on all processes and receive on root).
 * @param count Number of elements to send/receive per process.
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void gather_in_place(T* buf, int count, int root, MPI_Comm comm) {
    gather_in_place(buf, count, mpi_type<T>::get(), root, comm);
}

/**
 * @brief Gather a single value on the root process.
 *
 * @details It gathers exactly one element per process by calling simplemc::mpi::gather(const T*, T*,
 * int, int, MPI_Comm) with a count of 1.
 *
 * It asserts that the output range size is at least the number of processes in the communicator on
 * the root process.
 *
 * @tparam R simplemc::mpi::mpi_range type.
 * @param in_value Input value to gather.
 * @param out_rg Output range to gather into (only valid on root).
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_range R>
void gather(const ranges::range_value_t<R>& in_value, R&& out_rg, int root, MPI_Comm comm) { // NOLINT
    assert((comm_rank(comm) != root) || (comm_size(comm) <= static_cast<int>(ranges::size(out_rg))));
    gather(&in_value, ranges::data(out_rg), 1, root, comm);
}

/**
 * @brief Gather a contiguous range to the root process.
 *
 * @details It gathers all elements in the input range and stores the result in the output range by
 * calling simplemc::mpi::gather(const T*, T*, int, int, MPI_Comm).
 *
 * It asserts that input ranges have equal size across all processes and that the output range size
 * is at least the input range size times the number of processes on the root process.
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
    assert(all_equal(ranges::size(in_rg), comm));
    assert((comm_rank(comm) != root) || (comm_size(comm) * ranges::size(in_rg) <= ranges::size(out_rg)));
    gather(ranges::data(in_rg), ranges::data(out_rg), ranges::size(in_rg), root, comm);
}

/**
 * @brief Gather a contiguous range in place on the root process.
 *
 * @details It gathers all elements in the range in place on the root process such that elements from
 * rank 0 can be found at the beginning of the range, then elements from rank 1, and so on. The
 * function calls simplemc::mpi::gather_in_place(T*, int, int, MPI_Comm) where the count is taken as
 * the size of the range (on root, it is divided by the number of processes in the communicator).
 *
 * It asserts that ranges have equal size across all processes, except on the root proccess where it
 * must be equal to the range size of the other ranks times the number of processes.
 *
 * @note On root, the elements to be sent must already be stored at their correct final position in
 * the range before calling this function.
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
    assert(all_equal(count, comm));
    gather_in_place(ranges::data(rg), count, root, comm);
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_GATHER_HPP
