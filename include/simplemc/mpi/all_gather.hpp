/**
 * @file
 * @brief Wrapper functions for `MPI_Allgather`.
 */

#ifndef SIMPLEMC_MPI_ALL_GATHER_HPP
#define SIMPLEMC_MPI_ALL_GATHER_HPP

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
 * @brief All-gather data on all processes (low-level).
 *
 * @details Thin wrapper around `MPI_Allgather` that accepts an explicit `MPI_Datatype`, allowing
 * users to gather custom or user-defined MPI datatypes. The caller is responsible for ensuring that
 * the send and receive buffers point to valid memory of the correct type and size.
 *
 * It asserts that the send and receive counts are the same on all processes and non-negative.
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
    assert(all_equal(sendcount, comm));
    assert(all_equal(recvcount, comm));
    assert(sendcount >= 0);
    assert(recvcount >= 0);
    check_mpi_call(MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm), "MPI_Allgather");
}

/**
 * @brief All-gather data in place on all processes (low-level).
 *
 * @details Thin wrapper around `MPI_Allgather` with `MPI_IN_PLACE` that accepts an explicit
 * `MPI_Datatype`, allowing users to gather custom or user-defined MPI datatypes. The caller is
 * responsible for ensuring that the buffer points to valid memory of the correct type and size.
 *
 * It calls simplemc::mpi::all_gather with the given arguments, except that it uses `MPI_IN_PLACE` for
 * the send buffer.
 *
 * @note The elements to be sent must already be stored at their correct final position in the buffer
 * before calling this function.
 *
 * @param buf Pointer to the buffer (send and receive).
 * @param count Number of elements to send/receive per process.
 * @param datatype MPI datatype of the elements.
 * @param comm MPI communicator.
 */
inline void all_gather_in_place(void* buf, int count, MPI_Datatype datatype, MPI_Comm comm) {
    all_gather(MPI_IN_PLACE, count, datatype, buf, count, datatype, comm);
}

/**
 * @brief All-gather a contiguous array of values on all processes.
 *
 * @details It simply calls simplemc::mpi::all_gather with the deduced `MPI_Datatype` from the C++
 * type `T` (see simplemc::mpi::mpi_type) for both send and receive buffers.
 *
 * @note Since the MPI datatype for both send and receive buffers is the same, the number of elements
 * sent and received per process must also be the same.
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
 * @details It simply calls simplemc::mpi::all_gather_in_place with the deduced `MPI_Datatype` from
 * the C++ type `T` (see simplemc::mpi::mpi_type).
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param buf Pointer to the buffer (send and receive).
 * @param count Number of elements to send/receive per process.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void all_gather_in_place(T* buf, int count, MPI_Comm comm) {
    all_gather_in_place(buf, count, mpi_type<T>::get(), comm);
}

/**
 * @brief All-gather a single value on all processes.
 *
 * @details It gathers exactly one element from each process by calling
 * simplemc::mpi::all_gather(const T*, T*, int, MPI_Comm) with a count of 1.
 *
 * It asserts that the output range size is at least the number of processes in the communicator.
 *
 * @tparam R simplemc::mpi::mpi_range type.
 * @param in_value Input value to gather.
 * @param out_rg Output range to gather into.
 * @param comm MPI communicator.
 */
template <mpi_range R>
void all_gather(
    const ranges::range_value_t<R>& in_value, R&& out_rg, MPI_Comm comm) { // NOLINT (ranges need not be forwarded)
    assert(comm_size(comm) <= static_cast<int>(ranges::size(out_rg)));
    all_gather(&in_value, ranges::data(out_rg), 1, comm);
}

/**
 * @brief All-gather a contiguous range on all processes.
 *
 * @details It gathers all elements in the input range and stores the result in the output range by
 * calling simplemc::mpi::all_gather(const T*, T*, int, MPI_Comm).
 *
 * It asserts that input ranges have equal size on all processes and that the output range size is at
 * least the input range size times the number of processes.
 *
 * @tparam R1 simplemc::mpi::mpi_range type.
 * @tparam R2 simplemc::mpi::mpi_range type.
 * @param in_rg Input range to gather.
 * @param out_rg Output range to gather into.
 * @param comm MPI communicator.
 */
template <mpi_range R1, mpi_range R2>
void all_gather(R1&& in_rg, R2&& out_rg, MPI_Comm comm) { // NOLINT (ranges need not be forwarded)
    assert(all_equal(ranges::size(in_rg), comm));
    assert(comm_size(comm) * ranges::size(in_rg) <= ranges::size(out_rg));
    all_gather(ranges::data(in_rg), ranges::data(out_rg), ranges::size(in_rg), comm);
}

/**
 * @brief All-gather a contiguous range in place on all processes.
 *
 * @details It gathers all elements in the range in place such that elements from rank 0 can be found
 * at the beginning of the range, then elements from rank 1, and so on. The function calls
 * simplemc::mpi::all_gather_in_place(T*, int, MPI_Comm) where the count is taken as the size of the
 * range divided by the number of processes in the communicator.
 *
 * It asserts that ranges have equal size on all processes and that the range size is a multiple of
 * the number of processes.
 *
 * @note The elements to be sent must already be stored at their correct final position in the range
 * before calling this function.
 *
 * @tparam R simplemc::mpi::mpi_range type.
 * @param rg Input/Output range to gather (into).
 * @param comm MPI communicator.
 */
template <mpi_range R>
void all_gather_in_place(R&& rg, MPI_Comm comm) { // NOLINT (ranges need not be forwarded)
    assert(all_equal(ranges::size(rg), comm));
    assert(ranges::size(rg) % comm_size(comm) == 0);
    all_gather_in_place(ranges::data(rg), ranges::size(rg) / comm_size(comm), comm);
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_ALL_GATHER_HPP
