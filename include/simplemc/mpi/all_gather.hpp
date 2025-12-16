/**
 * @file
 * @brief All-gather communications between MPI processes.
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
 * @addtogroup simplemc-mpi-coll
 * @{
 */

/**
 * @brief All-gather data across all processes (low-level).
 *
 * @details Thin wrapper around `MPI_Allgather` that accepts an explicit `MPI_Datatype`, allowing
 * users to gather custom or user-defined MPI datatypes. The caller is responsible for ensuring that
 * `sendbuf` and `recvbuf` point to valid memory of the correct type and size.
 *
 * It asserts that `sendcount` is the same on all processes and non-negative.
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
    assert(sendcount >= 0);
    check_mpi_call(MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm), "MPI_Allgather");
}

/**
 * @brief All-gather data in place across all processes (low-level).
 *
 * @details Thin wrapper around `MPI_Allgather` with `MPI_IN_PLACE` that accepts an explicit
 * `MPI_Datatype`, allowing users to gather custom or user-defined MPI datatypes. The caller is
 * responsible for ensuring that `buf` points to valid memory of the correct type and size.
 *
 * It asserts that `recvcount` is the same on all processes and non-negative.
 *
 * @param buf Pointer to the buffer (used as both send and receive).
 * @param recvcount Number of elements to receive from each process.
 * @param datatype MPI datatype of the elements.
 * @param comm MPI communicator.
 */
inline void all_gather_in_place(void* buf, int recvcount, MPI_Datatype datatype, MPI_Comm comm) {
    assert(all_equal(recvcount, comm));
    assert(recvcount >= 0);
    check_mpi_call(MPI_Allgather(MPI_IN_PLACE, 0, datatype, buf, recvcount, datatype, comm), "MPI_Allgather");
}

/**
 * @brief All-gather a contiguous array of values across all processes.
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
    all_gather(static_cast<const void*>(sendbuf), count, mpi_type<T>::get(), static_cast<void*>(recvbuf), count,
        mpi_type<T>::get(), comm);
}

/**
 * @brief All-gather a contiguous array of values in place across all processes.
 *
 * @details It simply calls simplemc::mpi::all_gather_in_place with the deduced `MPI_Datatype` from
 * the C++ type `T` (see simplemc::mpi::mpi_type).
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param buf Pointer to the buffer (used as both input and output).
 * @param count Number of elements to receive from each process.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void all_gather_in_place(T* buf, int count, MPI_Comm comm) {
    all_gather_in_place(static_cast<void*>(buf), count, mpi_type<T>::get(), comm);
}

/**
 * @brief All-gather a single value across all processes.
 *
 * @details It gathers exactly one element by calling simplemc::mpi::all_gather(const T*, T*, int,
 * MPI_Comm) with a count of 1.
 *
 * It asserts that the output range size equals the communicator size.
 *
 * @tparam R simplemc::mpi::mpi_range type.
 * @param in_value Input value to gather.
 * @param out_values Output range (result of gather).
 * @param comm MPI communicator.
 */
template <mpi_range R>
void all_gather(const ranges::range_value_t<R>& in_value, R&& out_values, MPI_Comm comm) { // NOLINT (forwarding ref)
    assert(comm_size(comm) == static_cast<int>(ranges::size(out_values)));
    all_gather(&in_value, ranges::data(out_values), 1, comm);
}

/**
 * @brief All-gather a contiguous range across all processes.
 *
 * @details It gathers all elements in the input range and stores the result in the output range by
 * calling simplemc::mpi::all_gather(const T*, T*, int, MPI_Comm).
 *
 * It asserts that input ranges have equal size across all processes and that the output range size
 * equals the input range size times the number of processes.
 *
 * @tparam R1 simplemc::mpi::mpi_range type.
 * @tparam R2 simplemc::mpi::mpi_range type.
 * @param in_values Input range to gather.
 * @param out_values Output range (result of gather).
 * @param comm MPI communicator.
 */
template <mpi_range R1, mpi_range R2>
void all_gather(R1&& in_values, R2&& out_values, MPI_Comm comm) { // NOLINT (forwarding ref as in/out)
    assert(all_equal(ranges::size(in_values), comm));
    assert(comm_size(comm) * ranges::size(in_values) == ranges::size(out_values));
    all_gather(ranges::data(in_values), ranges::data(out_values), static_cast<int>(ranges::size(in_values)), comm);
}

/**
 * @brief All-gather a contiguous range in place across all processes.
 *
 * @details It gathers all elements in the range in place by calling
 * simplemc::mpi::all_gather_in_place(T*, int, MPI_Comm).
 *
 * It asserts that ranges have equal size across all processes and that the range size is a multiple
 * of the number of processes.
 *
 * @tparam R simplemc::mpi::mpi_range type.
 * @param in_out_values Range to gather (input and output).
 * @param comm MPI communicator.
 */
template <mpi_range R>
void all_gather_in_place(R&& in_out_values, MPI_Comm comm) { // NOLINT (forwarding ref as in/out)
    assert(all_equal(ranges::size(in_out_values), comm));
    assert(ranges::size(in_out_values) % comm_size(comm) == 0);
    all_gather_in_place(
        ranges::data(in_out_values), static_cast<int>(ranges::size(in_out_values) / comm_size(comm)), comm);
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_ALL_GATHER_HPP
