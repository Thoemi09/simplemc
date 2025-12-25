/**
 * @file
 * @brief Wrapper functions for `MPI_Allgatherv`.
 */

#ifndef SIMPLEMC_MPI_ALL_GATHERV_HPP
#define SIMPLEMC_MPI_ALL_GATHERV_HPP

#include <simplemc/mpi/all_equal.hpp>
#include <simplemc/mpi/all_gather.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/mpi/mpi_type.hpp>
#include <simplemc/mpi/utils.hpp>
#include <simplemc/utils/ranges.hpp>

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <span>
#include <vector>

namespace simplemc::mpi {

namespace detail {

// Prepare recvcounts and displs arrays for all_gatherv(_in_place) from local count.
inline auto prepare_gatherv_counts_displs(int local_count, MPI_Comm comm) {
    const int size = comm_size(comm);

    // gather all input sizes
    std::vector<int> recvcounts(size);
    all_gather(local_count, recvcounts, comm);

    // compute displacements
    std::vector<int> displs(size);
    displs[0] = 0;
    for (int i = 1; i < size; ++i) {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    return std::make_pair(std::move(recvcounts), std::move(displs));
}

} // namespace detail

/**
 * @addtogroup simplemc-mpi-coll-gather
 * @{
 */

/**
 * @brief All-gatherv data on all processes (low-level).
 *
 * @details Thin wrapper around `MPI_Allgatherv` that accepts an explicit `MPI_Datatype`, allowing
 * users to gather custom or user-defined MPI datatypes with varying counts per process. The caller
 * is responsible for ensuring that send and receive bufferes point to valid memory of the correct
 * type and size.
 *
 * It asserts that the send count is non-negative and equal to the receive count for the calling
 * process, and that the receive counts and displacement arrays are equal across all processes.
 *
 * @param sendbuf Pointer to the send buffer.
 * @param sendcount Number of elements to send.
 * @param sendtype MPI datatype of the send elements.
 * @param recvbuf Pointer to the receive buffer.
 * @param recvcounts Array of counts to receive from each process.
 * @param displs Array of displacements in the receive buffer for each process.
 * @param recvtype MPI datatype of the receive elements.
 * @param comm MPI communicator.
 */
inline void all_gatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, const int* recvcounts,
    const int* displs, MPI_Datatype recvtype, MPI_Comm comm) {
    assert(all_equal(std::span(recvcounts, comm_size(comm)), comm));
    assert(all_equal(std::span(displs, comm_size(comm)), comm));
    assert(sendcount >= 0);
    assert(sendcount == recvcounts[comm_rank(comm)]);
    check_mpi_call(
        MPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm), "MPI_Allgatherv");
}

/**
 * @brief All-gatherv data in place on all processes (low-level).
 *
 * @details Thin wrapper around `MPI_Allgatherv` with `MPI_IN_PLACE` that accepts an explicit
 * `MPI_Datatype`, allowing users to gather custom or user-defined MPI datatypes with varying counts
 * per process. The caller is responsible for ensuring that the buffer points to valid memory of the
 * correct type and size.
 *
 * It calls simplemc::mpi::all_gatherv with the given arguments, except that it uses `MPI_IN_PLACE`
 * for the send buffer and the local receive count as the send count.
 *
 * @note The elements to be sent must already be stored at their correct final position in the buffer
 * before calling this function.
 *
 * @param buf Pointer to the buffer (send and receive).
 * @param recvcounts Array of counts to receive from each process.
 * @param displs Array of displacements in the buffer for each process.
 * @param datatype MPI datatype of the elements.
 * @param comm MPI communicator.
 */
inline void all_gatherv_in_place(
    void* buf, const int* recvcounts, const int* displs, MPI_Datatype datatype, MPI_Comm comm) {
    all_gatherv(MPI_IN_PLACE, recvcounts[comm_rank(comm)], datatype, buf, recvcounts, displs, datatype, comm);
}

/**
 * @brief All-gatherv a contiguous array of values on all processes.
 *
 * @details It simply calls simplemc::mpi::all_gatherv with the deduced `MPI_Datatype` from the C++
 * type `T` (see simplemc::mpi::mpi_type) for both send and receive buffers. The number of elements
 * to send is determined by the corresponding entry in the receive counts array.
 *
 * @note Since the MPI datatype for both send and receive buffers is the same, the send count is
 * automatically derived from the local receive count.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param sendbuf Pointer to the send buffer.
 * @param recvbuf Pointer to the receive buffer.
 * @param recvcounts Array of counts to receive from each process.
 * @param displs Array of displacements in the receive buffer for each process.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void all_gatherv(const T* sendbuf, T* recvbuf, const int* recvcounts, const int* displs, MPI_Comm comm) {
    all_gatherv(sendbuf, recvcounts[comm_rank(comm)], mpi_type<T>::get(), recvbuf, recvcounts, displs,
        mpi_type<T>::get(), comm);
}

/**
 * @brief All-gatherv a contiguous array of values in place on all processes.
 *
 * @details It simply calls simplemc::mpi::all_gatherv_in_place with the deduced `MPI_Datatype` from
 * the C++ type `T` (see simplemc::mpi::mpi_type).
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param buf Pointer to the buffer (send and receive).
 * @param recvcounts Array of counts to receive from each process.
 * @param displs Array of displacements in the buffer for each process.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void all_gatherv_in_place(T* buf, const int* recvcounts, const int* displs, MPI_Comm comm) {
    all_gatherv_in_place(buf, recvcounts, displs, mpi_type<T>::get(), comm);
}

/**
 * @brief All-gatherv contiguous ranges on all processes.
 *
 * @details It automatically gathers the input range sizes from all processes, computes displacements,
 * and performs the all-gatherv operation. Each process can contribute a different number of elements.
 *
 * The function first gathers all input sizes using simplemc::mpi::all_gather, then computes the
 * displacement array, and finally calls the lower-level simplemc::mpi::all_gatherv(const T*, T*,
 * const int*, const int*, MPI_Comm).
 *
 * It asserts that the output range size is at least the total number of elements gathered from all
 * processes.
 *
 * @tparam R1 simplemc::mpi::mpi_range type.
 * @tparam R2 simplemc::mpi::mpi_range type.
 * @param in_rg Input range to gather.
 * @param out_rg Output range to gather into.
 * @param comm MPI communicator.
 */
template <mpi_range R1, mpi_range R2>
void all_gatherv(R1&& in_rg, R2&& out_rg, MPI_Comm comm) { // NOLINT (ranges need not be forwarded)
    // gather all input sizes and compute displacements
    auto [recvcounts, displs] = detail::prepare_gatherv_counts_displs(ranges::size(in_rg), comm);

    // compute and check total expected size
    const int size = comm_size(comm);
    const int total_count = displs[size - 1] + recvcounts[size - 1];
    assert(total_count <= static_cast<int>(ranges::size(out_rg)));

    all_gatherv(ranges::data(in_rg), ranges::data(out_rg), recvcounts.data(), displs.data(), comm);
}

/**
 * @brief All-gatherv contiguous ranges in place on all processes.
 *
 * @details It automatically gathers the local count from all processes, computes displacements, and
 * performs the in place all-gatherv operation. Each process has to specify explicitly how many
 * elements it contributes.
 *
 * The function gathers all local counts using simplemc::mpi::all_gather, computes the displacement
 * array, and calls the lower-level simplemc::mpi::all_gatherv_in_place(T*, const int*, const int*,
 * MPI_Comm).
 *
 * It asserts that the range size is at least the total number of elements gathered from all
 * processes.
 *
 * @note The elements to be sent must already be stored at their correct final position in the range
 * before calling this function.
 *
 * @tparam R simplemc::mpi::mpi_range type.
 * @param rg Input/Output range to gather (into).
 * @param local_count Number of elements this process contributes.
 * @param comm MPI communicator.
 */
template <mpi_range R>
void all_gatherv_in_place(R&& rg, int local_count, MPI_Comm comm) { // NOLINT (ranges need not be forwarded)
    // gather all input sizes and compute displacements
    auto [recvcounts, displs] = detail::prepare_gatherv_counts_displs(local_count, comm);

    // compute total expected size
    const int size = comm_size(comm);
    const int total_count = displs[size - 1] + recvcounts[size - 1];
    assert(total_count <= static_cast<int>(ranges::size(rg)));

    all_gatherv_in_place(ranges::data(rg), recvcounts.data(), displs.data(), comm);
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_ALL_GATHERV_HPP
