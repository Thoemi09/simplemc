/**
 * @file
 * @brief Wrapper functions for `MPI_Allgatherv`.
 */

#ifndef SIMPLEMC_MPI_ALL_GATHERV_HPP
#define SIMPLEMC_MPI_ALL_GATHERV_HPP

#include <simplemc/mpi/all_gather.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/mpi/mpi_type.hpp>
#include <simplemc/mpi/utils.hpp>
#include <simplemc/utils/ranges.hpp>

#include <mpi.h>

#include <algorithm>
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
 * @details Thin wrapper around `MPI_Allgatherv`. See the official MPI documentation for more details,
 * e.g. <a href="https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Allgatherv.3.html">Open MPI
 * manual</a>.
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
    check_mpi_call(
        MPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm), "MPI_Allgatherv");
}

/**
 * @brief All-gatherv a contiguous array of values on all processes.
 *
 * @details It calls simplemc::mpi::all_gatherv with the deduced `MPI_Datatype` from the C++ type `T`
 * (see simplemc::mpi::mpi_type).
 *
 * The local receive count is used as the send count.
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
 * @details It calls simplemc::mpi::all_gatherv with the deduced `MPI_Datatype` from the C++ type `T`
 * (see simplemc::mpi::mpi_type). `MPI_IN_PLACE` is used as the send buffer.
 *
 * The local receive count is used as the send count.
 *
 * @note The elements to be sent must already be stored at their correct final position in the buffer
 * before calling this function.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param buf Pointer to the buffer (send and receive).
 * @param recvcounts Array of counts to receive from each process.
 * @param displs Array of displacements in the buffer for each process.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void all_gatherv_in_place(T* buf, const int* recvcounts, const int* displs, MPI_Comm comm) {
    all_gatherv(MPI_IN_PLACE, recvcounts[comm_rank(comm)], mpi_type<T>::get(), buf, recvcounts, displs,
        mpi_type<T>::get(), comm);
}

/**
 * @brief All-gatherv contiguous ranges on all processes.
 *
 * @details It first gathers all input range sizes using simplemc::mpi::all_gather, then computes the
 * displacement array, and finally calls the lower-level simplemc::mpi::all_gatherv(const T*, T*,
 * const int*, const int*, MPI_Comm).
 *
 * The input range size is used as the local count of elements to send per process.
 *
 * @tparam R1 simplemc::mpi::mpi_range type.
 * @tparam R2 simplemc::mpi::mpi_range type.
 * @param in_rg Input range to gather.
 * @param out_rg Output range to gather into.
 * @param comm MPI communicator.
 */
template <mpi_range R1, mpi_range R2>
void all_gatherv(R1&& in_rg, R2&& out_rg, MPI_Comm comm) { // NOLINT (ranges need not be forwarded)
    auto [recvcounts, displs] = detail::prepare_gatherv_counts_displs(ranges::size(in_rg), comm);
    all_gatherv(ranges::data(in_rg), ranges::data(out_rg), recvcounts.data(), displs.data(), comm);
}

/**
 * @brief All-gatherv contiguous ranges in place on all processes.
 *
 * @details It first gathers all local counts using simplemc::mpi::all_gather, then computes the
 * displacement array, and finally calls the lower-level simplemc::mpi::all_gatherv_in_place(T*, const
 * int*, const int*, MPI_Comm).
 *
 * Each process has to specify explicitly how many elements it contributes.
 *
 * @tparam R simplemc::mpi::mpi_range type.
 * @param rg Input/Output range to gather (into).
 * @param local_count Number of elements this process contributes.
 * @param comm MPI communicator.
 */
template <mpi_range R>
void all_gatherv_in_place(R&& rg, int local_count, MPI_Comm comm) { // NOLINT (ranges need not be forwarded)
    auto [recvcounts, displs] = detail::prepare_gatherv_counts_displs(local_count, comm);
    all_gatherv_in_place(ranges::data(rg), recvcounts.data(), displs.data(), comm);
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_ALL_GATHERV_HPP
