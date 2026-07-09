/**
 * @file
 * @brief Wrapper functions for `MPI_Gatherv`.
 */

#ifndef SIMPLEMC_MPI_GATHERV_HPP
#define SIMPLEMC_MPI_GATHERV_HPP

#include <simplemc/mpi/communicator.hpp>
#include <simplemc/mpi/gather.hpp>
#include <simplemc/mpi/mpi_type.hpp>
#include <simplemc/mpi/utils.hpp>
#include <simplemc/utils/ranges.hpp>

#include <mpi.h>

#include <algorithm>
#include <vector>

namespace simplemc::mpi {

namespace detail {

// Prepare recvcounts and displs arrays for gatherv(_in_place) from local count.
inline auto prepare_gatherv_counts_displs(int local_count, int root, MPI_Comm comm) {
    const int size = comm_size(comm);

    // gather all input sizes
    std::vector<int> recvcounts(size);
    gather(&local_count, recvcounts.data(), 1, root, comm);

    // compute displacements (only meaningful on root)
    std::vector<int> displs(size);
    if (comm_rank(comm) == root) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }

    return std::make_pair(std::move(recvcounts), std::move(displs));
}

} // namespace detail

/**
 * @addtogroup simplemc-mpi-coll-gather
 * @{
 */

/**
 * @brief Gatherv data on the root process (low-level).
 *
 * @details Thin wrapper around `MPI_Gatherv`. See the official MPI documentation for more details,
 * e.g. <a href="https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Gatherv.3.html">Open MPI
 * manual</a>.
 *
 * @param sendbuf Pointer to the send buffer.
 * @param sendcount Number of elements to send.
 * @param sendtype MPI datatype of the send elements.
 * @param recvbuf Pointer to the receive buffer.
 * @param recvcounts Array of counts to receive from each process.
 * @param displs Array of displacements in the receive buffer for each process.
 * @param recvtype MPI datatype of the receive elements.
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
inline void gatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, const int* recvcounts,
    const int* displs, MPI_Datatype recvtype, int root, MPI_Comm comm) {
    check_mpi_call(
        MPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm), "MPI_Gatherv");
}

/**
 * @brief Gatherv a contiguous array of values on the root process.
 *
 * @details It calls simplemc::mpi::gatherv with the deduced `MPI_Datatype` from the C++ type `T` (see
 * simplemc::mpi::mpi_type).
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param sendbuf Pointer to the send buffer.
 * @param sendcount Number of elements to send.
 * @param recvbuf Pointer to the receive buffer.
 * @param recvcounts Array of counts to receive from each process.
 * @param displs Array of displacements in the receive buffer for each process.
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void gatherv(
    const T* sendbuf, int sendcount, T* recvbuf, const int* recvcounts, const int* displs, int root, MPI_Comm comm) {
    gatherv(sendbuf, sendcount, mpi_type<T>::get(), recvbuf, recvcounts, displs, mpi_type<T>::get(), root, comm);
}

/**
 * @brief Gatherv a contiguous array of values in place on the root process.
 *
 * @details It calls simplemc::mpi::gatherv with the deduced `MPI_Datatype` from the C++ type `T` (see
 * simplemc::mpi::mpi_type). On the root process, `MPI_IN_PLACE` is used as the send buffer.
 *
 * @note On the root process, the elements to be sent must already be stored at their correct final
 * position in the buffer before calling this function.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param buf Pointer to the buffer (send on all processes and receive on root).
 * @param sendcount Number of elements to send (ignored on root).
 * @param recvcounts Array of counts to receive from each process (only used on root).
 * @param displs Array of displacements in the buffer for each process (only used on root).
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void gatherv_in_place(T* buf, int sendcount, const int* recvcounts, const int* displs, int root, MPI_Comm comm) {
    if (comm_rank(comm) == root) {
        gatherv(MPI_IN_PLACE, sendcount, mpi_type<T>::get(), buf, recvcounts, displs, mpi_type<T>::get(), root, comm);
    } else {
        gatherv(buf, sendcount, mpi_type<T>::get(), buf, recvcounts, displs, mpi_type<T>::get(), root, comm);
    }
}

/**
 * @brief Gatherv contiguous ranges on the root process.
 *
 * @details It first gathers all input range sizes using simplemc::mpi::gather, then computes the
 * displacement array, and finally calls the lower-level simplemc::mpi::gatherv(const T*, int, T*,
 * const int*, const int*, int, MPI_Comm).
 *
 * The input range size is used as the local count of elements to send per process.
 *
 * @tparam R1 simplemc::mpi::mpi_range type.
 * @tparam R2 simplemc::mpi::mpi_range type.
 * @param in_rg Input range to gather.
 * @param out_rg Output range to gather into (only valid on root).
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_range R1, mpi_range R2>
void gatherv(R1&& in_rg, R2&& out_rg, int root, MPI_Comm comm) { // NOLINT (ranges need not be forwarded)
    auto [recvcounts, displs] = detail::prepare_gatherv_counts_displs(ranges::size(in_rg), root, comm);
    gatherv(
        ranges::data(in_rg), ranges::size(in_rg), ranges::data(out_rg), recvcounts.data(), displs.data(), root, comm);
}

/**
 * @brief Gatherv contiguous ranges in place on the root process.
 *
 * @details It first gathers all local counts using simplemc::mpi::gather, then computes the
 * displacement array, and finally calls the lower-level simplemc::mpi::gatherv_in_place(T*, int,
 * const int*, const int*, int, MPI_Comm).
 *
 * Each process has to specify explicitly how many elements it contributes.
 *
 * @tparam R simplemc::mpi::mpi_range type.
 * @param rg Range to gather (input on all processes, output on root).
 * @param local_count Number of elements this process contributes.
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_range R>
void gatherv_in_place(R&& rg, int local_count, int root, MPI_Comm comm) { // NOLINT (ranges need not be forwarded)
    auto [recvcounts, displs] = detail::prepare_gatherv_counts_displs(local_count, root, comm);
    gatherv_in_place(ranges::data(rg), local_count, recvcounts.data(), displs.data(), root, comm);
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_GATHERV_HPP
