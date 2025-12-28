/**
 * @file
 * @brief Wrapper functions for `MPI_Scatterv`.
 */

#ifndef SIMPLEMC_MPI_SCATTERV_HPP
#define SIMPLEMC_MPI_SCATTERV_HPP

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

// Prepare sendcounts and displs arrays for scatterv(_in_place) from local count.
inline auto prepare_scatterv_counts_displs(int local_count, int root, MPI_Comm comm) {
    const int size = comm_size(comm);

    // gather all output sizes to the root
    std::vector<int> sendcounts(size);
    gather(&local_count, sendcounts.data(), 1, root, comm);

    // compute displacements (only meaningful on root)
    std::vector<int> displs(size);
    if (comm_rank(comm) == root) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i) {
            displs[i] = displs[i - 1] + sendcounts[i - 1];
        }
    }

    return std::make_pair(std::move(sendcounts), std::move(displs));
}

} // namespace detail

/**
 * @addtogroup simplemc-mpi-coll-scatter
 * @{
 */

/**
 * @brief Scatterv data from the root process (low-level).
 *
 * @details Thin wrapper around `MPI_Scatterv`. See the official MPI documentation for more details,
 * e.g. <a href="https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Scatterv.3.html">Open MPI
 * manual</a>.
 *
 * @param sendbuf Pointer to the send buffer.
 * @param sendcounts Array of counts to send to each process.
 * @param displs Array of displacements in the send buffer for each process.
 * @param sendtype MPI datatype of the send elements.
 * @param recvbuf Pointer to the receive buffer.
 * @param recvcount Number of elements to receive.
 * @param recvtype MPI datatype of the receive elements.
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
inline void scatterv(const void* sendbuf, const int* sendcounts, const int* displs, MPI_Datatype sendtype,
    void* recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) {
    check_mpi_call(
        MPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm), "MPI_Scatterv");
}

/**
 * @brief Scatterv a contiguous array of values from the root process.
 *
 * @details It calls simplemc::mpi::scatterv with the deduced `MPI_Datatype` from the C++ type `T`
 * (see simplemc::mpi::mpi_type).
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param sendbuf Pointer to the send buffer.
 * @param sendcounts Array of counts to send to each process.
 * @param displs Array of displacements in the send buffer for each process.
 * @param recvbuf Pointer to the receive buffer.
 * @param recvcount Number of elements to receive.
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void scatterv(
    const T* sendbuf, const int* sendcounts, const int* displs, T* recvbuf, int recvcount, int root, MPI_Comm comm) {
    scatterv(sendbuf, sendcounts, displs, mpi_type<T>::get(), recvbuf, recvcount, mpi_type<T>::get(), root, comm);
}

/**
 * @brief Scatterv a contiguous array of values in place from the root process.
 *
 * @details It calls simplemc::mpi::scatterv with the deduced `MPI_Datatype` from the C++ type `T`
 * (see simplemc::mpi::mpi_type). On the root process, `MPI_IN_PLACE` is used as the receive buffer.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param buf Pointer to the buffer (send on root and receive on all processes).
 * @param sendcounts Array of counts to send to each process.
 * @param displs Array of displacements in the buffer for each process.
 * @param recvcount Number of elements to receive.
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_compatible T>
void scatterv_in_place(T* buf, const int* sendcounts, const int* displs, int recvcount, int root, MPI_Comm comm) {
    if (comm_rank(comm) == root) {
        scatterv(buf, sendcounts, displs, mpi_type<T>::get(), MPI_IN_PLACE, recvcount, mpi_type<T>::get(), root, comm);
    } else {
        scatterv(buf, sendcounts, displs, mpi_type<T>::get(), buf, recvcount, mpi_type<T>::get(), root, comm);
    }
}

/**
 * @brief Scatterv contiguous ranges from the root process.
 *
 * @details It first gathers all output range sizes using simplemc::mpi::gather, then computes the
 * displacement array, and finally calls the lower-level simplemc::mpi::scatterv(const T*, const int*,
 * const int*, T*, int, int, MPI_Comm).
 *
 * The output range size is used as the local count of elements to receive per process.
 *
 * @tparam R1 simplemc::mpi::mpi_range type.
 * @tparam R2 simplemc::mpi::mpi_range type.
 * @param in_rg Input range to scatter (only valid on root).
 * @param out_rg Output range to scatter into.
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_range R1, mpi_range R2>
void scatterv(R1&& in_rg, R2&& out_rg, int root, MPI_Comm comm) { // NOLINT (ranges need not be forwarded)
    auto [sendcounts, displs] = detail::prepare_scatterv_counts_displs(ranges::size(out_rg), root, comm);
    scatterv(
        ranges::data(in_rg), sendcounts.data(), displs.data(), ranges::data(out_rg), ranges::size(out_rg), root, comm);
}

/**
 * @brief Scatterv contiguous ranges in place from the root process.
 *
 * @details It first gathers all local counts using simplemc::mpi::gather, then computes the
 * displacement array, and finally calls the lower-level simplemc::mpi::scatterv_in_place(T*, const
 * int*, const int*, int, int, MPI_Comm).
 *
 * Each process has to specify explicitly how many elements it receives.
 *
 * @tparam R simplemc::mpi::mpi_range type.
 * @param rg Input/Output range to scatter (into).
 * @param local_count Number of elements this process receives.
 * @param root Root process rank.
 * @param comm MPI communicator.
 */
template <mpi_range R>
void scatterv_in_place(R&& rg, int local_count, int root, MPI_Comm comm) { // NOLINT (ranges need not be forwarded)
    auto [sendcounts, displs] = detail::prepare_scatterv_counts_displs(local_count, root, comm);
    scatterv_in_place(ranges::data(rg), sendcounts.data(), displs.data(), local_count, root, comm);
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_SCATTERV_HPP
