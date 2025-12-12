/**
 * @file
 * @brief Scatter communications between MPI processes.
 */

#ifndef SIMPLEMC_MPI_SCATTER_HPP
#define SIMPLEMC_MPI_SCATTER_HPP

#include <simplemc/mpi/broadcast.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/mpi/mpi_type.hpp>
#include <simplemc/mpi/utils.hpp>
#include <simplemc/utils/ranges.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <mpi.h>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-coll
 * @{
 */

/**
 * @brief Scatter a given number of values.
 *
 * @details It calls `MPI_Scatter` if `count > 0`, otherwise it does nothing.
 *
 * If the MPI call fails or if `root` is not a valid process ID, a simplemc::simplemc_exception is
 * thrown.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param comm simplemc::mpi::communicator object.
 * @param in_values Pointer to the memory to be scattered.
 * @param count Number of values to be scattered to each process.
 * @param out_values Pointer to the memory to be scattered into.
 * @param root Root process.
 */
template <mpi_compatible T>
void scatter(const communicator& comm, const T* in_values, int count, T* out_values, int root) {
    if (count <= 0) {
        return;
    }
    if (root < 0 || root >= comm.size()) {
        throw simplemc_exception("Root process is out of bounds", "mpi::scatter");
    }
    check_mpi_call(MPI_Scatter(in_values, count, mpi_type<T>::get(), out_values, count, mpi_type<T>::get(), root, comm),
        "MPI_Scatter");
}

/**
 * @brief Scatter a given number of values in place.
 *
 * @details Same as simplemc::mpi::scatter except that the MPI call is made with `MPI_IN_PLACE`
 * instead of a receive buffer on the root process.
 *
 * @note The root process does not scatter anything to itself.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param comm simplemc::mpi::communicator object.
 * @param in_out_values Pointer to the memory to be scattered/scattered into.
 * @param count Number of values to be scattered to each process.
 * @param root Root process.
 */
template <mpi_compatible T>
void scatter_in_place(const communicator& comm, T* in_out_values, int count, int root) {
    if (count <= 0) {
        return;
    }
    if (root < 0 || root >= comm.size()) {
        throw simplemc_exception("Root process is out of bounds", "mpi::scatter_in_place");
    }
    if (comm.rank() == root) {
        check_mpi_call(
            MPI_Scatter(in_out_values, count, mpi_type<T>::get(), MPI_IN_PLACE, count, mpi_type<T>::get(), root, comm),
            "MPI_Scatter");
    } else {
        check_mpi_call(
            MPI_Scatter(in_out_values, count, mpi_type<T>::get(), in_out_values, count, mpi_type<T>::get(), root, comm),
            "MPI_Scatter");
    }
}

/**
 * @brief Scatter a range (from the root process) into a single value.
 *
 * @details It calls simplemc::mpi::scatter with a count of 1.
 *
 * The size of the input range on the root process has to be equal to the communicator size, otherwise
 * a simplemc::simplemc_exception is thrown.
 *
 * @code{.cpp}
 * // scatter a vector from the root process into single values
 * simplemc::mpi::communicator comm {};
 * std::vector<int> data {};
 * int root {};
 *
 * // set root and data on root...
 *
 * int result {};
 * simplemc::mpi::scatter(comm, data, result, root);
 * @endcode
 *
 * @tparam R Input range.
 * @param comm simplemc::mpi::communicator object.
 * @param in_values Values to be scattered.
 * @param out_value Value to be scattered into.
 * @param root Root process.
 */
template <mpi_range R>
void scatter(const communicator& comm, R&& in_values, ranges::range_value_t<R>& out_value, int root) { // NOLINT
    if (comm.rank() == root) {
        if (comm.size() != static_cast<int>(ranges::size(in_values))) {
            throw simplemc_exception("Input range size is not equal to communicator size", "mpi::scatter");
        }
    }
    scatter(comm, ranges::data(in_values), 1, &out_value, root);
}

/**
 * @brief Scatter a range (from the root process) into a range.
 *
 * @details It calls simplemc::mpi::scatter.
 *
 * The size of the input range on the root process has to be a multiple of the communicator size and
 * the size of the output range on each process has to be equal to that multiple, otherwise a
 * simplemc::simplemc_exception is thrown.
 *
 * @code{.cpp}
 * // scatter a vector from the root process into vectors
 * simplemc::mpi::communicator comm {};
 * std::vector<int> data {};
 * int root {};
 *
 * // set root and data on root...
 *
 * std::vector<int> result(data.size() / comm.size());
 * simplemc::mpi::scatter(comm, data, result, root);
 * @endcode
 *
 * @tparam R1 Input range.
 * @tparam R2 Output range.
 * @param comm simplemc::mpi::communicator object.
 * @param in_values Range to be scattered.
 * @param out_values Range to be scattered into.
 * @param root Root process.
 */
template <mpi_range R1, mpi_range R2>
void scatter(const communicator& comm, R1&& in_values, R2&& out_values, int root) { // NOLINT
    auto in_size = ranges::size(in_values);
    auto chunk_size = in_size / comm.size();
    if (comm.rank() == root) {
        if (in_size != comm.size() * chunk_size) {
            throw simplemc_exception("Input range size is not a mulitple of the communicator size", "mpi::scatter");
        }
    }
    broadcast(comm, chunk_size, root);
    if (chunk_size != ranges::size(out_values)) {
        throw simplemc_exception("Output range size is incorrect", "mpi::scatter");
    }
    scatter(comm, ranges::data(in_values), chunk_size, ranges::data(out_values), root);
}

/**
 * @brief Scatter a range (from the root process) into a range in place.
 *
 * @details It calls simplemc::mpi::scatter_in_place.
 *
 * The size of the range on the root process has to be a multiple of the communicator size and on each
 * other process it has to be equal to that multiple, otherwise a simplemc::simplemc_exception is
 * thrown.
 *
 * @note The root process does not scatter anything to itself.
 *
 * @code{.cpp}
 * // scatter a vector from the root process into vectors in place
 * simplemc::mpi::communicator comm {};
 * std::vector<int> data {};
 * int root {};
 *
 * // set root and data on root...
 *
 * simplemc::mpi::scatter(comm, data, root);
 * @endcode
 *
 * @tparam R Input/Output range.
 * @param comm simplemc::mpi::communicator object.
 * @param in_out_values Range to be scattered/scattered into.
 * @param root Root process.
 */
template <mpi_range R>
void scatter_in_place(const communicator& comm, R&& in_out_values, int root) { // NOLINT
    auto size = ranges::size(in_out_values);
    auto chunk_size = size / comm.size();
    if (comm.rank() == root) {
        if (size != comm.size() * chunk_size) {
            throw simplemc_exception("Range size is not a mulitple of the communicator size", "mpi::scatter_in_place");
        }
    }
    broadcast(comm, chunk_size, root);
    if (comm.rank() != root) {
        if (chunk_size != size) {
            throw simplemc_exception("Range size is incorrect", "mpi::scatter_in_place");
        }
    }
    scatter_in_place(comm, ranges::data(in_out_values), chunk_size, root);
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_SCATTER_HPP
