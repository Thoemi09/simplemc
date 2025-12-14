/**
 * @file
 * @brief Gather communications between MPI processes.
 */

#ifndef SIMPLEMC_MPI_GATHER_HPP
#define SIMPLEMC_MPI_GATHER_HPP

#include <simplemc/mpi/all_equal.hpp>
#include <simplemc/mpi/all_reduce.hpp>
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
 * @brief Gather a given number of values (only on root process).
 *
 * @details It calls `MPI_Gather` if `count > 0`, otherwise it does nothing.
 *
 * If the MPI call fails or if `root` is not a valid process ID, a simplemc::simplemc_exception is
 * thrown.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param comm simplemc::mpi::communicator object.
 * @param in_values Pointer to the memory to be gathered.
 * @param count Number of values to be gathered.
 * @param out_values Pointer to the memory to be gathered into.
 * @param root Root process.
 */
template <typename T>
void gather(const communicator& comm, const T* in_values, int count, T* out_values, int root) {
    if (count <= 0) {
        return;
    }
    if (root < 0 || root >= comm.size()) {
        throw simplemc_exception("Root process is out of bounds", "mpi::gather");
    }
    check_mpi_call(MPI_Gather(in_values, count, mpi_type<T>::get(), out_values, count, mpi_type<T>::get(), root, comm),
        "MPI_Gather");
}

/**
 * @brief Gather a given number of values in place (only on root process).
 *
 * @details Same as simplemc::mpi::gather except that the MPI call is made with `MPI_IN_PLACE`
 * instead of a send buffer on the root process.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param comm simplemc::mpi::communicator object.
 * @param in_out_values Pointer to the memory to be gathered/gathered into.
 * @param count Number of values to be gathered.
 * @param root Root process.
 */
template <typename T>
void gather_in_place(const communicator& comm, T* in_out_values, int count, int root) {
    if (count <= 0) {
        return;
    }
    if (root < 0 || root >= comm.size()) {
        throw simplemc_exception("Root process is out of bounds", "mpi::gather_in_place");
    }
    if (comm.rank() == root) {
        check_mpi_call(
            MPI_Gather(MPI_IN_PLACE, count, mpi_type<T>::get(), in_out_values, count, mpi_type<T>::get(), root, comm),
            "MPI_Gather");
    } else {
        check_mpi_call(
            MPI_Gather(in_out_values, count, mpi_type<T>::get(), in_out_values, count, mpi_type<T>::get(), root, comm),
            "MPI_Gather");
    }
}

/**
 * @brief Gather a single value (only on root process).
 *
 * @details It calls simplemc::mpi::gather with a count of 1.
 *
 * The size of the output range on the root process has to be equal to the communicator size,
 * otherwise a simplemc::simplemc_exception is thrown.
 *
 * @code{.cpp}
 * // gather a single value on the root processes
 * simplemc::mpi::communicator comm {};
 * int value {}, root {};
 *
 * // set value and root...
 *
 * std::vector<int> result(comm.size());
 * simplemc::mpi::gather(comm, value, result, root);
 * @endcode
 *
 * @tparam R Output range.
 * @param comm simplemc::mpi::communicator object.
 * @param in_value Value to be gathered.
 * @param out_values Range to be gathered into.
 * @param root Root process.
 */
template <mpi_range R>
void gather(const communicator& comm, const ranges::range_value_t<R>& in_value, R&& out_values, int root) { // NOLINT
    if (comm.rank() == root) {
        if (comm.size() != static_cast<int>(ranges::size(out_values))) {
            throw simplemc_exception("Output range size is not equal to communicator size", "mpi::gather");
        }
    }
    gather(comm, &in_value, 1, ranges::data(out_values), root);
}

/**
 * @brief Gather a range (only on root process).
 *
 * @details It calls simplemc::mpi::gather.
 *
 * The input ranges must have the same size across all processes and the size of the output range on
 * the root process must be equal to the size of the input range times the number of processes,
 * otherwise a simplemc::simplemc_exception is thrown.
 *
 * @code{.cpp}
 * // gather a vector of integers on the root processes
 * simplemc::mpi::communicator comm {};
 * std::vector<int> data {};
 * int root {};
 *
 * // set data and root...
 *
 * std::vector<int> result(comm.size() * data.size());
 * simplemc::mpi::gather(comm, data, result, root);
 * @endcode
 *
 * @tparam R1 Input range.
 * @tparam R2 Output range.
 * @param comm simplemc::mpi::communicator object.
 * @param in_values Range to gather.
 * @param out_values Range to be gathered into.
 * @param root Root process.
 */
template <mpi_range R1, mpi_range R2>
void gather(
    const communicator& comm, R1&& in_values, R2&& out_values, int root) { // NOLINT (ranges need not be forwarded)
    if (!all_equal(ranges::size(in_values), comm)) {
        throw simplemc_exception("Input range sizes are not equal across all processes", "mpi::gather");
    }
    if (comm.rank() == root) {
        if (comm.size() * ranges::size(in_values) != ranges::size(out_values)) {
            throw simplemc_exception("Output range size has incorrect size", "mpi::gather");
        }
    }
    gather(comm, ranges::data(in_values), ranges::size(in_values), ranges::data(out_values), root);
}

/**
 * @brief Gather a range in place (only on root process).
 *
 * @details It calls simplemc::mpi::gather_in_place.
 *
 * The ranges must have the same size across all non-root processes and on the root process its size
 * must be equal to the number of processes times the size of the other ranges, otherwise a
 * simplemc::simplemc_exception is thrown.
 *
 * @code{.cpp}
 * // gather a vector of integers in place on the processes
 * simplemc::mpi::communicator comm {};
 * std::vector<int> data {};
 * int root {};
 *
 * // resize and set data and root...
 *
 * simplemc::mpi::gather_in_place(comm, data, root);
 * @endcode
 *
 * @tparam R Input/Output range.
 * @param comm simplemc::mpi::communicator object.
 * @param in_out_values Range to be gathered/gathered into.
 * @param root Root process.
 */
template <mpi_range R>
void gather_in_place(const communicator& comm, R&& in_out_values, int root) { // NOLINT (ranges need not be forwarded)
    auto size = ranges::size(in_out_values);
    if (comm.rank() == root) {
        size /= comm.size();
    }
    if (!all_equal(size, comm)) {
        throw simplemc_exception("Range sizes are not compatible", "mpi::gather_in_place");
    }
    gather_in_place(comm, ranges::data(in_out_values), size, root);
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_GATHER_HPP
