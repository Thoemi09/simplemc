/**
 * @file
 * @brief Reduce communications between MPI processes.
 */

#ifndef SIMPLEMC_MPI_REDUCE_HPP
#define SIMPLEMC_MPI_REDUCE_HPP

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
 * @brief Reduce a given number of values (only on root process).
 *
 * @details It calls `MPI_Reduce` if `count > 0`, otherwise it does nothing.
 *
 * If the MPI call fails or if `root` is not a valid process ID, a simplemc::simplemc_exception is
 * thrown.
 *
 * The MPI operation should be one of the following: `MPI_MAX`, `MPI_MIN`, `MPI_SUM`, `MPI_PROD`,
 * `MPI_LAND`, `MPI_BAND`, `MPI_LOR`, `MPI_BOR`, `MPI_LXOR`, `MPI_BXOR`.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param comm simplemc::mpi::communicator object.
 * @param in_values Pointer to the memory to be reduced.
 * @param count Number of values to be reduced.
 * @param out_values Pointer to the memory to be reduced into.
 * @param op MPI operation.
 * @param root Root process.
 */
template <mpi_compatible T>
void reduce(const communicator& comm, const T* in_values, int count, T* out_values, MPI_Op op, int root) {
    if (count <= 0) {
        return;
    }
    if (root < 0 || root >= comm.size()) {
        throw simplemc_exception("Root process is out of bounds", "mpi::reduce");
    }
    check_mpi_call(MPI_Reduce(in_values, out_values, count, mpi_type<T>::get(), op, root, comm), "MPI_Reduce");
}

/**
 * @brief Reduce a given number of values in place (only on root process).
 *
 * @details Same as simplemc::mpi::reduce except that the MPI call is made with `MPI_IN_PLACE` instead
 * of a send buffer on the root process.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param comm simplemc::mpi::communicator object.
 * @param in_out_values Pointer to the memory to be reduced/reduced into.
 * @param count Number of values to be reduced.
 * @param op MPI operation.
 * @param root Root process.
 */
template <mpi_compatible T>
void reduce_in_place(const communicator& comm, T* in_out_values, int count, MPI_Op op, int root) {
    if (count <= 0) {
        return;
    }
    if (root < 0 || root >= comm.size()) {
        throw simplemc_exception("Root process is out of bounds", "mpi::reduce_in_place");
    }
    if (comm.rank() == root) {
        check_mpi_call(
            MPI_Reduce(MPI_IN_PLACE, in_out_values, count, mpi_type<T>::get(), op, root, comm), "MPI_Reduce");
    } else {
        check_mpi_call(
            MPI_Reduce(in_out_values, in_out_values, count, mpi_type<T>::get(), op, root, comm), "MPI_Reduce");
    }
}

/**
 * @brief Reduce a single value (only on root process).
 *
 * @details It calls simplemc::mpi::reduce with a count of 1.
 *
 * @code{.cpp}
 * // reduce a single value on the root processes
 * simplemc::mpi::communicator comm {};
 * int value {}, root {};
 *
 * // set value and root...
 *
 * int result {};
 * simplemc::mpi::reduce(comm, value, result, MPI_SUM, root);
 * @endcode
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param comm simplemc::mpi::communicator object.
 * @param in_value Value to be reduced.
 * @param out_value Value to be reduced into.
 * @param op MPI operation.
 * @param root Root process.
 */
template <mpi_compatible T>
void reduce(const communicator& comm, const T& in_value, T& out_value, MPI_Op op, int root) {
    reduce(comm, &in_value, 1, &out_value, op, root);
}

/**
 * @brief Reduce a single value in place (only on root process).
 *
 * @details It calls simplemc::mpi::reduce_in_place with a count of 1.
 *
 * @code{.cpp}
 * // reduce a single value on the root processes in place
 * simplemc::mpi::communicator comm {};
 * int value {}, root {};
 *
 * // set value and root...
 *
 * simplemc::mpi::reduce_in_place(comm, value, MPI_SUM, root);
 * @endcode
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param comm simplemc::mpi::communicator object.
 * @param in_out_value Value to be reduced/reduced into.
 * @param op MPI operation.
 * @param root Root process.
 */
template <mpi_compatible T>
void reduce_in_place(const communicator& comm, T& in_out_value, MPI_Op op, int root) {
    reduce_in_place(comm, &in_out_value, 1, op, root);
}

/**
 * @brief Reduce a range (only on root process).
 *
 * @details It calls simplemc::mpi::reduce.
 *
 * The input ranges must have the same size across all processes and the size of the output range on
 * the root process must be equal to the size of the input range, otherwise a
 * simplemc::simplemc_exception is thrown.
 *
 * @code{.cpp}
 * // reduce a vector of integers on the root processes
 * simplemc::mpi::communicator comm {};
 * std::vector<int> data {};
 * int root {}
 *
 * // fill data and set root...
 *
 * std::vector<int> result(data.size());
 * simplemc::mpi::reduce(comm, data, result, MPI_SUM, root);
 * @endcode
 *
 * @tparam R1 Input range.
 * @tparam R2 Output range.
 * @param comm simplemc::mpi::communicator object.
 * @param in_values Range to be reduced.
 * @param out_values Range to be reduced into.
 * @param op MPI operation.
 * @param root Root process.
 */
template <mpi_range R1, mpi_range R2>
void reduce(const communicator& comm, R1&& in_values, R2&& out_values, MPI_Op op, int root) { // NOLINT
    if (!all_equal(comm, ranges::size(in_values))) {
        throw simplemc_exception("Input range sizes are not equal across all processes", "mpi::reduce");
    }
    if (comm.rank() == root) {
        if (ranges::size(in_values) != ranges::size(out_values)) {
            throw simplemc_exception("Input and output range sizes are not equal", "mpi::reduce");
        }
    }
    reduce(comm, ranges::data(in_values), ranges::size(in_values), ranges::data(out_values), op, root);
}

/**
 * @brief Reduce a range in place (only on root process).
 *
 * @details It calls simplemc::mpi::reduce_in_place.
 *
 * The ranges must have the same size across all processes, otherwise a simplemc::simplemc_exception
 * is thrown.
 *
 * @code{.cpp}
 * // reduce a vector of integers on the root processes in place
 * simplemc::mpi::communicator comm {};
 * std::vector<int> data {};
 * int root {}
 *
 * // fill data and set root...
 *
 * simplemc::mpi::reduce(comm, data, MPI_SUM, root);
 * @endcode
 *
 * @tparam R Input/Output range.
 * @param comm simplemc::mpi::communicator object.
 * @param in_out_values Range to be reduced/reduced into.
 * @param op MPI operation.
 * @param root Root process.
 */
template <mpi_range R>
void reduce_in_place(
    const communicator& comm, R&& in_out_values, MPI_Op op, int root) { // NOLINT (ranges need not be forwarded)
    if (!all_equal(comm, ranges::size(in_out_values))) {
        throw simplemc_exception("Range sizes are not equal across all processes", "mpi::reduce_in_place");
    }
    reduce_in_place(comm, ranges::data(in_out_values), ranges::size(in_out_values), op, root);
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_REDUCE_HPP
