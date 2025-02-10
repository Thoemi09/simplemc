/**
 * @file
 * @brief All-reduce communications between MPI processes.
 */

#ifndef SIMPLEMC_MPI_ALL_REDUCE_HPP
#define SIMPLEMC_MPI_ALL_REDUCE_HPP

#include <simplemc/mpi/communicator.hpp>
#include <simplemc/mpi/datatypes.hpp>
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
 * @brief Reduce a given number of values (on all processes).
 *
 * @details It calls `MPI_Allreduce` if `count > 0`, otherwise it does nothing.
 *
 * If the MPI call fails, a simplemc::simplemc_exception is thrown.
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
 */
template <mpi_compatible T>
void all_reduce(const communicator& comm, const T* in_values, int count, T* out_values, MPI_Op op) {
    if (count <= 0) {
        return;
    }
    check_mpi_call(MPI_Allreduce(in_values, out_values, count, mpi_type<T>::get(), op, comm), "MPI_Allreduce");
}

/**
 * @brief Reduce a given number of values in place (on all processes).
 *
 * @details Same as simplemc::mpi::all_reduce except that the MPI call is made with `MPI_IN_PLACE`
 * instead of a send buffer.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param comm simplemc::mpi::communicator object.
 * @param in_out_values Pointer to the memory to be reduced/reduced into.
 * @param count Number of values to be reduced.
 * @param op MPI operation.
 */
template <mpi_compatible T>
void all_reduce_in_place(const communicator& comm, T* in_out_values, int count, MPI_Op op) {
    if (count <= 0) {
        return;
    }
    check_mpi_call(MPI_Allreduce(MPI_IN_PLACE, in_out_values, count, mpi_type<T>::get(), op, comm), "MPI_Allreduce");
}

/**
 * @brief Reduce a single value (on all processes).
 *
 * @details It calls simplemc::mpi::all_reduce with a count of 1.
 *
 * @code{.cpp}
 * // reduce a single value on all processes
 * simplemc::mpi::communicator comm {};
 * int value {};
 *
 * // set value...
 *
 * int result {};
 * simplemc::mpi::all_reduce(comm, value, result, MPI_SUM);
 * @endcode
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param comm simplemc::mpi::communicator object.
 * @param in_value Value to be reduced.
 * @param out_value Value to be reduced into.
 * @param op MPI operation.
 */
template <mpi_compatible T>
void all_reduce(const communicator& comm, const T& in_value, T& out_value, MPI_Op op) {
    all_reduce(comm, &in_value, 1, &out_value, op);
}

/**
 * @brief Reduce a single value in place (on all processes).
 *
 * @details It calls simplemc::mpi::all_reduce_in_place with a count of 1.
 *
 * @code{.cpp}
 * // reduce a single value in place on all processes
 * simplemc::mpi::communicator comm {};
 * int value {};
 *
 * // set value...
 *
 * simplemc::mpi::all_reduce_in_place(comm, value, MPI_SUM);
 * @endcode
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param comm simplemc::mpi::communicator object.
 * @param in_out_value Value to be reduced/reduced into.
 * @param op MPI operation.
 */
template <mpi_compatible T>
void all_reduce_in_place(const communicator& comm, T& in_out_value, MPI_Op op) {
    all_reduce_in_place(comm, &in_out_value, 1, op);
}

/**
 * @brief Check if a value is equal across all processes (on all processes).
 *
 * @details It makes two calls to simplemc::mpi::all_reduce(const communicator&, const T&, T&,
 * MPI_Op), one with `MPI_MIN` and the other with `MPI_MAX`, and compares their results.
 *
 * If the results are equal, the function returns true, otherwise false.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param comm simplemc::mpi::communicator object.
 * @param in_value Value to be compared.
 * @return True if the value is equal on all processes.
 */
template <mpi_compatible T>
[[nodiscard]] bool all_equal(const communicator& comm, const T& in_value) {
    T min_val, max_val;
    all_reduce(comm, in_value, min_val, MPI_MIN);
    all_reduce(comm, in_value, max_val, MPI_MAX);
    return min_val == max_val;
}

/**
 * @brief Reduce a range (on all processes).
 *
 * @details It calls simplemc::mpi::all_reduce.
 *
 * The input and output ranges must have the same size across all processes, otherwise a
 * simplemc::simplemc_exception is thrown.
 *
 * @code{.cpp}
 * // reduce a vector of integers on all processes
 * simplemc::mpi::communicator comm {};
 * std::vector<int> data {};
 *
 * // fill data...
 *
 * std::vector<int> result(data.size());
 * simplemc::mpi::all_reduce(comm, data, result, MPI_SUM);
 * @endcode
 *
 * @tparam R1 Input range.
 * @tparam R2 Output range.
 * @param comm simplemc::mpi::communicator object.
 * @param in_values Range to be reduced.
 * @param out_values Range to be reduced into.
 * @param op MPI operation.
 */
template <mpi_range R1, mpi_range R2>
void all_reduce(
    const communicator& comm, R1&& in_values, R2&& out_values, MPI_Op op) { // NOLINT (ranges need not be forwarded)
    if (!all_equal(comm, ranges::size(in_values))) {
        throw simplemc_exception("Input range sizes are not equal across all processes", "mpi::all_reduce");
    }
    if (ranges::size(in_values) != ranges::size(out_values)) {
        throw simplemc_exception("Input and output range sizes are not equal", "mpi::all_reduce");
    }
    all_reduce(comm, ranges::data(in_values), ranges::size(in_values), ranges::data(out_values), op);
}

/**
 * @brief Reduce a range in place (on all processes).
 *
 * @details It calls simplemc::mpi::all_reduce_in_place.
 *
 * The ranges must have the same size across all processes, otherwise a simplemc::simplemc_exception
 * is thrown.
 *
 * @code{.cpp}
 * // reduce a vector of integers on all processes in place
 * simplemc::mpi::communicator comm {};
 * std::vector<int> data {};
 *
 * // fill data...
 *
 * simplemc::mpi::all_reduce_in_place(comm, data, MPI_SUM);
 * @endcode
 *
 * @tparam R Input/Output range.
 * @param comm simplemc::mpi::communicator object.
 * @param in_out_values Range to be reduced/reduced into.
 * @param op MPI operation.
 */
template <mpi_range R>
void all_reduce_in_place(
    const communicator& comm, R&& in_out_values, MPI_Op op) { // NOLINT (ranges need not be forwarded)
    if (!all_equal(comm, ranges::size(in_out_values))) {
        throw simplemc_exception("Range sizes are not equal across all processes", "mpi::all_reduce_in_place");
    }
    all_reduce_in_place(comm, ranges::data(in_out_values), ranges::size(in_out_values), op);
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_ALL_REDUCE_HPP
