/**
 * @file
 * @brief All-gather communications between MPI processes.
 */

#ifndef SIMPLEMC_MPI_ALL_GATHER_HPP
#define SIMPLEMC_MPI_ALL_GATHER_HPP

#include <simplemc/mpi/all_reduce.hpp>
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
 * @brief Gather a given number of values (on all processes).
 *
 * @details It calls `MPI_Allgather` if `count > 0`, otherwise it does nothing.
 *
 * If the MPI call fails, a simplemc::simplemc_exception is thrown.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param comm simplemc::mpi::communicator object.
 * @param in_values Pointer to the memory to be gathered.
 * @param count Number of values to be gathered.
 * @param out_values Pointer to the memory to be gathered into.
 */
template <mpi_compatible T>
void all_gather(const communicator& comm, const T* in_values, int count, T* out_values) {
    if (count <= 0) {
        return;
    }
    check_mpi_call(MPI_Allgather(in_values, count, mpi_type<T>::get(), out_values, count, mpi_type<T>::get(), comm),
        "MPI_'Allgather");
}

/**
 * @brief Gather a given number of values in place (on all processes).
 *
 * @details Same as simplemc::mpi::all_gather except that the MPI call is made with `MPI_IN_PLACE`
 * instead of a send buffer.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param comm simplemc::mpi::communicator object.
 * @param in_out_values Pointer to the memory to be gathered/gathered into.
 * @param count Number of values to be gathered.
 */
template <mpi_compatible T>
void all_gather_in_place(const communicator& comm, T* in_out_values, int count) {
    if (count <= 0) {
        return;
    }
    check_mpi_call(
        MPI_Allgather(MPI_IN_PLACE, count, mpi_type<T>::get(), in_out_values, count, mpi_type<T>::get(), comm),
        "MPI_'Allgather");
}

/**
 * @brief Gather a single value (on all processes).
 *
 * @details It calls simplemc::mpi::all_gather with a count of 1.
 *
 * The size of the output range has to be equal to the communicator size, otherwise a
 * simplemc::simplemc_exception is thrown.
 *
 * @code{.cpp}
 * // gather a single value on all processes
 * simplemc::mpi::communicator comm {};
 * int value {};
 *
 * // set value...
 *
 * std::vector<int> result(comm.size());
 * simplemc::mpi::all_gather(comm, value, result);
 * @endcode
 *
 * @tparam R Output range.
 * @param comm simplemc::mpi::communicator object.
 * @param in_value Value to be gathered.
 * @param out_values Range to gathered into.
 */
template <mpi_range R>
void all_gather(const communicator& comm, const ranges::range_value_t<R>& in_value,
    R&& out_values) { // NOLINT (ranges need not be forwarded)
    if (comm.size() != static_cast<int>(ranges::size(out_values))) {
        throw simplemc_exception("Output range size is not equal to communicator size", "mpi::all_gather");
    }
    all_gather(comm, &in_value, 1, ranges::data(out_values));
}

/**
 * @brief Gather a range (on all processes).
 *
 * @details It calls simplemc::mpi::all_gather.
 *
 * The input ranges must have the same size across all processes and the size of the output range
 * must be equal to the size of the input range times the number of processes, otherwise a
 * simplemc::simplemc_exception is thrown.
 *
 * @code{.cpp}
 * // gather a vector of integers on all processes
 * simplemc::mpi::communicator comm {};
 * std::vector<int> data {};
 *
 * // set data...
 *
 * std::vector<int> result(comm.size() * data.size());
 * simplemc::mpi::all_gather(comm, data, result);
 * @endcode
 *
 * @tparam R1 Input range.
 * @tparam R2 Output range.
 * @param comm simplemc::mpi::communicator object.
 * @param in_values Range to be gathered.
 * @param out_values Range to be gathered into.
 */
template <mpi_range R1, mpi_range R2>
void all_gather(const communicator& comm, R1&& in_values, R2&& out_values) { // NOLINT
    if (!all_equal(comm, ranges::size(in_values))) {
        throw simplemc_exception("Input range sizes are not equal across all processes", "mpi::all_gather");
    }
    if (comm.size() * ranges::size(in_values) != ranges::size(out_values)) {
        throw simplemc_exception("Output range size has incorrect size", "mpi::all_gather");
    }
    all_gather(comm, ranges::data(in_values), ranges::size(in_values), ranges::data(out_values));
}

/**
 * @brief Gather a range in place (on all processes).
 *
 * @details It calls simplemc::mpi::all_gather_in_place.
 *
 * The ranges must have the same size across all processes and the size of the range must be a
 * multiple of the number of processes, otherwise a simplemc::simplemc_exception is thrown.
 *
 * @code{.cpp}
 * // gather a vector of integers in place on all processes
 * simplemc::mpi::communicator comm {};
 * std::vector<int> data {};
 *
 * // resize and set data...
 *
 * simplemc::mpi::all_gather_in_place(comm, data, result);
 * @endcode
 *
 * @tparam R Input/Output range.
 * @param comm simplemc::mpi::communicator object.
 * @param in_out_values Range to be gathered/gathered into.
 */
template <mpi_range R>
void all_gather_in_place(const communicator& comm, R&& in_out_values) { // NOLINT
    if (!all_equal(comm, ranges::size(in_out_values))) {
        throw simplemc_exception("Range sizes are not equal across all processes", "mpi::all_gather_in_place");
    }
    if (ranges::size(in_out_values) % comm.size() != 0) {
        throw simplemc_exception("Range size has incorrect size", "mpi::all_gather_in_place");
    }
    all_gather_in_place(comm, ranges::data(in_out_values), ranges::size(in_out_values) / comm.size());
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_ALL_GATHER_HPP
