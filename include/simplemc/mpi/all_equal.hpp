/**
 * @file
 * @brief Check that a value or range of values is equal on all MPI processes.
 */

#ifndef SIMPLEMC_MPI_ALL_EQUAL_HPP
#define SIMPLEMC_MPI_ALL_EQUAL_HPP

#include <simplemc/mpi/mpi_type.hpp>
#include <simplemc/mpi/utils.hpp>
#include <simplemc/utils/ranges.hpp>

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <vector>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-coll-other
 * @{
 */

/**
 * @brief Check if a value is equal on all processes.
 *
 * @details It performs the following steps:
 *
 * - Broadcast the value from rank 0 to all other processes.
 * - Each process compares its local value to the broadcasted value for equality.
 * - All-reduce the comparison results using `MPI_LAND`.
 *
 * @note This function involves two collective operations, so all processes must call it.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param value Value to compare across processes.
 * @param comm MPI communicator.
 * @return True if all processes have the same value, false otherwise.
 */
template <mpi_compatible T>
[[nodiscard]] bool all_equal(const T& value, MPI_Comm comm) {
    // broadcast the value from rank 0
    T root_value = value;
    check_mpi_call(MPI_Bcast(&root_value, 1, mpi_type<T>::get(), 0, comm), "MPI_Bcast");

    // compare local value with broadcasted value
    const int local_equal = (value == root_value);

    // all-reduce the comparison result
    int all_equal {};
    check_mpi_call(MPI_Allreduce(&local_equal, &all_equal, 1, mpi_type<int>::get(), MPI_LAND, comm), "MPI_Allreduce");
    return all_equal;
}

/**
 * @brief Check if all elements in a range are equal on all processes.
 *
 * @details It performs the following steps:
 *
 * - Check that the range sizes are simplemc::mpi::all_equal.
 * - Broadcast the range from rank 0 to all other processes.
 * - Each process compares its local range to the broadcasted range element-wise for equality.
 * - All-reduce the comparison results using `MPI_LAND`.
 *
 * @note This function involves three collective operations, so all processes must call it.
 *
 * @tparam R simplemc::mpi::mpi_range type.
 * @param rg Range to compare across processes.
 * @param comm MPI communicator.
 * @return True if all processes have ranges with identical size and content, false otherwise.
 */
template <mpi_range R>
[[nodiscard]] bool all_equal(R&& rg, MPI_Comm comm) { // NOLINT (ranges need not be forwarded)
    using value_type = ranges::range_value_t<R>;

    // check if all processes have the same size
    const int size = ranges::size(rg);
    if (!all_equal(size, comm)) {
        return false;
    }

    // broadcast the range from rank 0
    auto root_rg = std::vector(ranges::begin(rg), ranges::end(rg));
    check_mpi_call(MPI_Bcast(root_rg.data(), size, mpi_type<value_type>::get(), 0, comm), "MPI_Bcast");

    // compare local range with broadcasted values
    const int local_equal = std::ranges::equal(rg, root_rg) ? 1 : 0;

    // all-reduce the comparison result
    int all_equal {};
    check_mpi_call(MPI_Allreduce(&local_equal, &all_equal, 1, mpi_type<int>::get(), MPI_LAND, comm), "MPI_Allreduce");
    return all_equal;
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_ALL_EQUAL_HPP
