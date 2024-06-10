/**
 * @file
 * @brief Collective communications between MPI processes.
 */

#ifndef SIMPLEMC_MPI_COLLECTIVES_HPP
#define SIMPLEMC_MPI_COLLECTIVES_HPP

#include <simplemc/mpi/communicator.hpp>
#include <simplemc/mpi/datatypes.hpp>
#include <simplemc/mpi/utils.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <mpi.h>
#include <range/v3/range/concepts.hpp>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-coll
 * @{
 */

/**
 * @brief Reduce a specific number of values (on all processes).
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
 * @brief Reduce a single value (on all processes).
 *
 * @details It calls simplemc::mpi::all_reduce with a count of 1.
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
 * @brief Check if a value is equal across all processes (on all processes).
 *
 * @details It makes two calls to simplemc::mpi::all_reduce, one with `MPI_MIN` and the other with
 * `MPI_MAX`, and compares their results. If the results are equal, the function returns true,
 * otherwise false.
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
 * @brief Gather a specific number of values (on all processes).
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
 * @brief Gather a single value (on all processes).
 *
 * @details It calls simplemc::mpi::all_gather with a count of 1.
 *
 * The size of the output range has to be equal to the communicator size, otherwise a
 * simplemc::simplemc_exception is thrown.
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
 * @brief Reduce a specific number of values (only on root process).
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
 * @brief Reduce a single value (only on root process).
 *
 * @details It calls simplemc::mpi::reduce with a count of 1.
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
 * @brief Reduce a range (only on root process).
 *
 * @details It calls simplemc::mpi::reduce.
 *
 * The input ranges must have the same size across all processes and the size of the output range on
 * the root process must be equal to the size of the input range, otherwise a
 * simplemc::simplemc_exception is thrown.
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
 * @brief Gather a specific number of values (only on root process).
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
 * @brief Gather a single value (only on root process).
 *
 * @details It calls simplemc::mpi::gather with a count of 1.
 *
 * The size of the output range on the root process has to be equal to the communicator size,
 * otherwise a simplemc::simplemc_exception is thrown.
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
 * @tparam R1 Input range.
 * @tparam R2 Output range.
 * @param comm simplemc::mpi::communicator object.
 * @param in_values Range to gather.
 * @param out_values Range to be gathered into.
 * @param root Root process.
 */
template <mpi_range R1, mpi_range R2>
void gather(const communicator& comm, R1&& in_values, R2&& out_values, int root) { // NOLINT
    if (!all_equal(comm, ranges::size(in_values))) {
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
 * @brief Broadcast a specific number of values.
 *
 * @details It calls `MPI_Bcast` if `count > 0`, otherwise it does nothing.
 *
 * If the MPI call fails or if `root` is not a valid process ID, a simplemc::simplemc_exception is
 * thrown.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param comm simplemc::mpi::communicator object.
 * @param values Pointer to the memory to be broadcasted from/into.
 * @param count Number of values to be broadcasted.
 * @param root Root process.
 */
template <mpi_compatible T>
void broadcast(const communicator& comm, T* values, int count, int root) {
    if (count <= 0) {
        return;
    }
    if (root < 0 || root >= comm.size()) {
        throw simplemc_exception("Root process is out of bounds", "mpi::broadcast");
    }
    check_mpi_call(MPI_Bcast(values, count, mpi_type<T>::get(), root, comm), "MPI_Bcast");
}

/**
 * @brief Broadcast a single value.
 *
 * @details It calls simplemc::mpi::broadcast with a count of 1.
 *
 * @tparam T simplemc::mpi::mpi_compatible type.
 * @param comm simplemc::mpi::communicator object.
 * @param value Value to be broadcasted from/into.
 * @param root Root process.
 */
template <mpi_compatible T>
void broadcast(const communicator& comm, T& value, int root) {
    broadcast(comm, &value, 1, root);
}

/**
 * @brief Broadcast a range.
 *
 * @details It calls simplemc::mpi::broadcast.
 *
 * The ranges must have the same size across all processes, otherwise a simplemc::simplemc_exception
 * is thrown.
 *
 * @tparam R Input/Output range.
 * @param comm simplemc::mpi::communicator object.
 * @param rg Range to be broadcasted from/into.
 * @param root Root process.
 */
template <mpi_range R>
void broadcast(const communicator& comm, R&& rg, int root) { // NOLINT
    if (!all_equal(comm, ranges::size(rg))) {
        throw simplemc_exception("Range sizes are not equal across all processes", "mpi::broadcast");
    }
    broadcast(comm, ranges::data(rg), ranges::size(rg), root);
}

/**
 * @brief Scatter a specific number of values.
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
 * @brief Scatter `N` values, where `N` is equal to the communicator size.
 *
 * @details It calls simplemc::mpi::scatter with a count of 1.
 *
 * The size of the input range on the root process has to be equal to the communicator size, otherwise
 * a simplemc::simplemc_exception is thrown.
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
 * @brief Scatter `M * N` values, where `N` is equal to the communicator size.
 *
 * @details It calls simplemc::mpi::scatter with a count of `M`, where `M` is some positive integer.
 *
 * The size of the input range on the root process has to be a multiple (`M`) of the communicator size
 * and the size of the output range on each process has to be equal to `M`, otherwise a
 * simplemc::simplemc_exception is thrown.
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
            throw simplemc_exception("Input range size is not a mulitple of communicator size", "mpi::scatter");
        }
    }
    broadcast(comm, chunk_size, root);
    if (chunk_size != ranges::size(out_values)) {
        throw simplemc_exception("Output range size is incorrect", "mpi::scatter");
    }
    scatter(comm, ranges::data(in_values), chunk_size, ranges::data(out_values), root);
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_COLLECTIVES_HPP
