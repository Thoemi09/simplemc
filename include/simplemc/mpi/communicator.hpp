/**
 * @file
 * @brief Wrapper around an `MPI_Comm` MPI communicator.
 */

#ifndef SIMPLEMC_MPI_COMMUNICATOR_HPP
#define SIMPLEMC_MPI_COMMUNICATOR_HPP

#include <simplemc/mpi/mpi_fwd.hpp>
#include <simplemc/mpi/utils.hpp>

#include <mpi.h>

#include <memory>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-essentials
 * @{
 */

/**
 * @brief A communicator that enables communication and synchronization among a set of MPI processes.
 *
 * @details This class provides a thin C++ wrapper around `MPI_Comm` objects. MPI communicators
 * consist of a simplemc::mpi::group of MPI processes that can communicate (send messages) among
 * themselves.
 *
 * The communicator is managed using a `std::shared_ptr` with a custom deleter providing automatic
 * resource management similar to <a href="https://www.boost.org/doc/libs/latest/doc/html/mpi.html">
 * Boost.MPI </a>. The behaviour of the deleter depends on the simplemc::mpi::resource_policy
 * specified during construction:
 * - `take_ownership`: The wrapper is responsible for managing the lifetime of the MPI communicator,
 * i.e. it calls `MPI_Comm_free` when the last reference is destroyed (unless it is `MPI_GROUP_NULL`
 * or `MPI_GROUP_EMPTY`).
 * - `attach`: The lifetime of the MPI communicator is managed somewhere else and the wrapper just
 * uses it.
 *
 * Implicit conversions to `MPI_Comm` are allowed for interoperability with MPI C functions.
 *
 * See @ref tut_mpi_1 for a full example.
 */
class communicator {
public:
    /**
     * @brief Default construct a communicator for `MPI_COMM_WORLD`.
     *
     * @details It wraps `MPI_COMM_WORLD` without taking ownership, meaning `MPI_Comm_free` will not
     * be called when the communicator is destroyed.
     */
    communicator();

    /**
     * @brief Construct a communicator from an existing `MPI_Comm` with a given
     * simplemc::mpi::resource_policy.
     *
     * @details If the resource policy is set to`take_ownership` (the default), the communicator takes
     * ownership of the provided `MPI_Comm` and will free it when the last reference is destroyed.
     *
     * If the policy is `attach`, the communicator simply attaches to the provided `MPI_Comm` without
     * taking ownership, and the caller remains responsible for freeing it.
     *
     * @param comm `MPI_Comm` object to wrap.
     * @param pol simplemc::mpi::resource_policy indicating ownership semantics.
     */
    explicit communicator(MPI_Comm comm, resource_policy pol = resource_policy::take_ownership);

    /**
     * @brief Create a new communicator by duplicating this one.
     *
     * @details It makes a call to `MPI_Comm_dup` to create a duplicate of this communicator. The new
     * communicator takes ownership of the duplicated `MPI_Comm` (see simplemc::mpi::resource_policy).
     *
     * It throws an exception if the call fails.
     *
     * @return A new communicator that is a duplicate of this one.
     */
    [[nodiscard]] communicator duplicate() const;

    /**
     * @brief Create a new communicator from a simplemc::mpi::group.
     *
     * @details It makes a call to `MPI_Comm_create` to create a new communicator containing the
     * processes in the given group. The group should be valid subgroup of the current commnunicator's
     * group. The new communicator takes ownership of the created `MPI_Comm` (see
     * simplemc::mpi::resource_policy).
     *
     * It throws an exception if the call fails.
     *
     * @param grp simplemc::mpi::group specifying which processes should be in the new communicator.
     * @return A new communicator containing the processes in the group.
     */
    [[nodiscard]] communicator create(const group& grp) const;

    /**
     * @brief Determine the rank of the calling process.
     *
     * @details It makes a call to `MPI_Comm_rank` and throws an exception if the call fails.
     *
     * @return Rank of the calling process in this communicator.
     */
    [[nodiscard]] int rank() const;

    /**
     * @brief Determine the size of the communicator.
     *
     * @details It makes a call to `MPI_Comm_size` and throws an exception if the call fails.
     *
     * @return Number of processes in this communicator.
     */
    [[nodiscard]] int size() const;

    /**
     * @brief Get the group associated with this communicator.
     *
     * @details It makes a call to `MPI_Comm_group` to retrieve the group associated with this
     * communicator.
     *
     * It throws an exception if the call fails.
     *
     * @return Group associated with this communicator.
     */
    [[nodiscard]] group get_group() const;

    /**
     * @brief Wait for all processes within the communicator to reach this barrier.
     *
     * @details It makes a call to `MPI_Barrier` and throws an exception if the call fails.
     */
    void barrier() const;

    /**
     * @brief Split the communicator into multiple communicators based on color and key.
     *
     * @details It makes a call to `MPI_Comm_split` to partition the communicator into disjoint
     * subgroups. Processes with the same color are placed in the same new communicator, and the key
     * determines the rank ordering within each new communicator.
     *
     * Use `MPI_UNDEFINED` as the color if a process should not be part of any new communicator
     * (returns an invalid communicator for that process).
     *
     * The new communicator takes ownership of the created `MPI_Comm`.
     *
     * It throws an exception if the call fails.
     *
     * @param color Control of subset assignment (processes with the same color are in the same
     * group).
     * @param key Control of rank assignment within the new communicator.
     * @return A new communicator for the calling process's subgroup.
     */
    [[nodiscard]] communicator split(int color, int key = 0) const;

    /**
     * @brief Implicit conversion to the underlying `MPI_Comm` object.
     *
     * @return The underlying `MPI_Comm` or `MPI_COMM_NULL` if no communicator is wrapped.
     */
    operator MPI_Comm() const { return comm_ ? *comm_ : MPI_COMM_NULL; }

    /**
     * @brief Explicit conversion to bool to check if the communicator is valid.
     *
     * @return True if the communicator is not `MPI_COMM_NULL`.
     */
    [[nodiscard]] explicit operator bool() const { return comm_ && *comm_ != MPI_COMM_NULL; }

private:
    // Custom deleter that makes a protected call to MPI_Comm_free.
    struct comm_deleter {
        void operator()(MPI_Comm* c) const;
    };

private:
    std::shared_ptr<MPI_Comm> comm_;
};

/**
 * @brief Compare two communicators.
 *
 * @details It makes a call to `MPI_Comm_compare` to compare the two communicators.
 *
 * The return value is one of:
 * - `MPI_IDENT`: Communicators are identical (same context and group).
 * - `MPI_CONGRUENT`: Communicators have the same group but different contexts.
 * - `MPI_SIMILAR`: Communicators have groups with the same members but in different order.
 * - `MPI_UNEQUAL`: Communicators have different groups.
 *
 * It throws an exception if the call fails.
 *
 * @param c1 First communicator.
 * @param c2 Second communicator.
 * @return The comparison result (`MPI_IDENT`, `MPI_CONGRUENT`, `MPI_SIMILAR`, or `MPI_UNEQUAL`).
 */
[[nodiscard]] int comm_compare(const communicator& c1, const communicator& c2);

/**
 * @brief Equality comparison operator to check if two communicators are identical.
 *
 * @details See simplemc::mpi::comm_compare for details.
 *
 * @param c1 First communicator.
 * @param c2 Second communicator.
 * @return True if the communicators are identical (same context and group).
 */
[[nodiscard]] inline bool operator==(const communicator& c1, const communicator& c2) {
    return comm_compare(c1, c2) == MPI_IDENT;
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_COMMUNICATOR_HPP
