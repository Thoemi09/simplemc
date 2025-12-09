/**
 * @file
 * @brief RAII wrapper for `MPI_Comm` objects and other essential MPI functions involving
 * communicators.
 */

#ifndef SIMPLEMC_MPI_COMMUNICATOR_HPP
#define SIMPLEMC_MPI_COMMUNICATOR_HPP

#include <simplemc/mpi/group.hpp>
#include <simplemc/mpi/mpi_fwd.hpp>
#include <simplemc/mpi/utils.hpp>

#include <mpi.h>

#include <memory>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-essentials-comms
 * @{
 */

/**
 * @brief Free an `MPI_Comm` object.
 *
 * @details It makes a call to `MPI_Comm_free` to free the communicator in case it is not
 * `MPI_COMM_NULL`, `MPI_COMM_WORLD`, or `MPI_COMM_SELF`. Otherwise, it does nothing.
 *
 * @param comm Reference to the communicator to free.
 */
void comm_free(MPI_Comm& comm);

/**
 * @brief Get the size of an MPI communicator by calling `MPI_Comm_size`.
 *
 * @param comm Communicator to query.
 * @return Number of processes in the communicator.
 */
[[nodiscard]] int comm_size(MPI_Comm comm);

/**
 * @brief Get the rank of the calling process in an MPI communicator by calling `MPI_Comm_rank`.
 *
 * @param comm Communicator to query.
 * @return Rank of the calling process in the communicator.
 */
[[nodiscard]] int comm_rank(MPI_Comm comm);

/**
 * @brief Get the MPI group associated with an MPI communicator by calling `MPI_Comm_group`.
 *
 * @note The caller is responsible for freeing the returned `MPI_Group` using
 * simplemc::mpi::group_free or `MPI_Group_free()`.
 *
 * @param comm Communicator to query.
 * @return Group associated with the communicator.
 */
[[nodiscard]] MPI_Group comm_group(MPI_Comm comm);

/**
 * @brief Duplicate an MPI communicator by calling `MPI_Comm_dup`.
 *
 * @note The caller is responsible for freeing the returned `MPI_Comm` using simplemc::mpi::comm_free
 * or `MPI_Comm_free()`.
 *
 * @param comm Communicator to duplicate.
 * @return New communicator object that is a duplicate of the input.
 */
[[nodiscard]] MPI_Comm comm_dup(MPI_Comm comm);

/**
 * @brief Split an MPI communicator by calling `MPI_Comm_split`.
 *
 * @details It partitions the communicator into disjoint subgroups based on color and key. Processes
 * with the same color are placed in the same new communicator, and the key determines the rank
 * ordering within each new communicator.
 *
 * Use `MPI_UNDEFINED` as the color if a process should not be part of any new communicator (returns
 * `MPI_COMM_NULL` for that process).
 *
 * @note The caller is responsible for freeing the returned `MPI_Comm` using simplemc::mpi::comm_free
 * or `MPI_Comm_free()`.
 *
 * @param comm Communicator to split.
 * @param color Control of subset assignment (processes with the same color are in the same group).
 * @param key Control of rank assignment within the new communicator.
 * @return New communicator object for the calling process's subgroup.
 */
[[nodiscard]] MPI_Comm comm_split(MPI_Comm comm, int color, int key = 0);

/**
 * @brief Create a new MPI communicator from an MPI group by calling `MPI_Comm_create`.
 *
 * @details It creates a new communicator containing all processes in the specified group. All
 * processes in `comm` must call this function, but only processes in `grp` will get a valid
 * communicator. Processes not in `grp` will receive `MPI_COMM_NULL`.
 *
 * @note The caller is responsible for freeing the returned `MPI_Comm` using simplemc::mpi::comm_free
 * or `MPI_Comm_free()`.
 *
 * @param comm Parent communicator.
 * @param grp Group defining the processes for the new communicator.
 * @return New communicator for processes in the group, or `MPI_COMM_NULL` for others.
 */
[[nodiscard]] MPI_Comm comm_create(MPI_Comm comm, MPI_Group grp);

/**
 * @brief Compare two MPI communicators by calling `MPI_Comm_compare`.
 *
 * @details The return value is one of:
 * - `MPI_IDENT`: Communicators are identical (same context and group).
 * - `MPI_CONGRUENT`: Communicators have the same group but different contexts.
 * - `MPI_SIMILAR`: Communicators have groups with the same members but in different order.
 * - `MPI_UNEQUAL`: Communicators have different groups.
 *
 * @param comm1 First communicator.
 * @param comm2 Second communicator.
 * @return The comparison result (`MPI_IDENT`, `MPI_CONGRUENT`, `MPI_SIMILAR`, or `MPI_UNEQUAL`).
 */
[[nodiscard]] int comm_compare(MPI_Comm comm1, MPI_Comm comm2);

/**
 * @brief Attach a new error handler to the given MPI communicator by calling
 * `MPI_Comm_set_errhandler`.
 *
 * @details Common error handlers are:
 * - `MPI_ERRORS_ARE_FATAL`: MPI will abort on error (default).
 * - `MPI_ERRORS_RETURN`: MPI will return an error code instead of aborting.
 * - `MPI_ERRORS_ABORT`: MPI will abort on error (similar to `MPI_ERRORS_ARE_FATAL`).
 *
 * @param comm Communicator to attach the error handler to.
 * @param errhandler MPI error handler.
 */
void comm_set_errhandler(MPI_Comm comm, MPI_Errhandler errhandler);

/**
 * @brief RAII wrapper for `MPI_Comm` objects.
 *
 * @details This class provides a thin C++ wrapper around `MPI_Comm` objects. MPI communicators
 * consist of a simplemc::mpi::group of MPI processes that can communicate (send messages) among
 * themselves.
 *
 * The wrapped `MPI_Comm` is managed using a `std::shared_ptr` with a custom deleter providing
 * automatic resource management similar to <a href="https://www.boost.org/doc/libs/latest/doc/html/
 * mpi.html">Boost.MPI</a>. The behaviour of the deleter depends on the simplemc::mpi::resource_policy
 * specified during construction:
 * - `take_ownership`: The wrapper is responsible for managing the lifetime of the MPI communicator,
 * i.e. it calls `MPI_Comm_free` when the last reference is destroyed (unless it is `MPI_COMM_NULL`,
 * `MPI_COMM_WORLD` or `MPI_COMM_SELF`).
 * - `attach`: The lifetime of the MPI communicator is managed somewhere else and the wrapper just
 * uses it.
 *
 * Implicit conversions to `MPI_Comm` are allowed for interoperability with MPI C functions.
 */
class communicator {
public:
    /**
     * @brief Default construct a communicator for `MPI_COMM_WORLD`.
     *
     * @details It wraps `MPI_COMM_WORLD` without taking ownership, meaning `MPI_Comm_free` will not
     * be called when the communicator is destroyed.
     */
    communicator() : comm_(std::make_shared<MPI_Comm>(MPI_COMM_WORLD)) {}

    /**
     * @brief Construct a communicator from an existing `MPI_Comm` with a given
     * simplemc::mpi::resource_policy.
     *
     * @details If the resource policy is set to `take_ownership` (the default), the communicator
     * takes ownership of the provided `MPI_Comm` and will free it when the last reference is
     * destroyed.
     *
     * If the policy is `attach`, the communicator simply attaches to the provided `MPI_Comm` without
     * taking ownership, and the caller remains responsible for freeing it.
     *
     * @param comm `MPI_Comm` object to wrap.
     * @param pol simplemc::mpi::resource_policy indicating ownership semantics.
     */
    explicit communicator(MPI_Comm comm, resource_policy pol = resource_policy::take_ownership);

    /**
     * @brief Get the rank of the calling process in the communicator by calling
     * simplemc::mpi::comm_rank.
     *
     * @return Rank of the calling process in this communicator.
     */
    [[nodiscard]] int rank() const { return comm_rank(*comm_); }

    /**
     * @brief Get the size of the communicator by calling simplemc::mpi::comm_size.
     *
     * @return Number of processes in this communicator.
     */
    [[nodiscard]] int size() const { return comm_size(*comm_); }

    /**
     * @brief Get the group associated with this communicator.
     *
     * @details It calls simplemc::mpi::comm_group with the wrapped `MPI_Comm` and wraps the result
     * in a simplemc::mpi::group object.
     *
     * @param pol simplemc::mpi::resource_policy indicating ownership semantics for the new group.
     * @return Group associated with this communicator.
     */
    [[nodiscard]] group get_group(resource_policy pol = resource_policy::take_ownership) const {
        return group { comm_group(*comm_), pol };
    }

    /**
     * @brief Create a new communicator by duplicating this one.
     *
     * @details It calls simplemc::mpi::comm_dup with the wrapped `MPI_Comm` and wraps the result.
     *
     * @param pol simplemc::mpi::resource_policy indicating ownership semantics for the new
     * communicator.
     * @return New communicator that is a duplicate of this one.
     */
    [[nodiscard]] communicator duplicate(resource_policy pol = resource_policy::take_ownership) const {
        return communicator { comm_dup(*comm_), pol };
    }

    /**
     * @brief Split the communicator into multiple communicators based on color and key.
     *
     * @details It calls simplemc::mpi::comm_split with the wrapped `MPI_Comm` and wraps the result.
     *
     * @param color Control of subset assignment (processes with the same color are in the same
     * group).
     * @param key Control of rank assignment within the new communicator.
     * @param pol simplemc::mpi::resource_policy indicating ownership semantics for the new
     * communicator.
     * @return New communicator for the calling process's subgroup.
     */
    [[nodiscard]] communicator split(
        int color, int key = 0, resource_policy pol = resource_policy::take_ownership) const {
        return communicator { comm_split(*comm_, color, key), pol };
    }

    /**
     * @brief Create a new communicator from an MPI group.
     *
     * @details It calls simplemc::mpi::comm_create with the wrapped `MPI_Comm` and the provided
     * group and wraps the result.
     *
     * @param grp Group specifying which processes should be in the new communicator.
     * @param pol simplemc::mpi::resource_policy indicating ownership semantics for the new
     * communicator.
     * @return New communicator containing the processes in the group.
     */
    [[nodiscard]] communicator create(MPI_Group grp, resource_policy pol = resource_policy::take_ownership) const {
        return communicator { comm_create(*comm_, grp), pol };
    }

    /**
     * @brief Compare this communicator with another communicator by calling
     * simplemc::mpi::comm_compare.
     *
     * @param other Other communicator to compare with.
     * @return The comparison result (`MPI_IDENT`, `MPI_CONGRUENT`, `MPI_SIMILAR`, or `MPI_UNEQUAL`).
     */
    [[nodiscard]] int compare(const communicator& other) const { return comm_compare(*comm_, other); }

    /**
     * @brief Wait for all processes within the communicator to reach this barrier.
     *
     * @details It calls `MPI_Barrier` with the wrapped `MPI_Comm`.
     */
    void barrier() const { check_mpi_call(MPI_Barrier(*comm_), "MPI_Barrier"); }

    /**
     * @brief Implicit conversion to the underlying `MPI_Comm` object.
     *
     * @return Underlying `MPI_Comm` or `MPI_COMM_NULL` if no communicator is wrapped.
     */
    operator MPI_Comm() const { return comm_ ? *comm_ : MPI_COMM_NULL; }

    /**
     * @brief Explicit conversion to bool to check if the communicator is valid.
     *
     * @return True if the communicator is not `MPI_COMM_NULL`.
     */
    [[nodiscard]] explicit operator bool() const { return comm_ && *comm_ != MPI_COMM_NULL; }

    /**
     * @brief Equality comparison operator to check if two communicators are identical.
     *
     * @details See also simplemc::mpi::comm_compare.
     *
     * @param c1 First communicator.
     * @param c2 Second communicator.
     * @return True if the communicators are identical (`MPI_IDENT`).
     */
    [[nodiscard]] friend bool operator==(const communicator& c1, const communicator& c2) {
        return c1.compare(c2) == MPI_IDENT;
    }

private:
    // Custom deleter that makes a protected call to MPI_Comm_free.
    struct comm_deleter {
        void operator()(MPI_Comm* c) const noexcept;
    };

private:
    std::shared_ptr<MPI_Comm> comm_;
};

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_COMMUNICATOR_HPP
