/**
 * @file
 * @brief Wrapper around an `MPI_Group` object.
 */

#ifndef SIMPLEMC_MPI_GROUP_HPP
#define SIMPLEMC_MPI_GROUP_HPP

#include <simplemc/mpi/utils.hpp>
#include <simplemc/utils/ranges.hpp>

#include <mpi.h>

#include <concepts>
#include <memory>
#include <vector>

namespace simplemc::mpi {

// Forward declaration.
class communicator;

/**
 * @addtogroup simplemc-mpi-essentials
 * @{
 */

/**
 * @brief A group that represents an ordered set of MPI processes.
 *
 * @details This class provides a thin C++ wrapper around `MPI_Group` objects. MPI groups are used
 * to represent ordered sets of processes and can be used to create new communicators.
 *
 * The group is managed using a `std::shared_ptr` with a custom deleter that calls `MPI_Group_free`
 * when the last reference is destroyed, providing automatic resource management similar to
 * <a href="https://www.boost.org/doc/libs/latest/doc/html/mpi.html">Boost.MPI</a>.
 *
 * Implicit conversions to `MPI_Group` are allowed for interoperability with MPI C functions.
 */
class group {
public:
    /**
     * @brief Construct an empty group.
     *
     * @details It creates a group that wraps `MPI_GROUP_EMPTY`. This represents an empty group with
     * no processes.
     */
    group();

    /**
     * @brief Construct a group from an existing `MPI_Group`.
     *
     * @details If `take_ownership` is true (the default), the group takes ownership of the provided
     * `MPI_Group` and will free it when the last reference is destroyed (unless it is
     * `MPI_GROUP_NULL` or `MPI_GROUP_EMPTY`).
     *
     * If `take_ownership` is false, the group wraps the provided `MPI_Group` without taking
     * ownership, and the caller remains responsible for freeing it.
     *
     * @param grp `MPI_Group` object to wrap.
     * @param take_ownership If true, the group will free the `MPI_Group` when destroyed.
     */
    explicit group(MPI_Group grp, bool take_ownership = true);

    /**
     * @brief Determine the rank of the calling process in the group.
     *
     * @details It makes a call to `MPI_Group_rank` and throws an exception if the call fails.
     *
     * @return Rank of the calling process in this group, or `MPI_UNDEFINED` if it is not a member.
     */
    [[nodiscard]] int rank() const;

    /**
     * @brief Determine the size of the group.
     *
     * @details It makes a call to `MPI_Group_size` and throws an exception if the call fails.
     *
     * @return Number of processes in the group.
     */
    [[nodiscard]] int size() const;

    /**
     * @brief Check if the group is empty.
     *
     * @return True if the group has no members, i.e. if `size() == 0`.
     */
    [[nodiscard]] bool empty() const { return size() == 0; }

    /**
     * @brief Create a new group containing only the specified ranks.
     *
     * @details It makes a call to `MPI_Group_incl` to create a new group containing only the
     * processes with the specified ranks from this group.
     *
     * It throws an exception if the call fails.
     *
     * @tparam R Range type whose value type is an integral type.
     * @param ranks Range of ranks to include in the new group.
     * @return New group containing only the specified ranks.
     */
    template <simplemc::ranges::input_range R>
        requires std::integral<simplemc::ranges::range_value_t<R>>
    [[nodiscard]] group include(const R& ranks) const {
        auto ranks_vec = std::vector<int> { simplemc::ranges::begin(ranks), simplemc::ranges::end(ranks) };
        MPI_Group new_grp {};
        check_mpi_call(
            MPI_Group_incl(*grp_, static_cast<int>(ranks_vec.size()), ranks_vec.data(), &new_grp), "MPI_Group_incl");
        return group { new_grp };
    }

    /**
     * @brief Create a new group excluding the specified ranks.
     *
     * @details It makes a call to `MPI_Group_excl` to create a new group containing all processes
     * from this group except those with the specified ranks.
     *
     * It throws an exception if the call fails.
     *
     * @tparam R Range type whose value type is an integral type.
     * @param ranks Range of ranks to exclude from the new group.
     * @return New group excluding the specified ranks.
     */
    template <simplemc::ranges::input_range R>
        requires std::integral<simplemc::ranges::range_value_t<R>>
    [[nodiscard]] group exclude(const R& ranks) const {
        auto ranks_vec = std::vector<int> { simplemc::ranges::begin(ranks), simplemc::ranges::end(ranks) };
        MPI_Group new_grp {};
        check_mpi_call(
            MPI_Group_excl(*grp_, static_cast<int>(ranks_vec.size()), ranks_vec.data(), &new_grp), "MPI_Group_excl");
        return group { new_grp };
    }

    /**
     * @brief Translate ranks from this group to another group.
     *
     * @details It makes a call to `MPI_Group_translate_ranks` to translate the specified ranks from
     * this group to the corresponding ranks in the other group.
     *
     * For ranks that are not in the other group, `MPI_UNDEFINED` is returned.
     *
     * It throws an exception if the call fails.
     *
     * @tparam R Range type whose value type is an integral type.
     * @param ranks Ranks in this group to translate.
     * @param other The other group to translate to.
     * @return Vector of translated ranks in the other group.
     */
    template <simplemc::ranges::input_range R>
        requires std::integral<simplemc::ranges::range_value_t<R>>
    [[nodiscard]] std::vector<int> translate_ranks(const R& ranks, const group& other) const {
        auto ranks_vec = std::vector<int> { simplemc::ranges::begin(ranks), simplemc::ranges::end(ranks) };
        auto translated = std::vector<int>(ranks_vec.size());
        check_mpi_call(MPI_Group_translate_ranks(
                           *grp_, static_cast<int>(ranks_vec.size()), ranks_vec.data(), other, translated.data()),
            "MPI_Group_translate_ranks");
        return translated;
    }

    /**
     * @brief Implicit conversion to the underlying `MPI_Group` object.
     *
     * @details The underlying `MPI_Group` or `MPI_GROUP_NULL` if no MPI group is wrapped.
     */
    operator MPI_Group() const { return grp_ ? *grp_ : MPI_GROUP_NULL; }

    /**
     * @brief Explicit conversion to bool to check if the group is valid.
     *
     * @return True if the group is not `MPI_GROUP_NULL`.
     */
    [[nodiscard]] explicit operator bool() const { return grp_ && *grp_ != MPI_GROUP_NULL; }

private:
    // Custom deleter that makes a protected call to MPI_Group_free.
    struct group_deleter {
        void operator()(MPI_Group* g) const;
    };

private:
    // Shared pointer to the underlying MPI_Group.
    std::shared_ptr<MPI_Group> grp_;
};

/**
 * @brief Create a new group that is the union of two groups.
 *
 * @details It makes a call to `MPI_Group_union` to create a new group containing all processes that
 * are in the first group or the second group (or both).
 *
 * It throws an exception if the call fails.
 *
 * @param g1 First group.
 * @param g2 Second group.
 * @return New simplemc::mpi::group containing the union of both groups.
 */
[[nodiscard]] group group_union(const group& g1, const group& g2);

/**
 * @brief Create a new group that is the intersection of two groups.
 *
 * @details It makes a call to `MPI_Group_intersection` to create a new group containing only
 * processes that are in both groups.
 *
 * It throws an exception if the call fails.
 *
 * @param g1 First group.
 * @param g2 Second group.
 * @return New simplemc::mpi::group containing the intersection of both groups.
 */
[[nodiscard]] group group_intersection(const group& g1, const group& g2);

/**
 * @brief Create a new group that is the difference of two groups.
 *
 * @details It makes a call to `MPI_Group_difference` to create a new group containing processes that
 * are in the first group but not in the second group.
 *
 * It throws an exception if the call fails.
 *
 * @param g1 First group.
 * @param g2 Second group (to subtract from from the first).
 * @return New simplemc::mpi::group containing the difference.
 */
[[nodiscard]] group group_difference(const group& g1, const group& g2);

/**
 * @brief Compare two groups.
 *
 * @details It makes a call to `MPI_Group_compare` to compare the two groups.
 *
 * The return value is one of:
 * - `MPI_IDENT`: Groups are identical (same members in same order).
 * - `MPI_SIMILAR`: Groups have same members but in different order.
 * - `MPI_UNEQUAL`: Groups have different members.
 *
 * It throws an exception if the call fails.
 *
 * @param g1 First group.
 * @param g2 Second group.
 * @return The comparison result (`MPI_IDENT`, `MPI_SIMILAR`, or `MPI_UNEQUAL`).
 */
[[nodiscard]] int group_compare(const group& g1, const group& g2);

/**
 * @brief Equality comparison operator to check if two groups are identical.
 *
 * @details See simplemc::mpi::group_compare for details.
 *
 * @param g1 First group.
 * @param g2 Second group.
 * @return True if the groups are identical.
 */
[[nodiscard]] inline bool operator==(const group& g1, const group& g2) {
    return group_compare(g1, g2) == MPI_IDENT;
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_GROUP_HPP
