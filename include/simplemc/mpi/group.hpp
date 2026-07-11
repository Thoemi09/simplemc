// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief RAII wrapper for `MPI_Group` objects and other essential MPI functions involving groups.
 */

#ifndef SIMPLEMC_MPI_GROUP_HPP
#define SIMPLEMC_MPI_GROUP_HPP

#include <simplemc/mpi/mpi_fwd.hpp>
#include <simplemc/mpi/utils.hpp>
#include <simplemc/utils/ranges.hpp>

#include <mpi.h>

#include <concepts>
#include <memory>
#include <span>
#include <vector>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-essentials-groups
 * @{
 */

/**
 * @brief Free an `MPI_Group` object.
 *
 * @details It makes a call to `MPI_Group_free` to free the group, unless it is `MPI_GROUP_NULL` or
 * `MPI_GROUP_EMPTY`, in which case it does nothing.
 *
 * @param grp Reference to the group to free.
 */
void group_free(MPI_Group& grp);

/**
 * @brief Get the size of an MPI group by calling `MPI_Group_size`.
 *
 * @param grp Group to query.
 * @return Number of processes in the group.
 */
[[nodiscard]] int group_size(MPI_Group grp);

/**
 * @brief Get the rank of the calling process in an MPI group by calling `MPI_Group_rank`.
 *
 * @param grp Group to query.
 * @return Rank of the calling process in the group, or `MPI_UNDEFINED` if it is not a member.
 */
[[nodiscard]] int group_rank(MPI_Group grp);

/**
 * @brief Create a new MPI group containing only the specified ranks.
 *
 * @details It makes a call to `MPI_Group_incl` to create a new group containing only the processes
 * with the specified ranks from the input group.
 *
 * @note The caller is responsible for freeing the returned `MPI_Group` using
 * simplemc::mpi::group_free or `MPI_Group_free()`.
 *
 * @param grp Input group.
 * @param ranks Span of ranks to include in the new group.
 * @return New group object containing only the specified ranks.
 */
[[nodiscard]] MPI_Group group_incl(MPI_Group grp, std::span<const int> ranks);

/**
 * @brief Create a new MPI group excluding the specified ranks.
 *
 * @details It makes a call to `MPI_Group_excl` to create a new group containing all processes from
 * the input group except those with the specified ranks.
 *
 * @note The caller is responsible for freeing the returned `MPI_Group` using
 * simplemc::mpi::group_free or `MPI_Group_free()`.
 *
 * @param grp Input group.
 * @param ranks Span of ranks to exclude from the new group.
 * @return New group object excluding the specified ranks.
 */
[[nodiscard]] MPI_Group group_excl(MPI_Group grp, std::span<const int> ranks);

/**
 * @brief Translate ranks from one MPI group to another.
 *
 * @details It makes a call to `MPI_Group_translate_ranks` to translate the specified ranks from
 * the source group to the corresponding ranks in the destination group.
 *
 * For ranks that are not in the destination group, `MPI_UNDEFINED` is returned.
 *
 * @param src_grp Source group containing the ranks to translate.
 * @param ranks Span of ranks in the source group to translate.
 * @param dest_grp Destination group to translate to.
 * @return Vector of translated ranks in the destination group.
 */
[[nodiscard]] std::vector<int> group_translate_ranks(MPI_Group src_grp, std::span<const int> ranks, MPI_Group dest_grp);

/**
 * @brief Create a new MPI group that is the union of two groups.
 *
 * @details It makes a call to `MPI_Group_union` to create a new group containing all processes that
 * are in the first group or the second group (or both).
 *
 * @note The caller is responsible for freeing the returned `MPI_Group` using
 * simplemc::mpi::group_free or `MPI_Group_free()`.
 *
 * @param grp1 First group.
 * @param grp2 Second group.
 * @return New group object containing the union of both groups.
 */
[[nodiscard]] MPI_Group group_union(MPI_Group grp1, MPI_Group grp2);

/**
 * @brief Create a new MPI group that is the intersection of two groups.
 *
 * @details It makes a call to `MPI_Group_intersection` to create a new group containing only
 * processes that are in both groups.
 *
 * @note The caller is responsible for freeing the returned `MPI_Group` using
 * simplemc::mpi::group_free or `MPI_Group_free()`.
 *
 * @param grp1 First group.
 * @param grp2 Second group.
 * @return New group object containing the intersection of both groups.
 */
[[nodiscard]] MPI_Group group_intersection(MPI_Group grp1, MPI_Group grp2);

/**
 * @brief Create a new MPI group that is the difference of two groups.
 *
 * @details It makes a call to `MPI_Group_difference` to create a new group containing processes
 * that are in the first group but not in the second group.
 *
 * @note The caller is responsible for freeing the returned `MPI_Group` using
 * simplemc::mpi::group_free or `MPI_Group_free()`.
 *
 * @param grp1 First group.
 * @param grp2 Second group (to subtract from the first).
 * @return New group object containing the difference.
 */
[[nodiscard]] MPI_Group group_difference(MPI_Group grp1, MPI_Group grp2);

/**
 * @brief Compare two MPI groups by calling `MPI_Group_compare`.
 *
 * @details The return value is one of:
 * - `MPI_IDENT`: Groups are identical (same members in the same order).
 * - `MPI_SIMILAR`: Groups have the same members but in different order.
 * - `MPI_UNEQUAL`: Groups have different members.
 *
 * @param grp1 First group.
 * @param grp2 Second group.
 * @return The comparison result (`MPI_IDENT`, `MPI_SIMILAR`, or `MPI_UNEQUAL`).
 */
[[nodiscard]] int group_compare(MPI_Group grp1, MPI_Group grp2);

/**
 * @brief RAII wrapper for `MPI_Group` objects.
 *
 * @details This class provides a thin C++ wrapper around `MPI_Group` objects. MPI groups represent
 * ordered sets of processes and can be used to create new communicators.
 *
 * The wrapped `MPI_Group` is managed using a `std::shared_ptr` with a custom deleter providing
 * automatic resource management similar to <a href="https://www.boost.org/doc/libs/latest/doc/html/
 * mpi.html">Boost.MPI</a>. The behavior of the deleter depends on the simplemc::mpi::resource_policy
 * specified during construction:
 * - `take_ownership`: The wrapper is responsible for managing the lifetime of the MPI group, i.e. it
 * calls `MPI_Group_free` when the last reference is destroyed (unless it is `MPI_GROUP_NULL` or
 * `MPI_GROUP_EMPTY`).
 * - `attach`: The lifetime of the MPI group is managed somewhere else and the wrapper just uses it.
 *
 * Implicit conversions to `MPI_Group` are allowed for interoperability with MPI C functions.
 *
 * See @ref tut_mpi_5 for examples.
 */
class group {
public:
    /**
     * @brief Default construct an empty group.
     *
     * @details It creates a group that wraps `MPI_GROUP_EMPTY`. This represents an empty group with
     * no processes.
     */
    group() : grp_(new MPI_Group(MPI_GROUP_EMPTY)) {}

    /**
     * @brief Construct a group from an existing `MPI_Group` with a given
     * simplemc::mpi::resource_policy.
     *
     * @details If the resource policy is set to `take_ownership` (the default), the group takes
     * ownership of the provided `MPI_Group` and will free it when the last reference is destroyed.
     *
     * If the policy is `attach`, the group simply attaches to the provided `MPI_Group` without taking
     * ownership, and the caller remains responsible for freeing it.
     *
     * @param grp `MPI_Group` object to wrap.
     * @param pol simplemc::mpi::resource_policy indicating ownership semantics.
     */
    explicit group(MPI_Group grp, resource_policy pol = resource_policy::take_ownership);

    /**
     * @brief Get the rank of the calling process in the group by calling simplemc::mpi::group_rank.
     *
     * @return Rank of the calling process in this group, or `MPI_UNDEFINED` if it is not a member.
     */
    [[nodiscard]] int rank() const { return group_rank(*grp_); }

    /**
     * @brief Get the size of the group by calling simplemc::mpi::group_size.
     *
     * @return Number of processes in the group.
     */
    [[nodiscard]] int size() const { return group_size(*grp_); }

    /**
     * @brief Check if the group is empty.
     *
     * @return True if the group has no members, i.e. if `size() == 0`.
     */
    [[nodiscard]] bool empty() const { return size() == 0; }

    /**
     * @brief Create a new group containing only the specified ranks.
     *
     * @details It calls simplemc::mpi::group_incl and wraps the result.
     *
     * @tparam R Range type whose value type is an integral type.
     * @param ranks Range of ranks to include in the new group.
     * @param pol simplemc::mpi::resource_policy indicating ownership semantics for the new group.
     * @return New group containing only the specified ranks.
     */
    template <simplemc::ranges::input_range R>
        requires std::integral<simplemc::ranges::range_value_t<R>>
    [[nodiscard]] group include(const R& ranks, resource_policy pol = resource_policy::take_ownership) const {
        auto ranks_vec = std::vector<int> { simplemc::ranges::begin(ranks), simplemc::ranges::end(ranks) };
        return group { group_incl(*grp_, ranks_vec), pol };
    }

    /**
     * @brief Create a new group excluding the specified ranks.
     *
     * @details It calls simplemc::mpi::group_excl and wraps the result.
     *
     * @tparam R Range type whose value type is an integral type.
     * @param ranks Range of ranks to exclude from the new group.
     * @param pol simplemc::mpi::resource_policy indicating ownership semantics for the new group.
     * @return New group excluding the specified ranks.
     */
    template <simplemc::ranges::input_range R>
        requires std::integral<simplemc::ranges::range_value_t<R>>
    [[nodiscard]] group exclude(const R& ranks, resource_policy pol = resource_policy::take_ownership) const {
        auto ranks_vec = std::vector<int> { simplemc::ranges::begin(ranks), simplemc::ranges::end(ranks) };
        return group { group_excl(*grp_, ranks_vec), pol };
    }

    /**
     * @brief Translate ranks from this group to another group.
     *
     * @details It calls simplemc::mpi::group_translate_ranks with `this` as the source group.
     *
     * @tparam R Range type whose value type is an integral type.
     * @param ranks Ranks in this group to translate.
     * @param other Destination group to translate to.
     * @return Vector of translated ranks in the other group.
     */
    template <simplemc::ranges::input_range R>
        requires std::integral<simplemc::ranges::range_value_t<R>>
    [[nodiscard]] std::vector<int> translate_ranks(const R& ranks, const group& other) const {
        auto ranks_vec = std::vector<int> { simplemc::ranges::begin(ranks), simplemc::ranges::end(ranks) };
        return group_translate_ranks(*grp_, ranks_vec, other);
    }

    /**
     * @brief Create a new group that is the union of this group and another group.
     *
     * @details It calls simplemc::mpi::group_union and wraps the result.
     *
     * @param other Other group.
     * @param pol simplemc::mpi::resource_policy indicating ownership semantics for the new group.
     * @return New group containing the union of both groups.
     */
    [[nodiscard]] group union_with(const group& other, resource_policy pol = resource_policy::take_ownership) const {
        return group { group_union(*grp_, other), pol };
    }

    /**
     * @brief Create a new group that is the intersection of this group and another group.
     *
     * @details It calls simplemc::mpi::group_intersection and wraps the result.
     *
     * @param other Other group.
     * @param pol simplemc::mpi::resource_policy indicating ownership semantics for the new group.
     * @return New group containing the intersection of both groups.
     */
    [[nodiscard]] group intersect_with(
        const group& other, resource_policy pol = resource_policy::take_ownership) const {
        return group { group_intersection(*grp_, other), pol };
    }

    /**
     * @brief Create a new group that is the difference of this group and another group.
     *
     * @details It calls simplemc::mpi::group_difference and wraps the result. The resulting group
     * only contains processes that are in `this` group but not in `other`.
     *
     * @param other Group to subtract from this group.
     * @param pol simplemc::mpi::resource_policy indicating ownership semantics for the new group.
     * @return New group containing the difference.
     */
    [[nodiscard]] group difference_with(
        const group& other, resource_policy pol = resource_policy::take_ownership) const {
        return group { group_difference(*grp_, other), pol };
    }

    /**
     * @brief Compare this group with another group by calling simplemc::mpi::group_compare.
     *
     * @param other Other group to compare with.
     * @return The comparison result (`MPI_IDENT`, `MPI_SIMILAR`, or `MPI_UNEQUAL`).
     */
    [[nodiscard]] int compare(const group& other) const { return group_compare(*grp_, other); }

    /**
     * @brief Implicit conversion to the underlying `MPI_Group` object.
     *
     * @return Underlying `MPI_Group` or `MPI_GROUP_NULL` if no MPI group is wrapped.
     */
    operator MPI_Group() const { return grp_ ? *grp_ : MPI_GROUP_NULL; }

    /**
     * @brief Explicit conversion to bool to check if the group is valid.
     *
     * @return True if the group is not `MPI_GROUP_NULL`.
     */
    [[nodiscard]] explicit operator bool() const { return grp_ && *grp_ != MPI_GROUP_NULL; }

    /**
     * @brief Equality comparison operator to check if two groups are identical.
     *
     * @details See also simplemc::mpi::group_compare.
     *
     * @param g1 First group.
     * @param g2 Second group.
     * @return True if the groups are identical (`MPI_IDENT`).
     */
    [[nodiscard]] friend bool operator==(const group& g1, const group& g2) { return g1.compare(g2) == MPI_IDENT; }

private:
    // Custom deleter that makes a protected call to MPI_Group_free.
    struct group_deleter {
        void operator()(MPI_Group* g) const noexcept;
    };

private:
    std::shared_ptr<MPI_Group> grp_;
};

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_GROUP_HPP
