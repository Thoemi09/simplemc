// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Implementation details for simplemc/mpi/group.hpp.
 */

#include <simplemc/mpi/environment.hpp>
#include <simplemc/mpi/group.hpp>
#include <simplemc/mpi/utils.hpp>

#include <mpi.h>

#include <memory>
#include <span>
#include <vector>

namespace simplemc::mpi {

void group_free(MPI_Group& grp) {
    if (grp != MPI_GROUP_NULL && grp != MPI_GROUP_EMPTY) {
        check_mpi_call(MPI_Group_free(&grp), "MPI_Group_free");
    }
}

int group_size(MPI_Group grp) {
    int sz {};
    check_mpi_call(MPI_Group_size(grp, &sz), "MPI_Group_size");
    return sz;
}

int group_rank(MPI_Group grp) {
    int rk {};
    check_mpi_call(MPI_Group_rank(grp, &rk), "MPI_Group_rank");
    return rk;
}

MPI_Group group_incl(MPI_Group grp, std::span<const int> ranks) {
    MPI_Group new_grp {};
    check_mpi_call(MPI_Group_incl(grp, static_cast<int>(ranks.size()), ranks.data(), &new_grp), "MPI_Group_incl");
    return new_grp;
}

MPI_Group group_excl(MPI_Group grp, std::span<const int> ranks) {
    MPI_Group new_grp {};
    check_mpi_call(MPI_Group_excl(grp, static_cast<int>(ranks.size()), ranks.data(), &new_grp), "MPI_Group_excl");
    return new_grp;
}

std::vector<int> group_translate_ranks(MPI_Group src_grp, std::span<const int> ranks, MPI_Group dest_grp) {
    auto translated = std::vector<int>(ranks.size());
    check_mpi_call(
        MPI_Group_translate_ranks(src_grp, static_cast<int>(ranks.size()), ranks.data(), dest_grp, translated.data()),
        "MPI_Group_translate_ranks");
    return translated;
}

MPI_Group group_union(MPI_Group grp1, MPI_Group grp2) {
    MPI_Group new_grp {};
    check_mpi_call(MPI_Group_union(grp1, grp2, &new_grp), "MPI_Group_union");
    return new_grp;
}

MPI_Group group_intersection(MPI_Group grp1, MPI_Group grp2) {
    MPI_Group new_grp {};
    check_mpi_call(MPI_Group_intersection(grp1, grp2, &new_grp), "MPI_Group_intersection");
    return new_grp;
}

MPI_Group group_difference(MPI_Group grp1, MPI_Group grp2) {
    MPI_Group new_grp {};
    check_mpi_call(MPI_Group_difference(grp1, grp2, &new_grp), "MPI_Group_difference");
    return new_grp;
}

int group_compare(MPI_Group grp1, MPI_Group grp2) {
    int result {};
    check_mpi_call(MPI_Group_compare(grp1, grp2, &result), "MPI_Group_compare");
    return result;
}

group::group(MPI_Group grp, resource_policy pol) {
    if (pol == resource_policy::take_ownership) {
        grp_ = std::shared_ptr<MPI_Group> { new MPI_Group(grp), group_deleter {} };
    } else {
        grp_ = std::make_shared<MPI_Group>(grp);
    }
}

void group::group_deleter::operator()(MPI_Group* g) const noexcept {
    if (!finalized() && g && *g != MPI_GROUP_NULL && *g != MPI_GROUP_EMPTY) {
        MPI_Group_free(g);
    }
    delete g;
}

} // namespace simplemc::mpi
