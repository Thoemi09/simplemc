/**
 * @file
 * @brief Implementation details for simplemc/mpi/group.hpp.
 */

#include <simplemc/mpi/environment.hpp>
#include <simplemc/mpi/group.hpp>
#include <simplemc/mpi/utils.hpp>

#include <mpi.h>

#include <memory>

namespace simplemc::mpi {

group::group() : grp_(new MPI_Group(MPI_GROUP_EMPTY)) {}

group::group(MPI_Group grp, resource_policy pol) {
    if (pol == resource_policy::take_ownership) {
        grp_ = std::shared_ptr<MPI_Group> { new MPI_Group(grp), group_deleter {} };
    } else {
        grp_ = std::make_shared<MPI_Group>(grp);
    }
}

int group::rank() const {
    int my_rank {};
    check_mpi_call(MPI_Group_rank(*grp_, &my_rank), "MPI_Group_rank");
    return my_rank;
}

int group::size() const {
    int my_size {};
    check_mpi_call(MPI_Group_size(*grp_, &my_size), "MPI_Group_size");
    return my_size;
}

void group::group_deleter::operator()(MPI_Group* g) const {
    // only free if valid and not a predefined group
    if (!finalized() && g && *g != MPI_GROUP_NULL && *g != MPI_GROUP_EMPTY) {
        MPI_Group_free(g);
    }
    delete g;
}

void group_free(MPI_Group& grp) {
    if (grp != MPI_GROUP_NULL && grp != MPI_GROUP_EMPTY) {
        check_mpi_call(MPI_Group_free(&grp), "MPI_Group_free");
    }
}

group group_union(const group& g1, const group& g2, resource_policy pol) {
    MPI_Group new_grp {};
    check_mpi_call(MPI_Group_union(g1, g2, &new_grp), "MPI_Group_union");
    return group { new_grp, pol };
}

group group_intersection(const group& g1, const group& g2, resource_policy pol) {
    MPI_Group new_grp {};
    check_mpi_call(MPI_Group_intersection(g1, g2, &new_grp), "MPI_Group_intersection");
    return group { new_grp, pol };
}

group group_difference(const group& g1, const group& g2, resource_policy pol) {
    MPI_Group new_grp {};
    check_mpi_call(MPI_Group_difference(g1, g2, &new_grp), "MPI_Group_difference");
    return group { new_grp, pol };
}

int group_compare(const group& g1, const group& g2) {
    int result {};
    check_mpi_call(MPI_Group_compare(g1, g2, &result), "MPI_Group_compare");
    return result;
}

} // namespace simplemc::mpi
