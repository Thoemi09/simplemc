// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/ranges.h>
#include <simplemc/mpi.hpp>

#include <array>
#include <string>

// Helper function to print the original rank, group size and group rank.
void print_group_info(int rank, const simplemc::mpi::group& grp) {
    auto rank_str = (grp.rank() == MPI_UNDEFINED ? "UNDEFINED" : std::to_string(grp.rank()));
    fmt::println("Rank {}: Group size = {}, Group rank = {}", rank, grp.size(), rank_str);
}

// Helper function to print the original rank, communicator size and communicator rank.
void print_comm_info(int rank, const simplemc::mpi::communicator& comm) {
    if (comm) {
        fmt::println("Rank {}: Communicator size = {}, Communicator rank = {}", rank, comm.size(), comm.rank());
    } else {
        fmt::println("Rank {}: Communicator is MPI_COMM_NULL", rank);
    }
}

// Helper function to compare two groups or communicators.
auto print_comparison(const auto& obj1, const auto& obj2) {
    const int cmp = obj1.compare(obj2);
    switch (cmp) {
        case MPI_IDENT: return "IDENTICAL";
        case MPI_CONGRUENT: return "CONGRUENT";
        case MPI_SIMILAR: return "SIMILAR";
        case MPI_UNEQUAL: return "UNEQUAL";
        default: return "UNKNOWN";
    }
}

int main(int argc, char** argv) {
    // initialize MPI environment and communicator
    simplemc::mpi::environment env { argc, argv };
    simplemc::mpi::communicator comm {};
    const int rank = comm.rank();

    // get the group associated with MPI_COMM_WORLD
    auto grp = comm.get_group();
    print_group_info(rank, grp);

    // include: create a subgroup with only ranks 0 and 2 (even ranks)
    std::array<int, 2> even_ranks { 0, 2 };
    auto even_grp = grp.include(even_ranks);
    print_group_info(rank, even_grp);

    // exclude: create a subgroup by excluding ranks 0 and 2 (odd ranks)
    auto odd_grp = grp.exclude(even_ranks);
    print_group_info(rank, odd_grp);

    // union: combine even and odd groups (similar to original group: same ranks but different order)
    auto union_grp = even_grp.union_with(odd_grp);
    print_group_info(rank, union_grp);
    if (rank == 0) {
        fmt::println("Union group vs original group: {}", print_comparison(union_grp, grp));
    }

    // difference: processes in original but not in even group (identical to odd group)
    auto diff_grp = grp.difference_with(even_grp);
    print_group_info(rank, diff_grp);
    if (rank == 0) {
        fmt::println("Difference group vs odd group: {}", print_comparison(diff_grp, odd_grp));
    }

    // intersection: common processes in even and odd group (identical to MPI_GROUP_EMPTY)
    auto intersect_grp = even_grp.intersect_with(odd_grp);
    print_group_info(rank, intersect_grp);
    if (rank == 0) {
        auto empty_grp = simplemc::mpi::group { MPI_GROUP_EMPTY };
        fmt::println("Intersection group vs MPI_GROUP_EMPTY: {}", print_comparison(intersect_grp, empty_grp));
    }

    // duplicate the communicator (congruent to original communicator)
    auto dup_comm = comm.duplicate();
    print_comm_info(rank, dup_comm);
    if (rank == 0) {
        fmt::println("Duplicated communicator vs original communicator: {}", print_comparison(dup_comm, comm));
    }

    // split communicator by even/odd rank
    const int color = comm.rank() % 2;
    auto split_comm = comm.split(color);
    print_comm_info(rank, split_comm);

    // create a communicator from the even group (congruent to split communicator for even ranks)
    auto even_comm = comm.create(even_grp);
    print_comm_info(rank, even_comm);
    if (rank == 0) {
        fmt::println("Even group communicator vs split communicator: {}", print_comparison(even_comm, split_comm));
    }
}
