#include "../gtest-mpi-listener.hpp"

#include <gtest/gtest.h>
#include <mpi.h>
#include <simplemc/mpi.hpp>

#include <vector>

// Get groups with even and odd ranks.
auto even_odd_groups(const simplemc::mpi::communicator& comm, simplemc::mpi::resource_policy pol) {
    auto even_ranks = std::vector<int> {};
    auto odd_ranks = std::vector<int> {};
    for (int i = 0; i < comm.size(); ++i) {
        if (i % 2 == 0) {
            even_ranks.push_back(i);
        } else {
            odd_ranks.push_back(i);
        }
    }
    auto grp = comm.get_group();
    auto even_grp = grp.include(even_ranks, pol);
    auto odd_grp = grp.include(odd_ranks, pol);
    return std::make_pair(even_grp, odd_grp);
}

// Test group_free operation.
TEST(SimplemcMPI, GroupFree) {
    // free a group obtained from a communicator
    simplemc::mpi::communicator comm {};
    MPI_Group mpi_grp = comm.get_group(simplemc::mpi::resource_policy::attach);
    ASSERT_NO_THROW(simplemc::mpi::group_free(mpi_grp));
    ASSERT_EQ(mpi_grp, MPI_GROUP_NULL);

    // freeing an empty group should have no effect
    MPI_Group empty_grp = MPI_GROUP_EMPTY;
    ASSERT_NO_THROW(simplemc::mpi::group_free(empty_grp));
    ASSERT_EQ(empty_grp, MPI_GROUP_EMPTY);
}

// Test group comparison and equality operators.
TEST(SimplemcMPI, GroupCompare) {
    simplemc::mpi::communicator comm {};
    auto grp1 = comm.get_group();
    auto grp2 = comm.get_group();

    // same group should be identical
    ASSERT_EQ(grp1.compare(grp2), MPI_IDENT);
    ASSERT_EQ(simplemc::mpi::group_compare(grp1, grp2), MPI_IDENT);
    ASSERT_EQ(grp1, grp2);

    // empty group compared with another empty group should also be identical
    simplemc::mpi::group empty_grp1 {};
    simplemc::mpi::group empty_grp2 {};
    ASSERT_EQ(empty_grp1.compare(empty_grp2), MPI_IDENT);
    ASSERT_EQ(simplemc::mpi::group_compare(empty_grp1, empty_grp2), MPI_IDENT);
    ASSERT_EQ(empty_grp1, empty_grp2);

    // different groups should be unequal
    if (comm.size() > 1) {
        auto subgrp = grp1.include(std::vector<int> { 0 });
        ASSERT_EQ(grp1.compare(subgrp), MPI_UNEQUAL);
        ASSERT_EQ(simplemc::mpi::group_compare(grp1, subgrp), MPI_UNEQUAL);
        ASSERT_NE(grp1, subgrp);
    }
}

// Test default constructor creates an empty group.
TEST(SimplemcMPI, GroupDefaultConstructor) {
    simplemc::mpi::group grp {};
    ASSERT_TRUE(grp);
    ASSERT_TRUE(grp.empty());
    ASSERT_EQ(grp.size(), 0);
    ASSERT_EQ(grp.compare(simplemc::mpi::group {}), MPI_IDENT);
}

// Test constructor with MPI_GROUP_EMPTY and attach policy.
TEST(SimplemcMPI, GroupConstructorWithAttachPolicy) {
    simplemc::mpi::group grp { MPI_GROUP_EMPTY, simplemc::mpi::resource_policy::attach };
    ASSERT_TRUE(grp);

    // compare with default constructor which also wraps MPI_GROUP_EMPTY
    simplemc::mpi::group default_grp {};
    ASSERT_EQ(grp.compare(default_grp), MPI_IDENT);
}

// Test constructor with MPI_GROUP_NULL.
TEST(SimplemcMPI, GroupConstructorWithGroupNull) {
    simplemc::mpi::group null_grp { MPI_GROUP_NULL };
    ASSERT_FALSE(null_grp);
}

// Test implicit conversion to MPI_Group.
TEST(SimplemcMPI, GroupImplicitConversion) {
    simplemc::mpi::communicator comm {};
    auto grp = comm.get_group();

    // should be able to use group where MPI_Group is expected
    int size {};
    MPI_Group_size(grp, &size);
    ASSERT_EQ(size, comm.size());
}

// Test explicit conversion to bool.
TEST(SimplemcMPI, GroupExplicitBoolConversion) {
    simplemc::mpi::group null_grp { MPI_GROUP_NULL };
    ASSERT_FALSE(static_cast<bool>(null_grp));

    simplemc::mpi::group empty_grp { MPI_GROUP_EMPTY };
    ASSERT_TRUE(static_cast<bool>(empty_grp));

    simplemc::mpi::communicator comm {};
    auto comm_grp = comm.get_group();
    ASSERT_TRUE(static_cast<bool>(comm_grp));
}

// Test group size and rank.
TEST(SimplemcMPI, GroupSizeGroupRank) {
    simplemc::mpi::communicator comm {};
    auto grp = comm.get_group();

    ASSERT_EQ(grp.size(), comm.size());
    ASSERT_EQ(simplemc::mpi::group_size(grp), comm.size());
    ASSERT_EQ(grp.rank(), comm.rank());
    ASSERT_EQ(simplemc::mpi::group_rank(grp), comm.rank());
}

// Test shared_ptr semantics - copying should share the underlying group.
TEST(SimplemcMPI, GroupSharedPtrSemantics) {
    simplemc::mpi::communicator comm {};
    auto grp1 = comm.get_group();

    // make copy
    auto grp2 = grp1;

    // both should refer to the same underlying group
    ASSERT_EQ(grp1.size(), grp2.size());
    ASSERT_EQ(grp1.rank(), grp2.rank());

    // they should compare as identical
    ASSERT_EQ(grp1, grp2);
}

// Test group inclusion and exclusion.
TEST(SimplemcMPI, GroupInclGroupExcl) {
    simplemc::mpi::communicator comm {};
    auto grp = comm.get_group();

    // include no ranks should create an empty group
    auto empty_grp = grp.include(std::vector<int> {});
    ASSERT_TRUE(empty_grp.empty());
    ASSERT_EQ(empty_grp.compare(simplemc::mpi::group {}), MPI_IDENT);

    // excluding all ranks creates an empty group that compares identical to MPI_GROUP_EMPTY
    auto all_ranks = std::vector<int> {};
    for (int i = 0; i < comm.size(); ++i) {
        all_ranks.push_back(i);
    }
    auto empty_grp2 = grp.exclude(all_ranks);
    ASSERT_TRUE(empty_grp2.empty());
    ASSERT_EQ(empty_grp2.compare(simplemc::mpi::group {}), MPI_IDENT);

    // include only rank 0
    auto grp_incl_0 = grp.include(std::vector<int> { 0 });
    ASSERT_EQ(grp_incl_0.size(), 1);
    if (comm.rank() == 0) {
        ASSERT_EQ(grp_incl_0.rank(), 0);
    } else {
        ASSERT_EQ(grp_incl_0.rank(), MPI_UNDEFINED);
    }

    // exclude rank 0
    auto grp_excl_0 = grp.exclude(std::vector<int> { 0 });
    ASSERT_EQ(grp_excl_0.size(), comm.size() - 1);
    if (comm.rank() == 0) {
        ASSERT_EQ(grp_excl_0.rank(), MPI_UNDEFINED);
    } else {
        ASSERT_EQ(grp_excl_0.rank(), comm.rank() - 1);
    }

    // include/exclude even ranks using free functions
    auto even_ranks = std::vector<int> {};
    for (int i = 0; i < comm.size(); i += 2) {
        even_ranks.push_back(i);
    }
    MPI_Group even_grp = simplemc::mpi::group_incl(grp, even_ranks);
    MPI_Group odd_grp = simplemc::mpi::group_excl(grp, even_ranks);

    ASSERT_EQ(simplemc::mpi::group_size(even_grp), (comm.size() + 1) / 2);
    ASSERT_EQ(simplemc::mpi::group_size(odd_grp), comm.size() / 2);
    if (comm.rank() % 2 == 0) {
        ASSERT_EQ(simplemc::mpi::group_rank(even_grp), comm.rank() / 2);
        ASSERT_EQ(simplemc::mpi::group_rank(odd_grp), MPI_UNDEFINED);
    } else {
        ASSERT_EQ(simplemc::mpi::group_rank(odd_grp), comm.rank() / 2);
        ASSERT_EQ(simplemc::mpi::group_rank(even_grp), MPI_UNDEFINED);
    }

    simplemc::mpi::group_free(even_grp);
    simplemc::mpi::group_free(odd_grp);
}

// Test translating ranks between groups.
TEST(SimplemcMPI, GroupTranslateRanks) {
    simplemc::mpi::communicator comm {};
    auto grp = comm.get_group();

    // get subgroups with even/odd ranks
    auto [even_grp, odd_grp] = even_odd_groups(comm, simplemc::mpi::resource_policy::take_ownership);

    // translate all ranks from original to even group
    auto all_ranks = std::vector<int> {};
    for (int i = 0; i < comm.size(); ++i) {
        all_ranks.push_back(i);
    }
    auto trans_even = grp.translate_ranks(all_ranks, even_grp);
    auto trans_odd = simplemc::mpi::group_translate_ranks(grp, all_ranks, odd_grp);

    // verify translation
    for (int i = 0; i < comm.size(); ++i) {
        if (i % 2 == 0) {
            ASSERT_EQ(trans_even[static_cast<std::size_t>(i)], i / 2);
            ASSERT_EQ(trans_odd[static_cast<std::size_t>(i)], MPI_UNDEFINED);
        } else {
            ASSERT_EQ(trans_even[static_cast<std::size_t>(i)], MPI_UNDEFINED);
            ASSERT_EQ(trans_odd[static_cast<std::size_t>(i)], i / 2);
        }
    }
}

// Test group union operation.
TEST(SimplemcMPI, GroupUnion) {
    simplemc::mpi::communicator comm {};
    auto grp = comm.get_group();

    // union with itself should be identical
    auto union_self_grp = grp.union_with(grp);
    ASSERT_EQ(union_self_grp.compare(grp), MPI_IDENT);

    // union with empty group should be identical to original
    auto union_empty_grp = simplemc::mpi::group_union(simplemc::mpi::group {}, grp);
    ASSERT_EQ(simplemc::mpi::group_compare(union_empty_grp, grp), MPI_IDENT);
    simplemc::mpi::group_free(union_empty_grp);

    // get subgroups with even/odd ranks
    auto [even_grp, odd_grp] = even_odd_groups(comm, simplemc::mpi::resource_policy::take_ownership);

    // union should give us back all ranks
    auto union_grp = even_grp.union_with(odd_grp, simplemc::mpi::resource_policy::attach);
    ASSERT_EQ(union_grp.size(), comm.size());
    ASSERT_NE(union_grp.rank(), MPI_UNDEFINED);

    MPI_Group mpi_union_grp = union_grp;
    simplemc::mpi::group_free(mpi_union_grp);
}

// Test group intersection operation.
TEST(SimplemcMPI, GroupIntersection) {
    simplemc::mpi::communicator comm {};
    auto grp = comm.get_group();

    // intersection with itself should be identical
    auto intersection_self_grp = grp.intersect_with(grp);
    ASSERT_EQ(intersection_self_grp.compare(grp), MPI_IDENT);

    // intersection with empty group should be empty
    auto intersection_empty_grp = simplemc::mpi::group_intersection(simplemc::mpi::group {}, grp);
    ASSERT_EQ(simplemc::mpi::group_compare(intersection_empty_grp, MPI_GROUP_EMPTY), MPI_IDENT);
    simplemc::mpi::group_free(intersection_empty_grp);

    // create two groups with some overlap
    auto grp1_ranks = std::vector<int> {};
    auto grp2_ranks = std::vector<int> {};
    for (int i = 0; i < comm.size(); ++i) {
        if (i < comm.size() / 2 + 1) {
            grp1_ranks.push_back(i);
        }
        if (i > 0) {
            grp2_ranks.push_back(i);
        }
    }
    auto grp1 = grp.include(grp1_ranks);
    auto grp2 = grp.include(grp2_ranks);

    // intersection should give us ranks 1 to size/2
    auto intersection_grp = grp1.intersect_with(grp2, simplemc::mpi::resource_policy::attach);
    ASSERT_EQ(intersection_grp.size(), grp.size() / 2);
    auto const in_grp = grp.rank() >= 1 && grp.rank() <= grp.size() / 2;
    if (in_grp) {
        ASSERT_NE(intersection_grp.rank(), MPI_UNDEFINED);
    } else {
        ASSERT_EQ(intersection_grp.rank(), MPI_UNDEFINED);
    }

    MPI_Group mpi_intersection_grp = intersection_grp;
    simplemc::mpi::group_free(mpi_intersection_grp);
}

// Test group difference operation.
TEST(SimplemcMPI, GroupDifference) {
    simplemc::mpi::communicator comm {};
    auto grp = comm.get_group();

    // difference with empty group should be the same as original
    auto diff_empty_grp = grp.difference_with(simplemc::mpi::group {});
    ASSERT_EQ(diff_empty_grp.compare(grp), MPI_IDENT);

    // difference with itself should be empty
    auto diff_self_grp = simplemc::mpi::group_difference(grp, grp);
    ASSERT_EQ(simplemc::mpi::group_compare(diff_self_grp, MPI_GROUP_EMPTY), MPI_IDENT);
    simplemc::mpi::group_free(diff_self_grp);

    // create group with first half of ranks
    auto first_half = std::vector<int> {};
    for (int i = 0; i < comm.size() / 2; ++i) {
        first_half.push_back(i);
    }
    auto first_half_grp = grp.include(first_half);

    // difference g - first_half should give second half
    auto diff_grp = grp.difference_with(first_half_grp, simplemc::mpi::resource_policy::attach);
    ASSERT_EQ(diff_grp.size(), comm.size() - static_cast<int>(first_half.size()));
    auto const in_grp = grp.rank() >= first_half_grp.size();
    if (in_grp) {
        ASSERT_NE(diff_grp.rank(), MPI_UNDEFINED);
    } else {
        ASSERT_EQ(diff_grp.rank(), MPI_UNDEFINED);
    }

    // manually free diff_group's MPI_Group
    MPI_Group mpi_diff_grp = diff_grp;
    simplemc::mpi::group_free(mpi_diff_grp);
}

// Custom main function for MPI.
int main(int argc, char** argv) {
    // filter out Google Test arguments
    ::testing::InitGoogleTest(&argc, argv);

    // initialize MPI
    MPI_Init(&argc, &argv);

    // add object that will finalize MPI on exit; Google Test owns this pointer
    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);

    // get the event listener list.
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();

    // remove default listener: the default printer and the default XML printer
    ::testing::TestEventListener* l = listeners.Release(listeners.default_result_printer());

    // adds MPI listener; Google Test owns this pointer
    listeners.Append(new GTestMPIListener::MPIWrapperPrinter(l, MPI_COMM_WORLD));

    // run tests, then clean up and exit. RUN_ALL_TESTS() returns 0 if all tests
    // pass and 1 if some test fails.
    int result = RUN_ALL_TESTS();

    return result;
}
