#include <gtest/gtest.h>
#include <mpi.h>
#include <simplemc/mpi.hpp>

#include <vector>

// Test default constructor creates an empty group.
TEST(SimplemcMPIGroup, DefaultConstructor) {
    simplemc::mpi::group g {};
    ASSERT_TRUE(g);
    ASSERT_TRUE(g.empty());
    ASSERT_EQ(g.size(), 0);
    ASSERT_EQ(simplemc::mpi::group_compare(g, simplemc::mpi::group {}), MPI_IDENT);
}

// Test constructor with MPI_GROUP_EMPTY and attach policy.
TEST(SimplemcMPIGroup, ConstructorWithAttachPolicy) {
    simplemc::mpi::group g { MPI_GROUP_EMPTY, simplemc::mpi::resource_policy::attach };
    ASSERT_TRUE(g);

    // compare with default constructor which also wraps MPI_GROUP_EMPTY
    simplemc::mpi::group default_g {};
    ASSERT_EQ(simplemc::mpi::group_compare(g, default_g), MPI_IDENT);
}

// Test constructor with MPI_GROUP_NULL.
TEST(SimplemcMPIGroup, ConstructorWithGroupNull) {
    simplemc::mpi::group null_group { MPI_GROUP_NULL };
    ASSERT_FALSE(null_group);
}

// Test implicit conversion to MPI_Group.
TEST(SimplemcMPIGroup, ImplicitConversion) {
    simplemc::mpi::communicator comm {};
    auto g = comm.get_group();

    // should be able to use group where MPI_Group is expected
    int size {};
    MPI_Group_size(g, &size);
    ASSERT_EQ(size, comm.size());
}

// Test include operation.
TEST(SimplemcMPIGroup, Include) {
    simplemc::mpi::communicator comm {};
    auto g = comm.get_group();

    // include only rank 0 and use resource_policy::attach
    auto ranks = std::vector<int> { 0 };
    auto subgroup = g.include(ranks, simplemc::mpi::resource_policy::attach);

    ASSERT_EQ(subgroup.size(), 1);
    if (comm.rank() == 0) {
        ASSERT_EQ(subgroup.rank(), 0);
    } else {
        ASSERT_EQ(subgroup.rank(), MPI_UNDEFINED);
    }

    // manually free subgroup's MPI_Group
    MPI_Group mpi_group = subgroup;
    simplemc::mpi::group_free(mpi_group);

    // include no ranks should create an empty group
    auto empty_subgroup = g.include(std::vector<int> {});
    ASSERT_TRUE(empty_subgroup.empty());
    ASSERT_EQ(simplemc::mpi::group_compare(empty_subgroup, simplemc::mpi::group {}), MPI_IDENT);
}

// Test exclude operation.
TEST(SimplemcMPIGroup, Exclude) {
    simplemc::mpi::communicator comm {};
    auto g = comm.get_group();

    // exclude rank 0 and use resource_policy::attach
    auto ranks = std::vector<int> { 0 };
    auto subgroup = g.exclude(ranks, simplemc::mpi::resource_policy::attach);

    ASSERT_EQ(subgroup.size(), comm.size() - 1);
    if (comm.rank() == 0) {
        ASSERT_EQ(subgroup.rank(), MPI_UNDEFINED);
    } else {
        // ranks shift down after excluding rank 0
        ASSERT_EQ(subgroup.rank(), comm.rank() - 1);
    }

    // manually free subgroup's MPI_Group
    MPI_Group mpi_group = subgroup;
    simplemc::mpi::group_free(mpi_group);

    // excluding all ranks creates an empty group that compares identical to MPI_GROUP_EMPTY
    auto all_ranks = std::vector<int> {};
    for (int i = 0; i < comm.size(); ++i) {
        all_ranks.push_back(i);
    }
    auto empty_excluded = g.exclude(all_ranks);
    ASSERT_TRUE(empty_excluded.empty());
    ASSERT_EQ(simplemc::mpi::group_compare(empty_excluded, simplemc::mpi::group {}), MPI_IDENT);
}

// Test include with even ranks and resource_policy::attach.
TEST(SimplemcMPIGroup, IncludeEvenRanks) {
    simplemc::mpi::communicator comm {};
    auto g = comm.get_group();

    // include only even ranks
    auto even_ranks = std::vector<int> {};
    for (int i = 0; i < comm.size(); i += 2) {
        even_ranks.push_back(i);
    }
    auto even_group = g.include(even_ranks, simplemc::mpi::resource_policy::attach);

    int const expected_size = (comm.size() + 1) / 2;
    ASSERT_EQ(even_group.size(), expected_size);
    if (comm.rank() % 2 == 0) {
        ASSERT_EQ(even_group.rank(), comm.rank() / 2);
    } else {
        ASSERT_EQ(even_group.rank(), MPI_UNDEFINED);
    }

    // manually free even_group's MPI_Group
    MPI_Group mpi_even_group = even_group;
    simplemc::mpi::group_free(mpi_even_group);
}

// Test union operation.
TEST(SimplemcMPIGroup, Union) {
    simplemc::mpi::communicator comm {};
    auto g = comm.get_group();

    // create two groups: even and odd ranks
    auto even_ranks = std::vector<int> {};
    auto odd_ranks = std::vector<int> {};
    for (int i = 0; i < comm.size(); ++i) {
        if (i % 2 == 0) {
            even_ranks.push_back(i);
        } else {
            odd_ranks.push_back(i);
        }
    }
    auto even_group = g.include(even_ranks);
    auto odd_group = g.include(odd_ranks);

    // union should give us back all ranks
    auto union_group = simplemc::mpi::group_union(even_group, odd_group, simplemc::mpi::resource_policy::attach);
    ASSERT_EQ(union_group.size(), comm.size());

    // manually free union_group's MPI_Group
    MPI_Group mpi_union_group = union_group;
    simplemc::mpi::group_free(mpi_union_group);

    // union with itself should be identical
    auto union_self = simplemc::mpi::group_union(even_group, even_group);
    ASSERT_EQ(simplemc::mpi::group_compare(union_self, even_group), MPI_IDENT);

    // union with empty group should be identical to original
    auto union_with_empty = simplemc::mpi::group_union(even_group, simplemc::mpi::group {});
    ASSERT_EQ(simplemc::mpi::group_compare(union_with_empty, even_group), MPI_IDENT);
}

// Test intersection operation.
TEST(SimplemcMPIGroup, Intersection) {
    simplemc::mpi::communicator comm {};
    auto g = comm.get_group();

    // create two groups with some overlap
    auto group1_ranks = std::vector<int> {};
    auto group2_ranks = std::vector<int> {};
    for (int i = 0; i < comm.size(); ++i) {
        if (i < comm.size() / 2 + 1) {
            group1_ranks.push_back(i);
        }
        if (i > 0) {
            group2_ranks.push_back(i);
        }
    }
    auto group1 = g.include(group1_ranks);
    auto group2 = g.include(group2_ranks);

    // intersection should give us ranks 1 to size/2
    auto intersection_group = simplemc::mpi::group_intersection(group1, group2, simplemc::mpi::resource_policy::attach);

    // calculate expected intersection size
    int expected_size = 0;
    for (int i = 0; i < comm.size(); ++i) {
        bool const in_group1 = (i < comm.size() / 2 + 1);
        bool const in_group2 = (i > 0);
        if (in_group1 && in_group2) {
            ++expected_size;
        }
    }
    ASSERT_EQ(intersection_group.size(), expected_size);

    // manually free intersection_group's MPI_Group
    MPI_Group mpi_intersection_group = intersection_group;
    simplemc::mpi::group_free(mpi_intersection_group);

    // intersection with itself should be identical
    auto intersection_self = simplemc::mpi::group_intersection(group1, group1);
    ASSERT_EQ(simplemc::mpi::group_compare(intersection_self, group1), MPI_IDENT);

    // intersection with empty group should be empty
    auto intersection_with_empty = simplemc::mpi::group_intersection(group1, simplemc::mpi::group {});
    ASSERT_TRUE(intersection_with_empty.empty());
}

// Test difference operation.
TEST(SimplemcMPIGroup, Difference) {
    simplemc::mpi::communicator comm {};
    auto g = comm.get_group();

    // create group with first half of ranks
    auto first_half = std::vector<int> {};
    for (int i = 0; i < (comm.size() + 1) / 2; ++i) {
        first_half.push_back(i);
    }
    auto first_half_group = g.include(first_half);

    // difference g - first_half should give second half
    auto diff_group = simplemc::mpi::group_difference(g, first_half_group, simplemc::mpi::resource_policy::attach);
    ASSERT_EQ(diff_group.size(), comm.size() - static_cast<int>(first_half.size()));

    // manually free diff_group's MPI_Group
    MPI_Group mpi_diff_group = diff_group;
    simplemc::mpi::group_free(mpi_diff_group);

    // difference with empty group should be the same as original
    auto diff_with_empty = simplemc::mpi::group_difference(g, simplemc::mpi::group {});
    ASSERT_EQ(simplemc::mpi::group_compare(diff_with_empty, g), MPI_IDENT);

    // difference with itself should be empty
    auto diff_self = simplemc::mpi::group_difference(first_half_group, first_half_group);
    ASSERT_TRUE(diff_self.empty());
}

// Test compare operation.
TEST(SimplemcMPIGroup, Compare) {
    simplemc::mpi::communicator comm {};
    auto g1 = comm.get_group();
    auto g2 = comm.get_group();

    // same group should be identical
    auto result = simplemc::mpi::group_compare(g1, g2);
    ASSERT_EQ(result, MPI_IDENT);

    // empty group compared with another empty group should also be identical
    simplemc::mpi::group empty1 {};
    simplemc::mpi::group empty2 {};
    ASSERT_EQ(simplemc::mpi::group_compare(empty1, empty2), MPI_IDENT);

    // different groups should be unequal
    if (comm.size() > 1) {
        auto subgroup = g1.include(std::vector<int> { 0 });
        ASSERT_EQ(simplemc::mpi::group_compare(g1, subgroup), MPI_UNEQUAL);
    }
}

// Test translate_ranks operation.
TEST(SimplemcMPIGroup, TranslateRanks) {
    simplemc::mpi::communicator comm {};
    auto g = comm.get_group();

    // create a subgroup with even ranks
    auto even_ranks = std::vector<int> {};
    for (int i = 0; i < comm.size(); i += 2) {
        even_ranks.push_back(i);
    }
    auto even_group = g.include(even_ranks);

    // translate all ranks from g to even_group
    auto all_ranks = std::vector<int> {};
    for (int i = 0; i < comm.size(); ++i) {
        all_ranks.push_back(i);
    }
    auto translated = g.translate_ranks(all_ranks, even_group);

    // verify translation
    for (int i = 0; i < comm.size(); ++i) {
        if (i % 2 == 0) {
            ASSERT_EQ(translated[i], i / 2);
        } else {
            ASSERT_EQ(translated[i], MPI_UNDEFINED);
        }
    }
}

// Test equality operators.
TEST(SimplemcMPIGroup, EqualityOperators) {
    simplemc::mpi::communicator comm {};
    auto g1 = comm.get_group();
    auto g2 = comm.get_group();

    // same group should be equal
    ASSERT_TRUE(g1 == g2);
    ASSERT_FALSE(g1 != g2);

    // different groups should not be equal
    if (comm.size() > 1) {
        auto subgroup = g1.include(std::vector<int> { 0 });
        ASSERT_FALSE(g1 == subgroup);
        ASSERT_TRUE(g1 != subgroup);
    }
}

// Test shared_ptr semantics - copying should share the underlying group.
TEST(SimplemcMPIGroup, SharedPtrSemantics) {
    simplemc::mpi::communicator comm {};
    auto g1 = comm.get_group();

    // make copy
    auto g2 = g1;

    // both should refer to the same underlying group
    ASSERT_EQ(g1.size(), g2.size());
    ASSERT_EQ(g1.rank(), g2.rank());

    // they should compare as identical
    ASSERT_TRUE(g1 == g2);
}

// Test group_free operation.
TEST(SimplemcMPIGroup, GroupFree) {
    simplemc::mpi::communicator comm {};

    // get a group we can free
    MPI_Group mpi_group = comm.get_group(simplemc::mpi::resource_policy::attach);

    // free should not throw
    ASSERT_NO_THROW(simplemc::mpi::group_free(mpi_group));

    // after free, group should be MPI_GROUP_NULL
    ASSERT_EQ(mpi_group, MPI_GROUP_NULL);
}

// Test group_free on MPI_GROUP_EMPTY does nothing (predefined group).
TEST(SimplemcMPIGroup, GroupFreeGroupEmpty) {
    MPI_Group mpi_group = MPI_GROUP_EMPTY;

    // free on MPI_GROUP_EMPTY should not actually free it
    ASSERT_NO_THROW(simplemc::mpi::group_free(mpi_group));

    // MPI_GROUP_EMPTY is predefined, group_free() should have no effect
    ASSERT_EQ(mpi_group, MPI_GROUP_EMPTY);
}

// Test group_free on included group.
TEST(SimplemcMPIGroup, GroupFreeIncludedGroup) {
    simplemc::mpi::communicator comm {};
    auto g = comm.get_group();

    // manually create a new group to test free
    MPI_Group new_group = g.include(std::vector<int> { 0 }, simplemc::mpi::resource_policy::attach);

    // free should not throw
    ASSERT_NO_THROW(simplemc::mpi::group_free(new_group));

    // after free, group should be MPI_GROUP_NULL
    ASSERT_EQ(new_group, MPI_GROUP_NULL);
}

// Test creating multiple groups and comparing them.
TEST(SimplemcMPIGroup, MultipleOperations) {
    simplemc::mpi::communicator comm {};
    auto g = comm.get_group();

    // create two subgroups
    auto even_ranks = std::vector<int> {};
    auto odd_ranks = std::vector<int> {};
    for (int i = 0; i < comm.size(); ++i) {
        if (i % 2 == 0) {
            even_ranks.push_back(i);
        } else {
            odd_ranks.push_back(i);
        }
    }
    auto even_group = g.include(even_ranks);
    auto odd_group = g.include(odd_ranks);

    // both subgroups should be unequal to original
    ASSERT_EQ(simplemc::mpi::group_compare(g, even_group), MPI_UNEQUAL);
    ASSERT_EQ(simplemc::mpi::group_compare(g, odd_group), MPI_UNEQUAL);
    ASSERT_EQ(simplemc::mpi::group_compare(even_group, odd_group), MPI_UNEQUAL);

    // union should give back original
    auto union_group = simplemc::mpi::group_union(even_group, odd_group);
    ASSERT_EQ(simplemc::mpi::group_compare(g, union_group), MPI_SIMILAR);
}

// Test chain of operations: include -> exclude -> intersection.
TEST(SimplemcMPIGroup, ChainedOperations) {
    simplemc::mpi::communicator comm {};
    auto g = comm.get_group();

    // include even ranks
    auto even_ranks = std::vector<int> {};
    for (int i = 0; i < comm.size(); i += 2) {
        even_ranks.push_back(i);
    }
    auto even_group = g.include(even_ranks);
    ASSERT_EQ(even_group.size(), (comm.size() + 1) / 2);

    // exclude rank 0 from original to get all but rank 0
    auto without_zero = g.exclude(std::vector<int> { 0 });
    ASSERT_EQ(without_zero.size(), comm.size() - 1);

    // intersection of even_group and without_zero should give even ranks except 0
    // even ranks: 0, 2, 4, ... without rank 0: 2, 4, ...
    // so expected_size = (number of even ranks) - 1, but only if size > 1
    auto intersection = simplemc::mpi::group_intersection(even_group, without_zero);
    int const expected_size = (comm.size() > 2) ? (static_cast<int>(even_ranks.size()) - 1) : 0;
    ASSERT_EQ(intersection.size(), expected_size);
}

// Custom main function for MPI.
int main(int argc, char* argv[]) {
    int result = 0;

    // Initialize GoogleTest and MPI.
    ::testing::InitGoogleTest(&argc, argv);
    simplemc::mpi::environment env(argc, argv);
    simplemc::mpi::communicator comm;

    // Remove the default GoogleTest listener on all ranks except 0.
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    if (comm.rank() != 0) {
        delete listeners.Release(listeners.default_result_printer());
    }

    // Run the tests.
    result = RUN_ALL_TESTS();

    return result;
}
