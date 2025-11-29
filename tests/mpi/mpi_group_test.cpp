#include <fmt/base.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include <simplemc/mpi.hpp>

#include <vector>

// Test default constructor creates an empty group.
TEST(SimplemcMPIGroup, DefaultConstructor) {
    simplemc::mpi::group g {};
    ASSERT_TRUE(g.empty());
    ASSERT_EQ(g.size(), 0);
    ASSERT_EQ(simplemc::mpi::group_compare(g, simplemc::mpi::group {}), MPI_IDENT);
}

// Test constructing a group from a communicator.
TEST(SimplemcMPIGroup, FromCommunicator) {
    simplemc::mpi::communicator comm {};
    simplemc::mpi::group g = comm.get_group();
    ASSERT_EQ(g.size(), comm.size());
    ASSERT_EQ(g.rank(), comm.rank());
    fmt::print("Group from communicator: rank {} of {} processes.\n", g.rank(), g.size());
}

// Test boolean conversion.
TEST(SimplemcMPIGroup, BoolConversion) {
    // default-constructed group wraps MPI_GROUP_EMPTY which is valid but empty
    simplemc::mpi::group empty_group {};
    ASSERT_TRUE(empty_group.empty());

    // group from communicator should be valid
    simplemc::mpi::communicator comm {};
    auto g = comm.get_group();
    ASSERT_TRUE(static_cast<bool>(g));
    ASSERT_FALSE(g.empty());

    // group from MPI_GROUP_NULL should be invalid
    simplemc::mpi::group null_group { MPI_GROUP_NULL, simplemc::mpi::resource_policy::attach };
    ASSERT_FALSE(static_cast<bool>(null_group));
}

// Test include operation.
TEST(SimplemcMPIGroup, Include) {
    simplemc::mpi::communicator comm {};
    auto g = comm.get_group();

    // include only rank 0
    auto ranks = std::vector<int> { 0 };
    auto subgroup = g.include(ranks);

    ASSERT_EQ(subgroup.size(), 1);
    if (comm.rank() == 0) {
        ASSERT_EQ(subgroup.rank(), 0);
    } else {
        ASSERT_EQ(subgroup.rank(), MPI_UNDEFINED);
    }
    fmt::print("Rank {}: subgroup rank = {}\n", comm.rank(), subgroup.rank());

    // include no ranks should create an empty group
    auto empty_subgroup = g.include(std::vector<int> {});
    ASSERT_TRUE(empty_subgroup.empty());
    ASSERT_EQ(simplemc::mpi::group_compare(empty_subgroup, simplemc::mpi::group {}), MPI_IDENT);
}

// Test exclude operation.
TEST(SimplemcMPIGroup, Exclude) {
    simplemc::mpi::communicator comm {};
    auto g = comm.get_group();

    // exclude rank 0
    auto ranks = std::vector<int> { 0 };
    auto subgroup = g.exclude(ranks);

    ASSERT_EQ(subgroup.size(), comm.size() - 1);
    if (comm.rank() == 0) {
        ASSERT_EQ(subgroup.rank(), MPI_UNDEFINED);
    } else {
        // ranks shift down after excluding rank 0
        ASSERT_EQ(subgroup.rank(), comm.rank() - 1);
    }
    fmt::print("Rank {}: excluded subgroup rank = {}\n", comm.rank(), subgroup.rank());

    // excluding all ranks creates an empty group that compares identical to MPI_GROUP_EMPTY
    auto all_ranks = std::vector<int> {};
    for (int i = 0; i < comm.size(); ++i) {
        all_ranks.push_back(i);
    }
    auto empty_excluded = g.exclude(all_ranks);
    ASSERT_TRUE(empty_excluded.empty());
    ASSERT_EQ(simplemc::mpi::group_compare(empty_excluded, simplemc::mpi::group {}), MPI_IDENT);
}

// Test include with even ranks.
TEST(SimplemcMPIGroup, IncludeEvenRanks) {
    simplemc::mpi::communicator comm {};
    auto g = comm.get_group();

    // include only even ranks
    auto even_ranks = std::vector<int> {};
    for (int i = 0; i < comm.size(); i += 2) {
        even_ranks.push_back(i);
    }
    auto even_group = g.include(even_ranks);

    int const expected_size = (comm.size() + 1) / 2;
    ASSERT_EQ(even_group.size(), expected_size);
    if (comm.rank() % 2 == 0) {
        ASSERT_EQ(even_group.rank(), comm.rank() / 2);
    } else {
        ASSERT_EQ(even_group.rank(), MPI_UNDEFINED);
    }
    fmt::print("Rank {}: even_group rank = {}\n", comm.rank(), even_group.rank());
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
    auto union_group = simplemc::mpi::group_union(even_group, odd_group);
    ASSERT_EQ(union_group.size(), comm.size());
    fmt::print("Union group size: {}\n", union_group.size());

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
    auto intersection_group = simplemc::mpi::group_intersection(group1, group2);

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
    fmt::print("Intersection group size: {}\n", intersection_group.size());

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
    auto diff_group = simplemc::mpi::group_difference(g, first_half_group);
    ASSERT_EQ(diff_group.size(), comm.size() - static_cast<int>(first_half.size()));
    fmt::print("Difference group size: {}\n", diff_group.size());

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

    // empty group compared with itself should also be identical
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

// test implicit conversion to MPI_Group
TEST(SimplemcMPIGroup, ImplicitConversion) {
    simplemc::mpi::communicator comm {};
    auto g = comm.get_group();

    // should be able to use group where MPI_Group is expected
    MPI_Group mpi_grp = g;
    int size {};
    MPI_Group_size(mpi_grp, &size);
    ASSERT_EQ(size, comm.size());
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
