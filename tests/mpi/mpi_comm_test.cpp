#include <gtest/gtest.h>
#include <mpi.h>
#include <simplemc/mpi.hpp>

#include <vector>

// Test default constructor wraps MPI_COMM_WORLD.
TEST(SimplemcMPIComm, DefaultConstructor) {
    simplemc::mpi::communicator comm {};
    ASSERT_TRUE(comm);
    ASSERT_GT(comm.size(), 0);
    ASSERT_GE(comm.rank(), 0);
    ASSERT_LT(comm.rank(), comm.size());
}

// Test constructor with MPI_COMM_WORLD and attach policy.
TEST(SimplemcMPIComm, ConstructorWithAttachPolicy) {
    simplemc::mpi::communicator comm { MPI_COMM_WORLD, simplemc::mpi::resource_policy::attach };
    ASSERT_TRUE(comm);

    // compare with default constructor which also wraps MPI_COMM_WORLD
    simplemc::mpi::communicator default_comm {};
    ASSERT_EQ(comm, default_comm);
}

// Test constructor with MPI_COMM_SELF.
TEST(SimplemcMPIComm, ConstructorWithCommSelf) {
    // we can take ownership since MPI_COMM_SELF is predefined and is not freed anyway
    simplemc::mpi::communicator comm { MPI_COMM_SELF };
    ASSERT_TRUE(comm);
    ASSERT_EQ(comm.size(), 1);
    ASSERT_EQ(comm.rank(), 0);
}

// Test constructor with MPI_COMM_NULL.
TEST(SimplemcMPIComm, ConstructorWithCommNull) {
    simplemc::mpi::communicator null_comm { MPI_COMM_NULL };
    ASSERT_FALSE(null_comm);
}

// Test implicit conversion to MPI_Comm.
TEST(SimplemcMPIComm, ImplicitConversion) {
    simplemc::mpi::communicator comm {};

    // should be able to use communicator where MPI_Comm is expected
    int size {};
    MPI_Comm_size(comm, &size);
    ASSERT_EQ(size, comm.size());
}

// Test duplicate operation.
TEST(SimplemcMPIComm, Duplicate) {
    MPI_Comm mpi_comm = MPI_COMM_NULL;

    {
        simplemc::mpi::communicator comm {};
        auto comm_owned = comm.duplicate();
        auto comm_attached = comm.duplicate(simplemc::mpi::resource_policy::attach);

        ASSERT_TRUE(comm_owned);
        ASSERT_TRUE(comm_attached);
        ASSERT_EQ(comm_owned.size(), comm.size());
        ASSERT_EQ(comm_attached.size(), comm.size());
        ASSERT_EQ(comm_owned.rank(), comm.rank());
        ASSERT_EQ(comm_attached.rank(), comm.rank());

        // if we'd assing comm_owned, the following ops would be UB
        mpi_comm = comm_attached;
    }

    // check that we can still use the attached communicator
    int size {};
    MPI_Comm_size(mpi_comm, &size);
    ASSERT_GT(size, 0);

    // manually free mpi_comm
    simplemc::mpi::comm_free(mpi_comm);
}

// Test get_group operation.
TEST(SimplemcMPIComm, GetGroup) {
    MPI_Group mpi_group = MPI_GROUP_NULL;

    {
        simplemc::mpi::communicator comm {};
        auto grp_owned = comm.get_group();
        auto grp_attached = comm.get_group(simplemc::mpi::resource_policy::attach);

        ASSERT_TRUE(grp_owned);
        ASSERT_TRUE(grp_attached);
        ASSERT_EQ(grp_owned.size(), comm.size());
        ASSERT_EQ(grp_attached.size(), comm.size());
        ASSERT_EQ(grp_owned.rank(), comm.rank());
        ASSERT_EQ(grp_attached.rank(), comm.rank());

        // if we'd assign grp_owned, the following ops would be UB
        mpi_group = grp_attached;
    }

    // check that we can still use the attached group
    int size {};
    MPI_Group_size(mpi_group, &size);
    ASSERT_GT(size, 0);

    // manually free mpi_group
    simplemc::mpi::group_free(mpi_group);
}

// Test create operation with a subgroup.
TEST(SimplemcMPIComm, Create) {
    simplemc::mpi::communicator comm {};
    auto grp = comm.get_group();

    // create a subgroup with only rank 0
    auto ranks = std::vector<int> { 0 };
    auto subgroup = grp.include(ranks);

    // create communicator from subgroup
    auto sub_comm = comm.create(subgroup);

    if (comm.rank() == 0) {
        ASSERT_TRUE(sub_comm);
        ASSERT_EQ(sub_comm.size(), 1);
        ASSERT_EQ(sub_comm.rank(), 0);
    } else {
        // processes not in the subgroup get MPI_COMM_NULL
        ASSERT_FALSE(sub_comm);
    }
}

// Test create with even ranks and resource_policy::attach.
TEST(SimplemcMPIComm, CreateEvenRanks) {
    simplemc::mpi::communicator comm {};
    auto grp = comm.get_group();

    // create a subgroup and communicator with only even ranks
    auto even_ranks = std::vector<int> {};
    for (int i = 0; i < comm.size(); i += 2) {
        even_ranks.push_back(i);
    }
    auto even_group = grp.include(even_ranks);
    auto even_comm = comm.create(even_group, simplemc::mpi::resource_policy::attach);

    if (comm.rank() % 2 == 0) {
        ASSERT_TRUE(even_comm);
        ASSERT_EQ(even_comm.size(), static_cast<int>(even_ranks.size()));
        ASSERT_EQ(even_comm.rank(), comm.rank() / 2);
    } else {
        ASSERT_FALSE(even_comm);
    }

    // manually free even_comm's MPI_Comm
    MPI_Comm mpi_even_comm = even_comm;
    simplemc::mpi::comm_free(mpi_even_comm);
}

// Test split operation by even/odd ranks.
TEST(SimplemcMPIComm, SplitEvenOdd) {
    simplemc::mpi::communicator comm {};

    int const color = comm.rank() % 2;
    auto split_comm = comm.split(color);
    ASSERT_TRUE(split_comm);

    // calculate expected size
    int const expected_size = (comm.rank() % 2 == 0) ? (comm.size() + 1) / 2 : comm.size() / 2;
    ASSERT_EQ(split_comm.size(), expected_size);

    // rank should be half the original rank (rounded down)
    ASSERT_EQ(split_comm.rank(), comm.rank() / 2);
}

// Test split with MPI_UNDEFINED color and resource_policy::attach.
TEST(SimplemcMPIComm, SplitUndefined) {
    simplemc::mpi::communicator comm {};

    // only rank 0 joins a communicator, others use MPI_UNDEFINED
    int const color = (comm.rank() == 0) ? 0 : MPI_UNDEFINED;
    auto split_comm = comm.split(color, 0, simplemc::mpi::resource_policy::attach);

    if (comm.rank() == 0) {
        ASSERT_TRUE(split_comm);
        ASSERT_EQ(split_comm.size(), 1);
        ASSERT_EQ(split_comm.rank(), 0);
    } else {
        ASSERT_FALSE(split_comm);
    }

    // manually free split_comm's MPI_Comm
    MPI_Comm mpi_split_comm = split_comm;
    simplemc::mpi::comm_free(mpi_split_comm);
}

// Test split with custom key for rank ordering.
TEST(SimplemcMPIComm, SplitWithKey) {
    simplemc::mpi::communicator comm {};

    // all processes in same communicator, but reverse rank ordering
    int const color = 0;
    int const key = comm.size() - comm.rank() - 1;
    auto split_comm = comm.split(color, key);

    ASSERT_TRUE(split_comm);
    ASSERT_EQ(split_comm.size(), comm.size());
    ASSERT_EQ(split_comm.rank(), comm.size() - comm.rank() - 1);
}

// Test barrier operation.
TEST(SimplemcMPIComm, Barrier) {
    simplemc::mpi::communicator comm {};

    // barrier should complete without error
    ASSERT_NO_THROW(comm.barrier());
}

// Test compare operation.
TEST(SimplemcMPIComm, Compare) {
    simplemc::mpi::communicator comm1 {};
    simplemc::mpi::communicator comm2 {};

    // same communicator should be identical
    ASSERT_EQ(simplemc::mpi::comm_compare(comm1, comm2), MPI_IDENT);

    // duplicated communicator should be congruent
    auto dup_comm = comm1.duplicate();
    ASSERT_EQ(simplemc::mpi::comm_compare(comm1, dup_comm), MPI_CONGRUENT);

    // MPI_COMM_WORLD and MPI_COMM_SELF have different groups
    simplemc::mpi::communicator world { MPI_COMM_WORLD };
    simplemc::mpi::communicator self { MPI_COMM_SELF };
    if (comm1.size() > 1) {
        ASSERT_EQ(simplemc::mpi::comm_compare(world, self), MPI_UNEQUAL);
    }
}

// Test equality operators.
TEST(SimplemcMPIComm, EqualityOperators) {
    simplemc::mpi::communicator comm1 {};
    simplemc::mpi::communicator comm2 {};

    // same communicator should be equal
    ASSERT_TRUE(comm1 == comm2);
    ASSERT_FALSE(comm1 != comm2);

    // duplicated communicator should not be equal (different context)
    auto dup_comm = comm1.duplicate();
    ASSERT_FALSE(comm1 == dup_comm);
    ASSERT_TRUE(comm1 != dup_comm);
}

// Test shared_ptr semantics - copying should share the underlying communicator.
TEST(SimplemcMPIComm, SharedPtrSemantics) {
    simplemc::mpi::communicator comm1 {};

    // make copy
    auto comm2 = comm1;

    // both should refer to the same underlying communicator
    ASSERT_EQ(comm1.size(), comm2.size());
    ASSERT_EQ(comm1.rank(), comm2.rank());

    // they should compare as identical
    ASSERT_TRUE(comm1 == comm2);
}

// Test comm_free operation.
TEST(SimplemcMPIComm, CommFree) {
    simplemc::mpi::communicator comm {};

    // duplicate to get a communicator we can free
    MPI_Comm mpi_comm = comm.duplicate(simplemc::mpi::resource_policy::attach);

    // free should not throw
    ASSERT_NO_THROW(simplemc::mpi::comm_free(mpi_comm));

    // after free, communicator should be MPI_COMM_NULL
    ASSERT_EQ(mpi_comm, MPI_COMM_NULL);
}

// Test comm_free on MPI_COMM_WORLD does nothing (predefined communicator).
TEST(SimplemcMPIComm, CommFreeCommWorld) {
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    // free on MPI_COMM_WORLD should not actually free it
    ASSERT_NO_THROW(simplemc::mpi::comm_free(mpi_comm));

    // MPI_COMM_WORLD is predefined, comm_free() should have no effect
    ASSERT_EQ(mpi_comm, MPI_COMM_WORLD);
}

// Test comm_free on split communicator.
TEST(SimplemcMPIComm, CommFreeSplitComm) {
    simplemc::mpi::communicator comm {};

    // split to get a communicator we can free
    MPI_Comm mpi_comm = comm.split(0, 0, simplemc::mpi::resource_policy::attach);

    // free should not throw
    ASSERT_NO_THROW(simplemc::mpi::comm_free(mpi_comm));

    // after free, communicator should be MPI_COMM_NULL
    ASSERT_EQ(mpi_comm, MPI_COMM_NULL);
}

// Test creating multiple communicators and comparing them.
TEST(SimplemcMPIComm, MultipleOperations) {
    simplemc::mpi::communicator comm {};

    // duplicate twice
    auto dup1 = comm.duplicate();
    auto dup2 = comm.duplicate();

    // both duplicates should be congruent to original but not identical to each other
    ASSERT_EQ(simplemc::mpi::comm_compare(comm, dup1), MPI_CONGRUENT);
    ASSERT_EQ(simplemc::mpi::comm_compare(comm, dup2), MPI_CONGRUENT);
    ASSERT_EQ(simplemc::mpi::comm_compare(dup1, dup2), MPI_CONGRUENT);

    // split into two groups
    int const color = comm.rank() % 2;
    auto split1 = comm.split(color);
    auto split2 = comm.split(color);

    // splits with same color pattern should be congruent
    ASSERT_EQ(simplemc::mpi::comm_compare(split1, split2), MPI_CONGRUENT);
    ASSERT_EQ(simplemc::mpi::comm_compare(comm, split1), MPI_UNEQUAL);
    ASSERT_EQ(simplemc::mpi::comm_compare(comm, split2), MPI_UNEQUAL);
}

// Test chain of operations: duplicate -> split -> create.
TEST(SimplemcMPIComm, ChainedOperations) {
    simplemc::mpi::communicator comm {};

    // duplicate
    auto dup_comm = comm.duplicate();
    ASSERT_EQ(dup_comm.size(), comm.size());

    // split the duplicate
    int const color = dup_comm.rank() % 2;
    auto split_comm = dup_comm.split(color);

    // get group from split communicator
    auto grp = split_comm.get_group();
    ASSERT_EQ(grp.size(), split_comm.size());

    // create a new communicator with all processes in the split group
    auto ranks = std::vector<int> {};
    for (int i = 0; i < split_comm.size(); ++i) {
        ranks.push_back(i);
    }
    auto all_subgroup = grp.include(ranks);
    auto created_comm = split_comm.create(all_subgroup);

    // created communicator should be congruent to split_comm
    ASSERT_EQ(simplemc::mpi::comm_compare(split_comm, created_comm), MPI_CONGRUENT);
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
