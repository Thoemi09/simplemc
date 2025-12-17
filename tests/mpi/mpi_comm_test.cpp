#include "../gtest-mpi-listener.hpp"

#include <gtest/gtest.h>
#include <mpi.h>
#include <simplemc/mpi.hpp>

#include <vector>

// Test comm_free operation.
TEST(SimplemcMPI, CommFree) {
    // free a communicator obtained by splitting
    simplemc::mpi::communicator comm {};
    MPI_Comm mpi_comm = comm.split(0, 0, simplemc::mpi::resource_policy::attach);
    ASSERT_NO_THROW(simplemc::mpi::comm_free(mpi_comm));
    ASSERT_EQ(mpi_comm, MPI_COMM_NULL);

    // freeing MPI_COMM_WORLD should have no effect (predefined communicator)
    MPI_Comm world_comm = MPI_COMM_WORLD;
    ASSERT_NO_THROW(simplemc::mpi::comm_free(world_comm));
    ASSERT_EQ(world_comm, MPI_COMM_WORLD);
}

// Test communicator comparison and equality operators.
TEST(SimplemcMPI, CommCompare) {
    simplemc::mpi::communicator comm1 {};
    simplemc::mpi::communicator comm2 {};

    // same communicator should be identical
    ASSERT_EQ(comm1.compare(comm2), MPI_IDENT);
    ASSERT_EQ(simplemc::mpi::comm_compare(comm1, comm2), MPI_IDENT);
    ASSERT_EQ(comm1, comm2);

    // duplicated communicator should be congruent (same group, different context)
    auto dup_comm = comm1.duplicate();
    ASSERT_EQ(comm1.compare(dup_comm), MPI_CONGRUENT);
    ASSERT_EQ(simplemc::mpi::comm_compare(comm1, dup_comm), MPI_CONGRUENT);
    ASSERT_NE(comm1, dup_comm);

    // different communicators should be unequal
    if (comm1.size() > 1) {
        simplemc::mpi::communicator self { MPI_COMM_SELF };
        ASSERT_EQ(comm1.compare(self), MPI_UNEQUAL);
        ASSERT_EQ(simplemc::mpi::comm_compare(comm1, self), MPI_UNEQUAL);
        ASSERT_NE(comm1, self);
    }
}

// Test default constructor wraps MPI_COMM_WORLD.
TEST(SimplemcMPI, CommDefaultConstructor) {
    simplemc::mpi::communicator comm {};
    ASSERT_TRUE(comm);
    ASSERT_GT(comm.size(), 0);
    ASSERT_GE(comm.rank(), 0);
    ASSERT_LT(comm.rank(), comm.size());
    ASSERT_EQ(comm.compare(simplemc::mpi::communicator {}), MPI_IDENT);
}

// Test constructor with MPI_COMM_WORLD and attach policy.
TEST(SimplemcMPI, CommConstructorWithAttachPolicy) {
    simplemc::mpi::communicator comm { MPI_COMM_WORLD, simplemc::mpi::resource_policy::attach };
    ASSERT_TRUE(comm);

    // compare with default constructor which also wraps MPI_COMM_WORLD
    simplemc::mpi::communicator default_comm {};
    ASSERT_EQ(comm, simplemc::mpi::communicator {});
}

// Test implicit conversion to MPI_Comm.
TEST(SimplemcMPI, CommImplicitConversion) {
    simplemc::mpi::communicator comm {};

    // should be able to use communicator where MPI_Comm is expected
    int size {};
    MPI_Comm_size(comm, &size);
    ASSERT_EQ(size, comm.size());
}

// Test explicit conversion to bool.
TEST(SimplemcMPI, CommExplicitBoolConversion) {
    simplemc::mpi::communicator null_comm { MPI_COMM_NULL };
    ASSERT_FALSE(static_cast<bool>(null_comm));

    simplemc::mpi::communicator world_comm { MPI_COMM_WORLD };
    ASSERT_TRUE(static_cast<bool>(world_comm));

    simplemc::mpi::communicator self_comm { MPI_COMM_SELF };
    ASSERT_TRUE(static_cast<bool>(self_comm));
}

// Test communicator size and rank.
TEST(SimplemcMPI, CommSizeCommRank) {
    simplemc::mpi::communicator comm {};

    ASSERT_GT(comm.size(), 0);
    ASSERT_EQ(simplemc::mpi::comm_size(comm), comm.size());
    ASSERT_GE(comm.rank(), 0);
    ASSERT_LT(comm.rank(), comm.size());
    ASSERT_EQ(simplemc::mpi::comm_rank(comm), comm.rank());
}

// Test shared_ptr semantics - copying should share the underlying communicator.
TEST(SimplemcMPI, CommSharedPtrSemantics) {
    simplemc::mpi::communicator comm1 {};

    // make copy
    auto comm2 = comm1;

    // both should refer to the same underlying communicator
    ASSERT_EQ(comm1.size(), comm2.size());
    ASSERT_EQ(comm1.rank(), comm2.rank());

    // they should compare as identical
    ASSERT_EQ(comm1, comm2);
}

// Test basic duplicate operations.
TEST(SimplemcMPI, CommDuplicate) {
    simplemc::mpi::communicator comm {};
    auto comm_owned = comm.duplicate();
    auto comm_attached = comm.duplicate(simplemc::mpi::resource_policy::attach);
    auto comm_free = simplemc::mpi::comm_dup(comm);

    ASSERT_TRUE(comm_owned);
    ASSERT_TRUE(comm_attached);
    ASSERT_TRUE(comm_free);
    ASSERT_EQ(comm_owned.compare(comm), MPI_CONGRUENT);
    ASSERT_EQ(comm_attached.compare(comm), MPI_CONGRUENT);
    ASSERT_EQ(simplemc::mpi::comm_compare(comm_free, comm), MPI_CONGRUENT);

    MPI_Comm mpi_comm_attached = comm_attached;
    simplemc::mpi::comm_free(mpi_comm_attached);
    simplemc::mpi::comm_free(comm_free);
}

// Test basic split operations.
TEST(SimplemcMPI, CommSplit) {
    simplemc::mpi::communicator comm {};

    // trivial split where all ranks go to same communicator
    auto comm_trivial_split = comm.split(0);
    ASSERT_EQ(comm.compare(comm_trivial_split), MPI_CONGRUENT);

    // split by even/odd ranks
    int const expected_size = (comm.rank() % 2 == 0) ? (comm.size() + 1) / 2 : comm.size() / 2;
    int const color = comm.rank() % 2;
    auto split_comm = comm.split(color, 0, simplemc::mpi::resource_policy::attach);
    ASSERT_TRUE(split_comm);
    ASSERT_EQ(split_comm.size(), expected_size);
    ASSERT_EQ(split_comm.rank(), comm.rank() / 2);

    MPI_Comm mpi_split_comm = split_comm;
    simplemc::mpi::comm_free(mpi_split_comm);

    // split rank zero from others
    int const undef_color = (comm.rank() == 0) ? 0 : MPI_UNDEFINED;
    auto split_zero_comm = simplemc::mpi::comm_split(comm, undef_color);
    if (comm.rank() == 0) {
        ASSERT_TRUE(split_zero_comm);
        ASSERT_EQ(simplemc::mpi::comm_size(split_zero_comm), 1);
        ASSERT_EQ(simplemc::mpi::comm_rank(split_zero_comm), 0);
    } else {
        ASSERT_EQ(split_zero_comm, MPI_COMM_NULL);
    }
    simplemc::mpi::comm_free(split_zero_comm);
}

// Test basic create operations.
TEST(SimplemcMPI, CommCreate) {
    simplemc::mpi::communicator comm {};
    auto grp = comm.get_group();

    // create communicator with all ranks should give same size/rank
    auto full_comm = comm.create(grp);
    ASSERT_EQ(full_comm.compare(comm), MPI_CONGRUENT);

    // create communicator with only rank 0
    auto grp_zero = grp.include(std::vector<int> { 0 });
    auto zero_comm = comm.create(grp_zero, simplemc::mpi::resource_policy::attach);
    if (comm.rank() == 0) {
        ASSERT_TRUE(zero_comm);
        ASSERT_EQ(zero_comm.size(), 1);
        ASSERT_EQ(zero_comm.rank(), 0);
    } else {
        ASSERT_FALSE(zero_comm);
    }
    MPI_Comm mpi_zero_comm = zero_comm;
    simplemc::mpi::comm_free(mpi_zero_comm);

    // create communicator with last rank excluded
    if (comm.size() > 1) {
        auto grp_excl_last = grp.exclude(std::vector<int> { comm.size() - 1 });
        auto excl_last_comm = simplemc::mpi::comm_create(comm, grp_excl_last);
        if (comm.rank() != comm.size() - 1) {
            ASSERT_NE(excl_last_comm, MPI_COMM_NULL);
            ASSERT_EQ(simplemc::mpi::comm_size(excl_last_comm), comm.size() - 1);
            ASSERT_EQ(simplemc::mpi::comm_rank(excl_last_comm), comm.rank());
        } else {
            ASSERT_EQ(excl_last_comm, MPI_COMM_NULL);
        }
        simplemc::mpi::comm_free(excl_last_comm);
    }
}

// Test splitting and creating communicators with even/odd ranks.
TEST(SimplemcMPI, CommSplitCreateEvenOdd) {
    simplemc::mpi::communicator comm {};

    // split communicator into even/odd ranks
    int const color = comm.rank() % 2;
    auto split_comm = comm.split(color);

    // create communicator with even/odd ranks using group operations
    auto even_ranks = std::vector<int> {};
    for (int i = 0; i < comm.size(); i += 2) {
        even_ranks.push_back(i);
    }
    auto grp = comm.get_group();
    auto even_grp = grp.include(even_ranks);
    auto odd_grp = grp.exclude(even_ranks);

    auto even_comm = comm.create(even_grp);
    auto odd_comm = comm.create(odd_grp);

    // check that communicators match
    if (comm.rank() % 2 == 0) {
        ASSERT_EQ(split_comm.compare(even_comm), MPI_CONGRUENT);
    } else {
        ASSERT_EQ(split_comm.compare(odd_comm), MPI_CONGRUENT);
    }
}

// Test barrier operation.
TEST(SimplemcMPI, CommBarrier) {
    simplemc::mpi::communicator comm {};

    // barrier should complete without error
    ASSERT_NO_THROW(comm.barrier());
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
