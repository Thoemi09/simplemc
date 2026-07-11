// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include "../gtest-mpi-listener.hpp"

#include <gtest/gtest.h>
#include <mpi.h>
#include <simplemc/mpi.hpp>

#include <vector>

// Test type_free and type_commit operations.
TEST(SimplemcMPI, TypeFreeAndCommit) {
    // commit and free a derived datatype
    MPI_Datatype contiguous_dt = simplemc::mpi::type_contiguous(2, MPI_INT);
    simplemc::mpi::type_commit(contiguous_dt);
    simplemc::mpi::type_commit(contiguous_dt); // committing again should be no-op
    ASSERT_NO_THROW(simplemc::mpi::type_free(contiguous_dt));
    ASSERT_EQ(contiguous_dt, MPI_DATATYPE_NULL);

    // committing and freeing a predefined type should have no effect
    MPI_Datatype int_dt = MPI_INT;
    simplemc::mpi::type_commit(int_dt);
    ASSERT_NO_THROW(simplemc::mpi::type_free(int_dt));
    ASSERT_EQ(int_dt, MPI_INT);

    // committing and freeing MPI_DATATYPE_NULL should have no effect
    MPI_Datatype null_dt = MPI_DATATYPE_NULL;
    simplemc::mpi::type_commit(null_dt);
    ASSERT_NO_THROW(simplemc::mpi::type_free(null_dt));
    ASSERT_EQ(null_dt, MPI_DATATYPE_NULL);
}

// Test default constructor creates null datatype.
TEST(SimplemcMPI, TypeDefaultAndNullConstructor) {
    simplemc::mpi::datatype dt {};
    ASSERT_FALSE(dt);
    ASSERT_EQ(dt, MPI_DATATYPE_NULL);

    // explicit construction with MPI_DATATYPE_NULL
    simplemc::mpi::datatype dt_null { MPI_DATATYPE_NULL };
    ASSERT_FALSE(dt_null);
}

// Test constructor with predefined types.
TEST(SimplemcMPI, TypeConstructorWithPredefinedType) {
    simplemc::mpi::datatype dt_int { MPI_INT };
    ASSERT_TRUE(dt_int);
    ASSERT_EQ(dt_int, MPI_INT);

    simplemc::mpi::datatype dt_db { MPI_DOUBLE, simplemc::mpi::resource_policy::attach };
    ASSERT_TRUE(dt_db);
    ASSERT_EQ(dt_db, MPI_DOUBLE);
}

// Test implicit conversion to MPI_Datatype.
TEST(SimplemcMPI, TypeImplicitConversion) {
    simplemc::mpi::datatype dt { MPI_INT };

    // should be able to use datatype where MPI_Datatype is expected
    int size = 0;
    MPI_Type_size(dt, &size);
    ASSERT_EQ(size, static_cast<int>(sizeof(int)));
}

// Test explicit conversion to bool.
TEST(SimplemcMPI, TypeExplicitBoolConversion) {
    simplemc::mpi::datatype null_dt { MPI_DATATYPE_NULL };
    ASSERT_FALSE(static_cast<bool>(null_dt));

    simplemc::mpi::datatype int_dt { MPI_INT };
    ASSERT_TRUE(static_cast<bool>(int_dt));
}

// Test datatype size and extent.
TEST(SimplemcMPI, TypeSizeAndExtent) {
    simplemc::mpi::datatype dt_int { MPI_INT };
    simplemc::mpi::datatype dt_double { MPI_DOUBLE };

    // size
    ASSERT_EQ(dt_int.size(), static_cast<int>(sizeof(int)));
    ASSERT_EQ(simplemc::mpi::type_size(dt_double), static_cast<int>(sizeof(double)));

    // extent and lower bound
    auto [lb_int, extent_int] = simplemc::mpi::type_get_extent(dt_int);
    ASSERT_EQ(lb_int, 0);
    ASSERT_EQ(extent_int, static_cast<MPI_Aint>(sizeof(int)));

    auto [lb_db, extent_db] = dt_double.extent();
    ASSERT_EQ(lb_db, 0);
    ASSERT_EQ(extent_db, static_cast<MPI_Aint>(sizeof(double)));
}

// Test predefined type check.
TEST(SimplemcMPI, TypeIsPredefined) {
    ASSERT_TRUE(simplemc::mpi::datatype(MPI_INT).is_predefined());
    ASSERT_TRUE(simplemc::mpi::type_is_predefined(MPI_DOUBLE));
    ASSERT_TRUE(simplemc::mpi::datatype(MPI_CHAR).is_predefined());
    ASSERT_TRUE(simplemc::mpi::type_is_predefined(MPI_BYTE));
    ASSERT_FALSE(simplemc::mpi::datatype().is_predefined());
    ASSERT_FALSE(simplemc::mpi::type_is_predefined(MPI_DATATYPE_NULL));

    // derived type should not be predefined
    auto dt = simplemc::mpi::datatype { 2, MPI_INT };
    ASSERT_FALSE(dt.is_predefined());
    ASSERT_FALSE(simplemc::mpi::type_is_predefined(dt));
}

// Test type envelope.
TEST(SimplemcMPI, TypeEnvelope) {
    // predefined types
    simplemc::mpi::datatype dt_int { MPI_INT };
    auto env_int = dt_int.envelope();
    ASSERT_EQ(env_int.combiner, MPI_COMBINER_NAMED);

    auto env_char = simplemc::mpi::type_get_envelope(MPI_CHAR);
    ASSERT_EQ(env_char.combiner, MPI_COMBINER_NAMED);

    // derived type
    simplemc::mpi::datatype dt_der { 2, MPI_INT };
    auto env_der = dt_der.envelope();
    ASSERT_EQ(env_der.combiner, MPI_COMBINER_CONTIGUOUS);
    ASSERT_EQ(env_der.num_datatypes, 1);
    ASSERT_EQ(env_der.num_integers, 1);
    ASSERT_EQ(env_der.num_addresses, 0);
}

// Test constructor for contiguous datatype.
TEST(SimplemcMPI, TypeContiguous) {
    simplemc::mpi::datatype dt_owned { 2, MPI_INT };
    simplemc::mpi::datatype dt_attached { 3, MPI_DOUBLE, simplemc::mpi::resource_policy::attach };
    auto dt_free = simplemc::mpi::type_contiguous(4, MPI_CHAR);
    simplemc::mpi::type_commit(dt_free);

    // validity checks
    ASSERT_TRUE(dt_owned);
    ASSERT_TRUE(dt_attached);
    ASSERT_NE(dt_free, MPI_DATATYPE_NULL);

    // size checks
    ASSERT_EQ(dt_owned.size(), 2 * static_cast<int>(sizeof(int)));
    ASSERT_EQ(dt_attached.size(), 3 * static_cast<int>(sizeof(double)));
    ASSERT_EQ(simplemc::mpi::type_size(dt_free), 4 * static_cast<int>(sizeof(char)));

    // envelope checks
    auto check_envelope = [](auto env) {
        ASSERT_EQ(env.combiner, MPI_COMBINER_CONTIGUOUS);
        ASSERT_EQ(env.num_datatypes, 1);
        ASSERT_EQ(env.num_integers, 1);
        ASSERT_EQ(env.num_addresses, 0);
    };
    check_envelope(dt_owned.envelope());
    check_envelope(dt_attached.envelope());
    check_envelope(simplemc::mpi::type_get_envelope(dt_free));

    MPI_Datatype mpi_dt_attached = dt_attached;
    simplemc::mpi::type_free(mpi_dt_attached);
    simplemc::mpi::type_free(dt_free);
}

// Test vector constructor.
TEST(SimplemcMPI, TypeVector) {
    simplemc::mpi::datatype dt_owned { 2, 3, 5, MPI_INT };
    simplemc::mpi::datatype dt_attached { 2, 3, 5, MPI_DOUBLE, simplemc::mpi::resource_policy::attach };
    auto dt_free = simplemc::mpi::type_vector(2, 3, 5, MPI_CHAR);
    simplemc::mpi::type_commit(dt_free);

    // validity checks
    ASSERT_TRUE(dt_owned);
    ASSERT_TRUE(dt_attached);
    ASSERT_NE(dt_free, MPI_DATATYPE_NULL);

    // size checks
    ASSERT_EQ(dt_owned.size(), 2 * 3 * static_cast<int>(sizeof(int)));
    ASSERT_EQ(dt_attached.size(), 2 * 3 * static_cast<int>(sizeof(double)));
    ASSERT_EQ(simplemc::mpi::type_size(dt_free), 2 * 3 * static_cast<int>(sizeof(char)));

    // envelope checks
    auto check_envelope = [](auto env) {
        ASSERT_EQ(env.combiner, MPI_COMBINER_VECTOR);
        ASSERT_EQ(env.num_datatypes, 1);
        ASSERT_EQ(env.num_integers, 3);
        ASSERT_EQ(env.num_addresses, 0);
    };
    check_envelope(dt_owned.envelope());
    check_envelope(dt_attached.envelope());
    check_envelope(simplemc::mpi::type_get_envelope(dt_free));

    MPI_Datatype mpi_dt_attached = dt_attached;
    simplemc::mpi::type_free(mpi_dt_attached);
    simplemc::mpi::type_free(dt_free);
}

// Test struct constructor.
TEST(SimplemcMPI, TypeCreateStruct) {
    // create a struct with int, double, char
    std::vector<int> blocklengths = { 1, 1, 1 };
    std::vector<MPI_Aint> displacements = { 0, sizeof(int), sizeof(int) + sizeof(double) };
    std::vector<MPI_Datatype> oldtypes = { MPI_INT, MPI_DOUBLE, MPI_CHAR };

    simplemc::mpi::datatype dt_owned { blocklengths, displacements, oldtypes };
    simplemc::mpi::datatype dt_attached { blocklengths, displacements, oldtypes,
        simplemc::mpi::resource_policy::attach };
    auto dt_free = simplemc::mpi::type_create_struct(blocklengths, displacements, oldtypes);

    // validity checks
    ASSERT_TRUE(dt_owned);
    ASSERT_TRUE(dt_attached);
    ASSERT_NE(dt_free, MPI_DATATYPE_NULL);

    // size checks
    const int expected_size = static_cast<int>(sizeof(int) + sizeof(double) + sizeof(char));
    ASSERT_EQ(dt_owned.size(), expected_size);
    ASSERT_EQ(dt_attached.size(), expected_size);
    ASSERT_EQ(simplemc::mpi::type_size(dt_free), expected_size);

    // envelope checks
    auto check_envelope = [](auto env) {
        ASSERT_EQ(env.combiner, MPI_COMBINER_STRUCT);
        ASSERT_EQ(env.num_datatypes, 3);
        ASSERT_EQ(env.num_integers, 4);
        ASSERT_EQ(env.num_addresses, 3);
    };
    check_envelope(dt_owned.envelope());
    check_envelope(dt_attached.envelope());
    check_envelope(simplemc::mpi::type_get_envelope(dt_free));

    MPI_Datatype mpi_dt_attached = dt_attached;
    simplemc::mpi::type_free(mpi_dt_attached);
    simplemc::mpi::type_free(dt_free);
}

// Test shared_ptr semantics - copying should share the underlying datatype.
TEST(SimplemcMPI, TypeSharedPtrSemantics) {
    simplemc::mpi::datatype dt1 { MPI_INT };

    // make copy
    auto dt2 = dt1;

    // both should refer to the same underlying datatype
    ASSERT_EQ(static_cast<MPI_Datatype>(dt1), static_cast<MPI_Datatype>(dt2));
    ASSERT_TRUE(dt1);
    ASSERT_TRUE(dt2);
}

// Test attach policy doesn't free on destruction.
TEST(SimplemcMPI, TypeAttachPolicyNoFree) {
    MPI_Datatype raw_dt = simplemc::mpi::type_contiguous(2, MPI_INT);
    simplemc::mpi::type_commit(raw_dt);

    // wrap with attach policy (no ownership)
    {
        simplemc::mpi::datatype dt { raw_dt, simplemc::mpi::resource_policy::attach };
        ASSERT_TRUE(dt);
    }

    // raw datatype should NOT have been freed
    ASSERT_EQ(simplemc::mpi::type_size(raw_dt), 2 * static_cast<int>(sizeof(int)));

    // manual free
    simplemc::mpi::type_free(raw_dt);
    ASSERT_EQ(raw_dt, MPI_DATATYPE_NULL);
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
