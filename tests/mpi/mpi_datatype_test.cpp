#include <gtest/gtest.h>
#include <mpi.h>
#include <simplemc/mpi.hpp>

#include <vector>

// Test default constructor creates null datatype.
TEST(SimplemcMPIDatatype, DefaultConstructor) {
    simplemc::mpi::datatype dt {};
    ASSERT_FALSE(dt);
    ASSERT_EQ(static_cast<MPI_Datatype>(dt), MPI_DATATYPE_NULL);
}

// Test constructor with predefined type and take_ownership policy.
TEST(SimplemcMPIDatatype, ConstructorWithPredefinedType) {
    simplemc::mpi::datatype dt { MPI_INT };
    ASSERT_TRUE(dt);
    ASSERT_EQ(static_cast<MPI_Datatype>(dt), MPI_INT);
}

// Test constructor with predefined type and attach policy.
TEST(SimplemcMPIDatatype, ConstructorWithAttachPolicy) {
    simplemc::mpi::datatype dt { MPI_DOUBLE, simplemc::mpi::resource_policy::attach };
    ASSERT_TRUE(dt);
    ASSERT_EQ(static_cast<MPI_Datatype>(dt), MPI_DOUBLE);
}

// Test constructor with MPI_DATATYPE_NULL.
TEST(SimplemcMPIDatatype, ConstructorWithDatatypeNull) {
    simplemc::mpi::datatype dt { MPI_DATATYPE_NULL };
    ASSERT_FALSE(dt);
}

// Test implicit conversion to MPI_Datatype.
TEST(SimplemcMPIDatatype, ImplicitConversion) {
    simplemc::mpi::datatype dt { MPI_INT };

    // Should be able to use datatype where MPI_Datatype is expected.
    int size = 0;
    MPI_Type_size(dt, &size);
    ASSERT_EQ(size, static_cast<int>(sizeof(int)));
}

// Test size() method.
TEST(SimplemcMPIDatatype, Size) {
    simplemc::mpi::datatype dt_int { MPI_INT };
    simplemc::mpi::datatype dt_double { MPI_DOUBLE };

    ASSERT_EQ(dt_int.size(), static_cast<int>(sizeof(int)));
    ASSERT_EQ(dt_double.size(), static_cast<int>(sizeof(double)));
}

// Test type_combiner() function.
TEST(SimplemcMPIDatatype, TypeCombiner) {
    // Predefined types have combiner MPI_COMBINER_NAMED.
    ASSERT_EQ(simplemc::mpi::type_combiner(MPI_INT), MPI_COMBINER_NAMED);
    ASSERT_EQ(simplemc::mpi::type_combiner(MPI_DOUBLE), MPI_COMBINER_NAMED);

    // Derived types have different combiners.
    MPI_Datatype contiguous_dt = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(2, MPI_INT, &contiguous_dt);
    MPI_Type_commit(&contiguous_dt);
    ASSERT_EQ(simplemc::mpi::type_combiner(contiguous_dt), MPI_COMBINER_CONTIGUOUS);
    MPI_Type_free(&contiguous_dt);
}

// Test type_is_predefined() function.
TEST(SimplemcMPIDatatype, TypeIsPredefined) {
    ASSERT_TRUE(simplemc::mpi::type_is_predefined(MPI_INT));
    ASSERT_TRUE(simplemc::mpi::type_is_predefined(MPI_DOUBLE));
    ASSERT_TRUE(simplemc::mpi::type_is_predefined(MPI_CHAR));
    ASSERT_TRUE(simplemc::mpi::type_is_predefined(MPI_BYTE));
    ASSERT_FALSE(simplemc::mpi::type_is_predefined(MPI_DATATYPE_NULL));
}

// Test shared_ptr semantics - copying should share the underlying datatype.
TEST(SimplemcMPIDatatype, SharedPtrSemantics) {
    simplemc::mpi::datatype dt1 { MPI_INT };

    // Make copy.
    auto dt2 = dt1;

    // Both should refer to the same underlying datatype.
    ASSERT_EQ(static_cast<MPI_Datatype>(dt1), static_cast<MPI_Datatype>(dt2));
    ASSERT_TRUE(dt1);
    ASSERT_TRUE(dt2);
}

// Test datatype_free() on predefined type does nothing.
TEST(SimplemcMPIDatatype, DatatypeFreePredefined) {
    MPI_Datatype dt = MPI_INT;

    // Free on predefined type should not actually free it.
    ASSERT_NO_THROW(simplemc::mpi::type_free(dt));

    // MPI_INT is predefined, datatype_free() should have no effect.
    ASSERT_EQ(dt, MPI_INT);
}

// Test datatype_free() on MPI_DATATYPE_NULL does nothing.
TEST(SimplemcMPIDatatype, DatatypeFreeNull) {
    MPI_Datatype dt = MPI_DATATYPE_NULL;

    // Free on null should not throw.
    ASSERT_NO_THROW(simplemc::mpi::type_free(dt));

    // Should still be null.
    ASSERT_EQ(dt, MPI_DATATYPE_NULL);
}

// Test datatype_free() on derived type.
TEST(SimplemcMPIDatatype, DatatypeFreeDerived) {
    // Create a simple derived type.
    MPI_Datatype dt = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(2, MPI_INT, &dt);
    MPI_Type_commit(&dt);

    // Should be a valid derived type.
    ASSERT_FALSE(simplemc::mpi::type_is_predefined(dt));

    // Free should work.
    ASSERT_NO_THROW(simplemc::mpi::type_free(dt));

    // After free, should be null.
    ASSERT_EQ(dt, MPI_DATATYPE_NULL);
}

// Test RAII with derived type and auto-commit.
TEST(SimplemcMPIDatatype, RAIIDerivedType) {
    MPI_Datatype raw_dt = MPI_DATATYPE_NULL;

    {
        // Create a derived type (not committed).
        MPI_Type_contiguous(3, MPI_DOUBLE, &raw_dt);

        // Wrap with ownership - should auto-commit.
        simplemc::mpi::datatype dt { raw_dt };
        ASSERT_TRUE(dt);
        ASSERT_EQ(dt.size(), 3 * static_cast<int>(sizeof(double)));

        // Store copy of raw handle before wrapper goes out of scope.
        raw_dt = dt;
    }

    // After scope, the datatype should have been freed by the wrapper.
    // We can't easily test this since raw_dt still holds the old handle value,
    // but using it would be undefined behavior.
}

// Test attach policy doesn't free on destruction.
TEST(SimplemcMPIDatatype, AttachPolicyNoFree) {
    MPI_Datatype raw_dt = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(2, MPI_INT, &raw_dt);
    MPI_Type_commit(&raw_dt);

    {
        // Wrap with attach policy (no ownership).
        simplemc::mpi::datatype dt { raw_dt, simplemc::mpi::resource_policy::attach };
        ASSERT_TRUE(dt);
    }

    // After scope, the datatype should NOT have been freed.
    // We should still be able to use it.
    int size = 0;
    MPI_Type_size(raw_dt, &size);
    ASSERT_EQ(size, 2 * static_cast<int>(sizeof(int)));

    // Manually free.
    simplemc::mpi::type_free(raw_dt);
    ASSERT_EQ(raw_dt, MPI_DATATYPE_NULL);
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
