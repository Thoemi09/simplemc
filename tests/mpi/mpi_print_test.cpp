// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include "../gtest-mpi-listener.hpp"

#include <simplemc/mpi.hpp>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <string>

namespace {

// Read back everything written to a temporary file.
std::string read_all(std::FILE* fp) {
    std::rewind(fp); // NOLINT
    std::string content;
    for (int c = std::fgetc(fp); c != EOF; c = std::fgetc(fp)) {
        content.push_back(static_cast<char>(c));
    }
    return content;
}

} // namespace

// Test fixture for MPI print tests.
class SimplemcMPIPrint : public ::testing::Test {
protected:
    void SetUp() override {
        rank = comm.rank();
        size = comm.size();
    }
    simplemc::mpi::communicator comm;
    int rank { 0 };
    int size { 0 };
};

// Test that the default forms emit on rank 0 only and forward the format arguments.
TEST_F(SimplemcMPIPrint, DefaultFormsPrintOnRankZeroOnly) {
    ::testing::internal::CaptureStdout();
    simplemc::mpi::print(comm, "value = {}", 42);
    simplemc::mpi::println(comm, " on rank {}", rank);
    const std::string out = ::testing::internal::GetCapturedStdout();

    if (rank == 0) {
        EXPECT_EQ(out, "value = 42 on rank 0\n");
    } else {
        EXPECT_TRUE(out.empty());
    }
}

// Test that a non-zero root argument moves the emitting rank.
TEST_F(SimplemcMPIPrint, RootArgumentMovesEmittingRank) {
    const int root = size - 1;
    ::testing::internal::CaptureStdout();
    simplemc::mpi::print(comm, root, "value = {}", 1.5);
    simplemc::mpi::println(comm, root, " on rank {}", rank);
    const std::string out = ::testing::internal::GetCapturedStdout();

    if (rank == root) {
        EXPECT_EQ(out, fmt::format("value = 1.5 on rank {}\n", root));
    } else {
        EXPECT_TRUE(out.empty());
    }
}

// Test that the file-pointer overloads write to the given file on the root rank only.
TEST_F(SimplemcMPIPrint, FilePointerOverloadWritesOnRootOnly) {
    std::FILE* fp = std::tmpfile();
    ASSERT_NE(fp, nullptr);

    simplemc::mpi::print(comm, fp, "a = {}", 1);
    simplemc::mpi::println(comm, fp, ", b = {}", 2);

    if (rank == 0) {
        EXPECT_EQ(read_all(fp), "a = 1, b = 2\n");
    } else {
        EXPECT_TRUE(read_all(fp).empty());
    }
    std::fclose(fp);
}

// Test the full form with both a file pointer and a non-zero root rank.
TEST_F(SimplemcMPIPrint, FullFormWritesOnGivenRootOnly) {
    const int root = size - 1;
    std::FILE* fp = std::tmpfile();
    ASSERT_NE(fp, nullptr);

    simplemc::mpi::print(comm, fp, root, "a = {}", 1);
    simplemc::mpi::println(comm, fp, root, ", b = {}", 2);

    if (rank == root) {
        EXPECT_EQ(read_all(fp), "a = 1, b = 2\n");
    } else {
        EXPECT_TRUE(read_all(fp).empty());
    }
    std::fclose(fp);
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
