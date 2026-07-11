// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include <simplemc/mpi.hpp>

#include <atomic>
#include <thread>
#include <vector>

// Test that thread support can be queried.
TEST(SimplemcMPIThreading, QueryThreadSupport) {
    simplemc::mpi::communicator comm {};
    int provided = simplemc::mpi::query_thread();

    if (comm.rank() == 0) {
        fmt::print("Provided thread support level: ");
        switch (provided) {
            case MPI_THREAD_SINGLE: fmt::println("MPI_THREAD_SINGLE"); break;
            case MPI_THREAD_FUNNELED: fmt::println("MPI_THREAD_FUNNELED"); break;
            case MPI_THREAD_SERIALIZED: fmt::println("MPI_THREAD_SERIALIZED"); break;
            case MPI_THREAD_MULTIPLE: fmt::println("MPI_THREAD_MULTIPLE"); break;
            default: fmt::println("UNKNOWN ({})", provided);
        }
    }

    // thread support level should be valid
    ASSERT_TRUE(provided == MPI_THREAD_SINGLE || provided == MPI_THREAD_FUNNELED || provided == MPI_THREAD_SERIALIZED ||
        provided == MPI_THREAD_MULTIPLE);
}

// Test that is_main_thread returns true in the main thread.
TEST(SimplemcMPIThreading, IsMainThreadInMainThread) {
    bool is_main = simplemc::mpi::is_thread_main();
    ASSERT_TRUE(is_main);
}

// Test that is_main_thread returns false in spawned threads.
TEST(SimplemcMPIThreading, IsMainThreadInSpawnedThreads) {
    simplemc::mpi::communicator comm {};
    int provided = simplemc::mpi::query_thread();

    // only test with actual threading if MPI supports it
    if (provided >= MPI_THREAD_SERIALIZED) {
        const int num_threads = 4;
        std::vector<std::thread> threads {};
        std::atomic<int> main_thread_count { 0 };
        std::atomic<int> non_main_thread_count { 0 };

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&main_thread_count, &non_main_thread_count]() {
                bool is_main = simplemc::mpi::is_thread_main();
                if (is_main) {
                    main_thread_count++;
                } else {
                    non_main_thread_count++;
                }
            });
        }

        // wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }

        // all spawned threads should report they are not the main thread
        ASSERT_EQ(main_thread_count, 0);
        ASSERT_EQ(non_main_thread_count, num_threads);

        if (comm.rank() == 0) {
            fmt::println("Spawned {} threads, all correctly identified as non-main threads", num_threads);
        }
    } else {
        if (comm.rank() == 0) {
            fmt::println("Skipping spawned thread test (insufficient thread support)");
        }
    }
}

// Custom main function for MPI with threading support.
int main(int argc, char* argv[]) {
    int result = 0;

    // Initialize GoogleTest and MPI with thread support.
    ::testing::InitGoogleTest(&argc, argv);
    simplemc::mpi::environment env { argc, argv, MPI_THREAD_MULTIPLE };
    simplemc::mpi::communicator comm {};

    // Remove the default GoogleTest listener on all ranks except 0.
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    if (comm.rank() != 0) {
        delete listeners.Release(listeners.default_result_printer());
    }

    // Run the tests.
    result = RUN_ALL_TESTS();

    return result;
}
