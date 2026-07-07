#include <fmt/base.h>
#include <gtest/gtest.h>
#include <simplemc/config.hpp>
#include <simplemc/utils.hpp>
#include <simplemc/utils/fmt_complex.hpp>

#include <array>
#include <cstdio>
#include <filesystem>
#include <string>

namespace {

// Read a whole file into a std::string.
std::string read_file(const std::filesystem::path& path) {
    auto file = simplemc::open_file(path.string(), "r");
    std::string out;
    std::array<char, 256> buf {};
    while (auto n = std::fread(buf.data(), 1, buf.size(), file.get())) {
        out.append(buf.data(), n);
    }
    return out;
}

// Remove and count leftover temp files ("<filename>.tmp*", the suffix carries a PID) next to `path`.
// Used to verify that a failed atomic_file_write cleans up after itself.
int remove_tmp_siblings(const std::filesystem::path& path) {
    const auto prefix = path.filename().string() + ".tmp";
    int n = 0;
    for (const auto& e : std::filesystem::directory_iterator { path.parent_path() }) {
        if (e.path().filename().string().starts_with(prefix)) {
            std::filesystem::remove(e.path());
            ++n;
        }
    }
    return n;
}

} // namespace

// Test if the config header is generated correctly.
TEST(SimplemcUtils, ConfigHeader) {
    fmt::println("VERSION: {}", SIMPLEMC_VERSION);
#ifdef SIMPLEMC_WITH_H5
    fmt::println("SIMPLEMC_WITH_H5: True");
#else
    fmt::println("SIMPLEMC_WITH_H5: False");
#endif // SIMPLEMC_WITH_H5
#ifdef SIMPLEMC_USE_STD_RANGES
    fmt::println("SIMPLEMC_USE_STD_RANGES: True");
#else
    fmt::println("SIMPLEMC_USE_STD_RANGES: False");
#endif // SIMPLEMC_USE_STD_RANGES
}

// Test concepts.
TEST(SimplemcUtils, Concepts) {
    ASSERT_TRUE(simplemc::integer_only<int>);
    ASSERT_FALSE(simplemc::integer_only<char>);
    ASSERT_TRUE(simplemc::integer_only<std::uint64_t>);
    ASSERT_FALSE(simplemc::integer_only<std::int8_t>);
    ASSERT_TRUE(simplemc::nd_order<simplemc::row_major>);
    ASSERT_TRUE(simplemc::nd_order<simplemc::column_major>);
    ASSERT_FALSE(simplemc::nd_order<int>);
    ASSERT_TRUE(simplemc::double_or_complex<double>);
    ASSERT_TRUE(simplemc::double_or_complex<std::complex<double>>);
    ASSERT_FALSE(simplemc::double_or_complex<float>);
    ASSERT_FALSE(simplemc::double_or_complex<std::complex<float>>);
}

// Test specialized formatter for std::complex.
TEST(SimplemcUtils, ComplexFormatter) {
    std::complex<double> z(1.0003, 2.00041291823);
    fmt::print("z = {:^30.15}\n", z);
}

// Test file IO with raw functions.
TEST(SimplemcUtils, FileIORaw) {
    auto fp = simplemc::detail::open_file_raw("test_file_raw.txt", "w");
    fmt::print(fp, "This is test file #{}\n", 1);
    simplemc::detail::close_file_raw(fp, "test_file_raw.txt");
}

// Test file IO with RAII wrapper (recommended API).
TEST(SimplemcUtils, FileIO) {
    auto file = simplemc::open_file("test_file.txt", "w");
    fmt::print(file, "This is test file #{}\n", 1);
    ASSERT_EQ(file.name(), "test_file.txt");
}

// Test file_handle direct construction and advanced features.
TEST(SimplemcUtils, FileHandleRAII) {
    {
        simplemc::file_handle file("test_file_raii.txt", "w");
        fmt::print(file.get(), "This is RAII test file #{}\n", 2);
        ASSERT_EQ(file.name(), "test_file_raii.txt");
    }

    // test move semantics
    simplemc::file_handle file1("test_file_move.txt", "w");
    fmt::print(file1.get(), "Line 1\n");

    simplemc::file_handle file2(std::move(file1));
    ASSERT_EQ(file1.get(), nullptr);
    ASSERT_NE(file2.get(), nullptr);
    ASSERT_EQ(file2.name(), "test_file_move.txt");
    fmt::print(file2, "Line 2\n");

    // test explicit close (and verify no double-close in destructor)
    file2.close();
    ASSERT_EQ(file2.get(), nullptr);
}

// Test that atomic_file_write produces the file and leaves no temp sibling behind.
TEST(SimplemcUtils, AtomicFileWriteCreatesFileAndCleansTmp) {
    const auto path = std::filesystem::temp_directory_path() / "simplemc_test_atomic.txt";
    std::filesystem::remove(path);
    remove_tmp_siblings(path);

    simplemc::atomic_file_write(path, [](const std::filesystem::path& tmp) {
        auto file = simplemc::open_file(tmp.string(), "w");
        fmt::print(file, "content\n");
    });
    ASSERT_TRUE(std::filesystem::exists(path));
    EXPECT_EQ(read_file(path), "content\n");
    EXPECT_EQ(remove_tmp_siblings(path), 0);

    std::filesystem::remove(path);
}

// Test that a failing write leaves the previous file intact and removes the temp file.
TEST(SimplemcUtils, AtomicFileWriteFailureKeepsPreviousFileAndRemovesTmp) {
    const auto path = std::filesystem::temp_directory_path() / "simplemc_test_atomic_fail.txt";
    std::filesystem::remove(path);
    remove_tmp_siblings(path);

    simplemc::atomic_file_write(path, [](const std::filesystem::path& tmp) {
        auto file = simplemc::open_file(tmp.string(), "w");
        fmt::print(file, "old\n");
    });

    const auto failing_write = [](const std::filesystem::path& tmp) {
        auto file = simplemc::open_file(tmp.string(), "w");
        fmt::print(file, "new\n");
        throw simplemc::simplemc_exception("intentional write failure");
    };
    EXPECT_THROW(simplemc::atomic_file_write(path, failing_write), simplemc::simplemc_exception);
    EXPECT_EQ(read_file(path), "old\n");
    EXPECT_EQ(remove_tmp_siblings(path), 0);

    std::filesystem::remove(path);
}

// Test that copy_existing seeds the temp file with the current content (append-style writers).
TEST(SimplemcUtils, AtomicFileWriteCopyExistingSeedsTmpWithCurrentContent) {
    const auto path = std::filesystem::temp_directory_path() / "simplemc_test_atomic_copy.txt";
    std::filesystem::remove(path);

    simplemc::atomic_file_write(path, [](const std::filesystem::path& tmp) {
        auto file = simplemc::open_file(tmp.string(), "w");
        fmt::print(file, "old\n");
    });

    simplemc::atomic_file_write(
        path,
        [](const std::filesystem::path& tmp) {
            EXPECT_EQ(read_file(tmp), "old\n");
            auto file = simplemc::open_file(tmp.string(), "a");
            fmt::print(file, "new\n");
        },
        /*copy_existing=*/true);
    EXPECT_EQ(read_file(path), "old\nnew\n");

    std::filesystem::remove(path);
}

// Test that copy_existing is a no-op when there is no prior file at the target path.
TEST(SimplemcUtils, AtomicFileWriteCopyExistingWithNoPriorFile) {
    const auto path = std::filesystem::temp_directory_path() / "simplemc_test_atomic_nofile.txt";
    std::filesystem::remove(path);

    simplemc::atomic_file_write(
        path,
        [](const std::filesystem::path& tmp) {
            auto file = simplemc::open_file(tmp.string(), "w");
            fmt::print(file, "content\n");
        },
        /*copy_existing=*/true);
    EXPECT_EQ(read_file(path), "content\n");

    std::filesystem::remove(path);
}

// Test file_handle construction from existing FILE* pointer.
TEST(SimplemcUtils, FileHandleFromPointer) {
    // test with a file opened via raw function
    auto fp = simplemc::detail::open_file_raw("test_from_ptr.txt", "w");
    {
        simplemc::file_handle file(fp, "test_from_ptr.txt");
        ASSERT_EQ(file.get(), fp);
        ASSERT_EQ(file.name(), "test_from_ptr.txt");
        fmt::print(file, "Wrapped existing file pointer\n");
    } // file closed here by file_handle destructor

    // test with stdout (should not be closed)
    {
        simplemc::file_handle stdout_handle(stdout, "stdout");
        ASSERT_EQ(stdout_handle.get(), stdout);
        ASSERT_EQ(stdout_handle.name(), "stdout");
        fmt::print(stdout_handle, "This goes to stdout via file_handle\n");
    } // stdout not closed due to special handling

    // test with nullptr
    {
        simplemc::file_handle null_handle(nullptr, "null");
        ASSERT_EQ(null_handle.get(), nullptr);
        ASSERT_EQ(null_handle.name(), "null");
    } // no-op close
}
