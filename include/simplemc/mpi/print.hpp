// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Root-gated formatted printing over an MPI communicator.
 */

#ifndef SIMPLEMC_MPI_PRINT_HPP
#define SIMPLEMC_MPI_PRINT_HPP

#include <simplemc/mpi/communicator.hpp>

#include <fmt/base.h>

#include <cstdio>
#include <utility>

namespace simplemc::mpi {

/**
 * @addtogroup simplemc-mpi-utils
 * @{
 */

/**
 * @brief Print formatted output on a single MPI rank.
 *
 * @details `fmt::print`-style formatting that emits only on rank `root` of the given communicator. On
 * all other ranks the call is a no-op.
 *
 * It replaces the common pattern
 *
 * @code{.cpp}
 * if (comm.rank() == root) { fmt::print(...); }
 * @endcode
 *
 * @tparam Args Format argument types.
 * @param comm simplemc::mpi::communicator object.
 * @param fp Destination file pointer.
 * @param root Rank that prints.
 * @param f Format string.
 * @param args Format arguments.
 */
template <typename... Args>
void print(const communicator& comm, std::FILE* fp, int root, fmt::format_string<Args...> f, Args&&... args) {
    if (comm.rank() == root) {
        fmt::print(fp, f, std::forward<Args>(args)...);
    }
}

/**
 * @brief Print formatted output on a single MPI rank to `stdout`.
 *
 * @details It calls simplemc::mpi::print with `fp = stdout`.
 *
 * @tparam Args Format argument types.
 * @param comm simplemc::mpi::communicator object.
 * @param root Rank that prints.
 * @param f Format string.
 * @param args Format arguments.
 */
template <typename... Args>
void print(const communicator& comm, int root, fmt::format_string<Args...> f, Args&&... args) {
    print(comm, stdout, root, f, std::forward<Args>(args)...);
}

/**
 * @brief Print formatted output on rank \f$ 0 \f$.
 *
 * @details It calls simplemc::mpi::print with `root = 0`.
 *
 * @tparam Args Format argument types.
 * @param comm simplemc::mpi::communicator object.
 * @param fp Destination file pointer.
 * @param f Format string.
 * @param args Format arguments.
 */
template <typename... Args>
void print(const communicator& comm, std::FILE* fp, fmt::format_string<Args...> f, Args&&... args) {
    print(comm, fp, 0, f, std::forward<Args>(args)...);
}

/**
 * @brief Print formatted output on the rank \f$ 0 \f$ to `stdout`.
 *
 * @details It calls simplemc::mpi::print with `fp = stdout` and `root = 0`.
 *
 * @tparam Args Format argument types.
 * @param comm simplemc::mpi::communicator object.
 * @param f Format string.
 * @param args Format arguments.
 */
template <typename... Args>
void print(const communicator& comm, fmt::format_string<Args...> f, Args&&... args) {
    print(comm, stdout, 0, f, std::forward<Args>(args)...);
}

/**
 * @brief Print formatted output followed by a newline on a single MPI rank.
 *
 * @details `fmt::println`-style counterpart of simplemc::mpi::print: it emits only on rank `root`
 * of the given communicator and is a no-op on all other ranks.
 *
 * @tparam Args Format argument types.
 * @param comm simplemc::mpi::communicator object.
 * @param fp Destination file pointer.
 * @param root Rank that prints.
 * @param f Format string.
 * @param args Format arguments.
 */
template <typename... Args>
void println(const communicator& comm, std::FILE* fp, int root, fmt::format_string<Args...> f, Args&&... args) {
    if (comm.rank() == root) {
        fmt::println(fp, f, std::forward<Args>(args)...);
    }
}

/**
 * @brief Print formatted output followed by a newline on a single MPI rank to `stdout`.
 *
 * @details It calls simplemc::mpi::println with `fp = stdout`.
 *
 * @tparam Args Format argument types.
 * @param comm simplemc::mpi::communicator object.
 * @param root Rank that prints.
 * @param f Format string.
 * @param args Format arguments.
 */
template <typename... Args>
void println(const communicator& comm, int root, fmt::format_string<Args...> f, Args&&... args) {
    println(comm, stdout, root, f, std::forward<Args>(args)...);
}

/**
 * @brief Print formatted output followed by a newline on rank \f$ 0 \f$.
 *
 * @details It calls simplemc::mpi::println with `root = 0`.
 *
 * @tparam Args Format argument types.
 * @param comm simplemc::mpi::communicator object.
 * @param fp Destination file pointer.
 * @param f Format string.
 * @param args Format arguments.
 */
template <typename... Args>
void println(const communicator& comm, std::FILE* fp, fmt::format_string<Args...> f, Args&&... args) {
    println(comm, fp, 0, f, std::forward<Args>(args)...);
}

/**
 * @brief Print formatted output followed by a newline on rank \f$ 0 \f$ to `stdout`.
 *
 * @details It calls simplemc::mpi::println with `fp = stdout` and `root = 0`.
 *
 * @tparam Args Format argument types.
 * @param comm simplemc::mpi::communicator object.
 * @param f Format string.
 * @param args Format arguments.
 */
template <typename... Args>
void println(const communicator& comm, fmt::format_string<Args...> f, Args&&... args) {
    println(comm, stdout, 0, f, std::forward<Args>(args)...);
}

/** @} */

} // namespace simplemc::mpi

#endif // SIMPLEMC_MPI_PRINT_HPP
