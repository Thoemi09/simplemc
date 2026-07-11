// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#ifndef SIMPLEMC_TESTS_MC_TEST_UTILS_HPP
#define SIMPLEMC_TESTS_MC_TEST_UTILS_HPP

// Minimal MC update for tests: attempt() returns a fixed probability; counts accepts/rejects.
// Satisfies simplemc::mc_update; has no serialization hooks, so guarded save/load no-ops on it.
struct dummy_update {
    double prob = 1.0;
    int accepts = 0;
    int rejects = 0;
    double attempt() { return prob; }
    void accept() { ++accepts; }
    void reject() { ++rejects; }
};

// Minimal MC measurement for tests: counts measure() calls. Satisfies simplemc::mc_measurement.
struct dummy_measurement {
    int count = 0;
    void measure() { ++count; }
};

#endif // SIMPLEMC_TESTS_MC_TEST_UTILS_HPP
