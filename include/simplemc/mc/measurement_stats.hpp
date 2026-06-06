/**
 * @file
 * @brief Per-measurement metadata held by the Monte Carlo simulation driver.
 */

#ifndef SIMPLEMC_MC_MEASUREMENT_STATS_HPP
#define SIMPLEMC_MC_MEASUREMENT_STATS_HPP


#include <fmt/format.h>

#include <cstdio>
#include <span>
#include <string>

namespace simplemc {

/**
 * @addtogroup simplemc-mc
 * @{
 */

/**
 * @brief Plain aggregate of metadata for a single Monte Carlo measurement.
 *
 * @details Mirrors simplemc::update_stats in role: it carries identity for a measurement registered
 * with the simulation driver. At present the only field is a name, which exists for printed
 * reports and for future JSON-checkpoint keys. The struct is held in a vector parallel to the
 * driver's measurement vector — symmetric with how `update_stats` parallels the update vector.
 */
struct measurement_stats {
    /// Identifier used in printed reports.
    std::string name;

    /// Whether `measure()` is invoked on the corresponding measurement during each cycle.
    /// Toggled at runtime via simplemc::simulation::set_measurement_active.
    bool is_active = true;
};

/**
 * @brief Print a simplemc::measurement_stats as a one-line summary.
 *
 * @param fp Destination file handle.
 * @param s Metadata to print.
 */
inline void print(std::FILE* fp, const measurement_stats& s) {
    fmt::print(fp, "Measurement '{}' ({})\n", s.name, s.is_active ? "active" : "inactive");
}

/**
 * @brief Print a collection of simplemc::measurement_stats as a list.
 *
 * @param fp Destination file handle.
 * @param entries Metadata entries to print.
 */
inline void print(std::FILE* fp, std::span<const measurement_stats> entries) {
    fmt::print(fp,
        "============================\n"
        "MEASUREMENTS:\n"
        "============================\n");
    for (const auto& s : entries) {
        fmt::print(fp, "  {} ({})\n", s.name, s.is_active ? "active" : "inactive");
    }
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_MEASUREMENT_STATS_HPP
