/**
 * @file
 * @brief Per-measurement metadata held by the Monte Carlo simulation driver.
 */

#ifndef SIMPLEMC_MC_MEASUREMENT_STATS_HPP
#define SIMPLEMC_MC_MEASUREMENT_STATS_HPP

#include <simplemc/serialize/concepts.hpp>

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
 * @details One simplemc::measurement_stats instance lives alongside each simplemc::measurement
 * registered with the simulation driver. It carries the name of the measurement as well as a boolean
 * flag that indicates whether the measurement is active.
 */
struct measurement_stats {
    /**
     * @brief Identifier used in printed reports.
     */
    std::string name;

    /**
     * @brief Whether `measure()` is invoked on the corresponding measurement during each cycle.
     */
    bool is_active = true;
};

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

/**
 * @brief Serialize the persistent fields of simplemc::measurement_stats.
 *
 * @tparam S Serializer type.
 * @param s Serializer handle.
 * @param ms Stats to write.
 */
template <serializer S>
void simplemc_save(S& s, const measurement_stats& ms) {
    s.save_at("is_active", ms.is_active);
}

/**
 * @brief Deserialize the persistent fields of simplemc::measurement_stats.
 *
 * @tparam S Serializer type.
 * @param s Serializer handle.
 * @param ms Stats to read into.
 */
template <serializer S>
void simplemc_load(const S& s, measurement_stats& ms) {
    s.load_at("is_active", ms.is_active);
}

/**
 * @brief Serialize the user-input config of simplemc::measurement_stats.
 *
 * @tparam S Serializer type.
 * @param s Serializer handle.
 * @param ms Stats to write.
 */
template <serializer S>
void simplemc_save_input_config(S& s, const measurement_stats& ms) {
    s.save_at("is_active", ms.is_active);
}

/**
 * @brief Deserialize the user-input config of simplemc::measurement_stats.
 *
 * @tparam S Serializer type.
 * @param s Serializer handle.
 * @param ms Stats to read into.
 */
template <serializer S>
void simplemc_load_input_config(const S& s, measurement_stats& ms) {
    s.try_load_at("is_active", ms.is_active);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_MEASUREMENT_STATS_HPP
