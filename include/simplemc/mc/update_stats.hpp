/**
 * @file
 * @brief Per-update statistics tracked over the lifetime of a Monte Carlo update.
 */

#ifndef SIMPLEMC_MC_UPDATE_STATS_HPP
#define SIMPLEMC_MC_UPDATE_STATS_HPP

#include <simplemc/serialize/concepts.hpp>

#include <fmt/format.h>

#include <cstdint>
#include <cstdio>
#include <span>
#include <string>

namespace simplemc {

/**
 * @addtogroup simplemc-mc
 * @{
 */

/**
 * @brief Plain aggregate of statistics tracked for a single Monte Carlo update.
 *
 * @details One simplemc::update_stats instance lives alongside each simplemc::update registered with
 * the simulation driver. It carries both the configuration of the update (name, selection weight,
 * detailed-balance ratio, name of the inverse update) and the statistics that record how the update
 * behaved during the run.
 */
struct update_stats {
    /**
     * @brief Identifier used in printed reports.
     */
    std::string name;

    /**
     * @brief Identifier of the inverse update. Empty for self-inverse updates.
     */
    std::string inv_name;

    /**
     * @brief Unnormalized selection weight (\f$ w \geq 0 \f$).
     */
    double weight = 0.0;

    /**
     * @brief Detailed-balance correction multiplier.
     *
     * @details This ratio is applied by the simulation driver to the value returned by `attempt()`:
     *
     * - \f$ 1 \f$ for a self-inverse update, and
     * - \f$ w_{\text{inv}} / w \f$ for a pair of updates that are inverses of each other.
     */
    double ratio = 1.0;

    /**
     * @brief Number of times this update has been proposed in the current run.
     */
    std::uint64_t nprops = 0;

    /**
     * @brief Number of times this update has been accepted in the current run.
     */
    std::uint64_t naccs = 0;

    /**
     * @brief Number of times this update has been signaled as impossible in the current run.
     */
    std::uint64_t nimps = 0;

    /**
     * @brief Cumulative number of proposals across multiple runs.
     */
    std::uint64_t cumulative_nprops = 0;

    /**
     * @brief Cumulative number of acceptances across multiple runs.
     */
    std::uint64_t cumulative_naccs = 0;

    /**
     * @brief Cumulative number of impossible signals across multiple runs.
     */
    std::uint64_t cumulative_nimps = 0;
};

/**
 * @brief Reset the current-run counters to zero.
 *
 * @details The cumulative fields are left untouched.
 *
 * @param s Update statistics to reset.
 */
inline void reset_update_stats(update_stats& s) noexcept {
    s.nprops = 0;
    s.naccs = 0;
    s.nimps = 0;
}

/**
 * @brief Accumulate the current-run counters into the cumulative ones and reset the current ones.
 *
 * @details It calls simplemc::reset_update_stats after accumulating.
 *
 * @param s Update statistics to accumulate.
 */
inline void accumulate_update_stats(update_stats& s) noexcept {
    s.cumulative_nprops += s.nprops;
    s.cumulative_naccs += s.naccs;
    s.cumulative_nimps += s.nimps;
    reset_update_stats(s);
}

/**
 * @brief Print a collection of simplemc::update_stats as a table.
 *
 * @details Produces a header followed by one row per entry. Each row shows the name and the
 * current-run counters together with the acceptance ratio.
 *
 * @param fp Destination file handle.
 * @param entries Statistics to print.
 */
inline void print(std::FILE* fp, std::span<const update_stats> entries) {
    fmt::print(fp,
        "============================\n"
        "UPDATE STATISTICS:\n"
        "============================\n"
        "{:<16} {:>10} {:>10} {:>10} {:>10}\n",
        "Name", "nprops", "naccs", "nimps", "acc");
    for (const auto& s : entries) {
        const double acc = s.nprops > 0 ? static_cast<double>(s.naccs) / static_cast<double>(s.nprops) : 0.0;
        fmt::print(fp, "{:<16} {:>10} {:>10} {:>10} {:>10.4f}\n", s.name, s.nprops, s.naccs, s.nimps, acc);
    }
}

/**
 * @brief Serialize the persistent fields of simplemc::update_stats.
 *
 * @tparam S Serializer type.
 * @param s Serializer handle.
 * @param us Stats to write.
 */
template <serializer S>
void simplemc_save(S& s, const update_stats& us) {
    s.save_at("inv_name", us.inv_name);
    s.save_at("weight", us.weight);
    s.save_at("ratio", us.ratio);
    s.save_at("cumulative_nprops", us.cumulative_nprops);
    s.save_at("cumulative_naccs", us.cumulative_naccs);
    s.save_at("cumulative_nimps", us.cumulative_nimps);
}

/**
 * @brief Deserialize the persistent fields of simplemc::update_stats.
 *
 * @tparam S Serializer type.
 * @param s Serializer handle.
 * @param us Stats to read into.
 */
template <serializer S>
void simplemc_load(const S& s, update_stats& us) {
    s.load_at("inv_name", us.inv_name);
    s.load_at("weight", us.weight);
    s.load_at("ratio", us.ratio);
    s.load_at("cumulative_nprops", us.cumulative_nprops);
    s.load_at("cumulative_naccs", us.cumulative_naccs);
    s.load_at("cumulative_nimps", us.cumulative_nimps);
}

/**
 * @brief Serialize the user-input config of simplemc::update_stats.
 *
 * @tparam S Serializer type.
 * @param s Serializer handle.
 * @param us Stats to write.
 */
template <serializer S>
void simplemc_save_input_config(S& s, const update_stats& us) {
    s.save_at("weight", us.weight);
}

/**
 * @brief Deserialize the user-input config of simplemc::update_stats.
 *
 * @tparam S Serializer type.
 * @param s Serializer handle.
 * @param us Stats to read into.
 */
template <serializer S>
void simplemc_load_input_config(const S& s, update_stats& us) {
    s.try_load_at("weight", us.weight);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_UPDATE_STATS_HPP
