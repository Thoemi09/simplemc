/**
 * @file
 * @brief MC update statistics.
 */

#ifndef SIMPLEMC_MC_UPDATE_STATS_HPP
#define SIMPLEMC_MC_UPDATE_STATS_HPP

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace simplemc {

namespace detail {

// Format the acceptance percentage of a counter window.
inline std::string acceptance_cell(std::uint64_t nprops, std::uint64_t naccs) {
    if (nprops == 0) {
        return "--";
    }
    return fmt::format("{:.1f}%", 100.0 * static_cast<double>(naccs) / static_cast<double>(nprops));
}

// Per-column widths of a table: the wider of the header and the widest cell.
template <std::size_t N>
std::array<std::size_t, N> column_widths(
    const std::array<std::string_view, N>& headers, const std::vector<std::array<std::string, N>>& rows) {
    std::array<std::size_t, N> widths {};
    for (std::size_t c = 0; c < N; ++c) {
        widths[c] = headers[c].size();
        for (const auto& row : rows) {
            widths[c] = std::max(widths[c], row[c].size());
        }
    }
    return widths;
}

} // namespace detail

/**
 * @addtogroup simplemc-mc-entities-updates
 * @{
 */

/**
 * @brief Metadata and statistics of an MC update.
 *
 * @details A plain aggregate bundling all metadata of a simplemc::update:
 *
 * - the unique @ref name identifying the update,
 * - the name of its inverse update (see @ref inv_name),
 * - the unnormalized selection @ref weight,
 * - the detailed-balance correction @ref ratio, and
 * - the counters tracking the number of proposals (@ref nprops), acceptances (@ref naccs) and
 * impossible signals (@ref nimps).
 */
struct update_stats {
    /**
     * @brief Identifier used in lookups and printed reports.
     */
    std::string name;

    /**
     * @brief Identifier of the inverse update. Equal to @ref name for self-inverse updates.
     */
    std::string inv_name;

    /**
     * @brief Unnormalized selection weight (\f$ w \geq 0 \f$).
     */
    double weight = 0.0;

    /**
     * @brief Detailed-balance correction multiplier applied by the simplemc::metropolis_kernel.
     */
    double ratio = 1.0;

    /**
     * @brief Number of proposals. Usually incremented by update::attempt().
     */
    std::uint64_t nprops = 0;

    /**
     * @brief Number of acceptances. Usually incremented by update::accept().
     */
    std::uint64_t naccs = 0;

    /**
     * @brief Number of impossible signals. Usually incremented by update::mark_impossible().
     */
    std::uint64_t nimps = 0;
};

/** @} */

/**
 * @relates simplemc::update_stats
 * @brief Print a set of simplemc::update_stats as a human-readable table.
 *
 * @param ss Set of update statistics to print, one row per update.
 * @param fp Destination file pointer (defaults to stdout).
 */
inline void print(const std::vector<update_stats>& ss, std::FILE* fp = stdout) {
    constexpr std::array<std::string_view, 8> headers { "name", "inverse", "weight", "ratio", "prop", "acc", "acc%",
        "imps" };

    // collect all cells up front so that every column can be sized to its widest entry
    std::vector<std::array<std::string, 8>> rows;
    rows.reserve(ss.size());
    for (const auto& s : ss) {
        rows.push_back({ s.name, s.inv_name, fmt::format("{:g}", s.weight), fmt::format("{:g}", s.ratio),
            fmt::format("{}", s.nprops), fmt::format("{}", s.naccs), detail::acceptance_cell(s.nprops, s.naccs),
            fmt::format("{}", s.nimps) });
    }
    const auto widths = detail::column_widths(headers, rows);

    // one table row: text columns left-aligned, numbers right-aligned, two-space gaps
    const auto table_line = [&widths](const auto& cells) {
        return fmt::format("{:<{}}  {:<{}}  {:>{}}  {:>{}}  {:>{}}  {:>{}}  {:>{}}  {:>{}}", cells[0], widths[0],
            cells[1], widths[1], cells[2], widths[2], cells[3], widths[3], cells[4], widths[4], cells[5], widths[5],
            cells[6], widths[6], cells[7], widths[7]);
    };

    // print the table
    const std::string header_line = table_line(headers);
    fmt::println(fp, "{}\n{}", header_line, std::string(header_line.size(), '-'));
    for (const auto& row : rows) {
        fmt::println(fp, "{}", table_line(row));
    }
}

/**
 * @relates simplemc::update_stats
 * @brief Print a simplemc::update_stats as a single-row table.
 *
 * @details Delegates to the vector overload, so a single update renders exactly like one row of the
 * set-level table.
 *
 * @param s Update statistics to print.
 * @param fp Destination file pointer (defaults to stdout).
 */
inline void print(const update_stats& s, std::FILE* fp = stdout) {
    print(std::vector { s }, fp);
}

} // namespace simplemc

#endif // SIMPLEMC_MC_UPDATE_STATS_HPP
