// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Throttled progress-printer callback for the simplemc::run loop.
 */

#ifndef SIMPLEMC_MC_PROGRESS_PRINTER_HPP
#define SIMPLEMC_MC_PROGRESS_PRINTER_HPP

#include <simplemc/mc/simulation_ctx.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-callbacks
 * @{
 */

/**
 * @brief Flexible generator for a one-line Monte Carlo progress report.
 *
 * @details Turns a simulation_ctx into a single human-readable status string: an optional
 * @ref prefix, an optional ASCII progress bar, the live step and time counters, and an overall
 * completion percentage. It also owns the notion of progress via fraction(), so the budget math lives
 * in one place and is shared with simplemc::progress_printer (which uses it to throttle).
 *
 * The report assumes a budgeted run — at least one of @ref max_steps or @ref max_time is set. With no
 * budget there is no fraction, so the bar and percentage are omitted and only the raw counters print.
 * The bar fraction (and percentage) is the larger of the step and time completion fractions, since
 * the run stops as soon as *either* budget is reached.
 *
 * All fields are plain options, so the type is an aggregate and can be brace-initialised, e.g.
 *
 * @code{.cpp}
 * progress_line{ .max_steps = 1000, .show_bar = true }
 * @endcode
 */
struct progress_line {
    /// Optional label prepended to each line (printed verbatim, followed by a space). Empty omits it.
    std::string prefix = "[progress]";

    /// Step budget: target number of MC steps to perform (0 disables it).
    std::uint64_t max_steps = 0;

    /// Time budget: target simulation runtime in seconds (0 disables it).
    double max_time = 0.0;

    /// If `true`, insert an ASCII progress bar (requires a non-zero step or time budget).
    bool show_bar = false;

    /// Number of cells in the progress bar.
    int bar_width = 30;

    /// Character for the filled portion of the bar.
    char fill_char = '#';

    /// Character for the empty portion of the bar.
    char empty_char = '-';

    /**
     * @brief Completion fraction of the run, in \f$ [0, 1] \f$.
     *
     * @details The larger of the step and time completion fractions over whichever budgets are
     * enabled, clamped to \f$ [0, 1] \f$. Returns `std::nullopt` when no budget is set (un-budgeted
     * run), which callers use to suppress the bar and percentage.
     *
     * @param ctx Simulation context.
     * @return Clamped completion fraction, or `std::nullopt` if neither budget is set.
     */
    [[nodiscard]] std::optional<double> fraction(const simulation_ctx& ctx) const {
        std::optional<double> frac;
        if (max_steps > 0) {
            frac = static_cast<double>(ctx.steps_done) / static_cast<double>(max_steps);
        }
        if (max_time > 0.0) {
            frac = std::max(frac.value_or(0.0), ctx.elapsed() / max_time);
        }
        if (frac) {
            frac = std::clamp(*frac, 0.0, 1.0);
        }
        return frac;
    }

    /**
     * @brief Render the status line for the current run state (no trailing newline).
     *
     * @param ctx Simulation context.
     * @return The formatted line.
     */
    [[nodiscard]] std::string render(const simulation_ctx& ctx) const {
        const std::optional<double> frac = fraction(ctx);

        // start with an optional prefix
        std::string line = prefix.empty() ? "" : fmt::format("{} ", prefix);

        // add optional progress bar
        if (show_bar && frac) {
            line += bar(*frac);
        }

        // add step and time counters (with their targets when budgeted)
        line += max_steps > 0 ? fmt::format(" steps {}/{}", ctx.steps_done, max_steps) :
                                fmt::format(" steps {}", ctx.steps_done);
        line += max_time > 0.0 ? fmt::format(", time {:.2f}/{:.2f} s", ctx.elapsed(), max_time) :
                                 fmt::format(", time {:.2f} s", ctx.elapsed());

        // final overall completion percentage
        if (frac) {
            line += fmt::format(" ({:.1f}%)", 100.0 * *frac);
        }
        return line;
    }

private:
    /// Build the `[####----]` bar for a fraction in \f$ [0, 1] \f$.
    [[nodiscard]] std::string bar(double frac) const {
        const int filled = static_cast<int>(std::lround(frac * bar_width));
        return fmt::format("[{}{}]", std::string(filled, fill_char), std::string(bar_width - filled, empty_char));
    }
};

/**
 * @brief Throttled progress-printer callback.
 *
 * @details A lightweight callable intended to be installed as a simplemc::run callback and invoked
 * once per cycle. It owns the callback mechanics and delegates all formatting to its @ref
 * progress_line.
 *
 * Throttling is progress-based: a line is emitted only once the completion fraction has advanced by
 * at least some specified fraction since the previous line. The first call always prints.
 *
 * Once the run reaches 100% one final line is always emitted, after which the printer goes quiet for
 * the rest of the run.
 *
 * The printer can be disabled entirely in the constructor (e.g. for rank-gated printing in a parallel
 * run).
 */
class progress_printer {
public:
    /**
     * @brief Construct a progress printer from a configured progress_line generator.
     *
     * @param line Line generator describing what to print (prefix, bar, budgets, etc.).
     * @param in_place If `true` (default), redraw intermediate lines in place with `\r`.
     * @param enabled If `false`, the printer emits nothing (caller-side rank gating).
     * @param throttle_frac Minimum fraction to complete between prints (e.g. \f$ 0.01 = 1\% \f$).
     * @param out Destination file handle (must outlive the printer). Defaults to `stdout`.
     */
    explicit progress_printer(progress_line line, bool in_place = true, bool enabled = true,
        double throttle_frac = 0.01, std::FILE* out = stdout) :
        line_(std::move(line)),
        inplace_(in_place),
        enabled_(enabled),
        throttle_frac_(throttle_frac),
        out_(out) {}

    /**
     * @brief Construct a progress printer from step/time budgets.
     *
     * @details Convenience constructor for the common case. It builds a default progress_line
     * carrying just the budgets and delegates to the line-generator constructor.
     *
     * @param max_steps Step budget: target number of steps (0 disables it).
     * @param max_time Time budget: target simulation runtime (0 disables it).
     * @param in_place If `true` (default), redraw intermediate lines in place with `\r`.
     * @param enabled If `false`, the printer emits nothing (caller-side rank gating).
     * @param throttle_frac Minimum fraction to complete between prints (e.g. \f$ 0.01 = 1\% \f$).
     * @param out Destination file handle (must outlive the printer). Defaults to `stdout`.
     */
    progress_printer(std::uint64_t max_steps, double max_time, bool in_place = true, bool enabled = true,
        double throttle_frac = 0.01, std::FILE* out = stdout) :
        progress_printer(
            progress_line { .max_steps = max_steps, .max_time = max_time }, in_place, enabled, throttle_frac, out) {}

    /**
     * @brief Emit one progress line for the current run state.
     *
     * @details Its behavior depends on the underlying progress_line, the parameters passed to the
     * constructor and the simulation context.
     *
     * @param ctx Simulation context.
     */
    void operator()(const simulation_ctx& ctx) {
        // skip disabled printers and any call after the final (100%) line
        if (!enabled_ || finished_) {
            return;
        }

        const std::optional<double> frac = line_.fraction(ctx);
        const double f = frac.value_or(0.0);

        // a budgeted run that reached 100% gets one final line, after which we go quiet
        finished_ = frac.has_value() && f >= 1.0;

        // progress throttle: only print once progress has advanced enough (first call + completion bypass)
        if (started_ && !finished_ && f - last_frac_ < throttle_frac_) {
            return;
        }

        // live redraws use '\r'; the final line (and non-inplace mode) commits with a newline
        fmt::print(out_, "{}{}", line_.render(ctx), (inplace_ && !finished_) ? '\r' : '\n');
        std::fflush(out_);

        // remember the fraction for the next throttle comparison
        last_frac_ = f;
        started_ = true;
    }

private:
    progress_line line_;
    bool inplace_;
    bool enabled_;
    double throttle_frac_;
    std::FILE* out_;
    double last_frac_ = 0.0;
    bool started_ = false;
    bool finished_ = false;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_PROGRESS_PRINTER_HPP
