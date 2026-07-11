// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Timer built on top of `std::chrono`.
 */

#ifndef SIMPLEMC_UTILS_TIMER_HPP
#define SIMPLEMC_UTILS_TIMER_HPP

#include <chrono>

namespace simplemc {

/**
 * @addtogroup simplemc-utils-timer
 * @{
 */

/**
 * @brief Convenient type aliases for different `std::chrono::duration` types.
 */
struct duration {
    /// Duration type for hours.
    using hour = std::chrono::duration<double, std::ratio<3600>>;

    /// Duration type for minutes.
    using min = std::chrono::duration<double, std::ratio<60>>;

    /// Duration type for seconds.
    using sec = std::chrono::duration<double, std::ratio<1>>;

    /// Duration type for milliseconds.
    using millisec = std::chrono::duration<double, std::milli>;

    /// Duration type for microseconds.
    using microsec = std::chrono::duration<double, std::micro>;

    /// Duration type for nanoseconds.
    using nanosec = std::chrono::duration<double, std::nano>;
};

/**
 * @brief Get the elapsed time between two time points.
 *
 * @details The third parameter specifies the type of the resulting duration. It is only used for type
 * deduction.
 *
 * @code{.cpp}
 * auto measured_time = time_passed(tp1, tp2, duration::millisec{});
 * @endcode
 *
 * Here, `tp1` and `tp2` are two `std::chrono::time_point` instances.
 *
 * @note This implementation is different from `std::chrono::duration_cast`, where the resulting
 * duration type has to be given explictly as a template parameter.
 *
 * @tparam C Clock type.
 * @tparam D1 Duration type of first point in time.
 * @tparam D2 Duration type of second point in time.
 * @tparam D Duration type of result (default: simplemc::duration::sec).
 *
 * @param time_point1 First point in time.
 * @param time_point2 Second point in time.
 * @param to_duration Instance of the duration type of the result (used for type deduction).
 * @return Time difference between the two points in time (can be negative).
 */
template <typename C, typename D1, typename D2, typename D = duration::sec>
[[nodiscard]] auto time_passed(const std::chrono::time_point<C, D1>& time_point1,
    const std::chrono::time_point<C, D2>& time_point2, [[maybe_unused]] const D& to_duration = D {}) {
    return std::chrono::duration_cast<D>(time_point2 - time_point1).count();
}

/**
 * @brief Measure time and record points in time.
 *
 * @details This timer is based on the clocks of `std::chrono`. The clock can be specified via the
 * template parameter.
 *
 * It can store three different points in time (`std::chrono::time_point`): start, stop and interim.
 *
 * See @ref tut_utils_1 for a detailed example.
 *
 * @tparam Clock Clock type from `std::chrono` (default: `std::chrono::steady_clock`).
 */
template <typename Clock = std::chrono::steady_clock>
class timer {
public:
    /**
     * @brief Type of clock being used in this timer instance.
     */
    using clock_type = Clock;

    /**
     * @brief Get current point in time.
     *
     * @return `std::chrono::time_point<clock_type>` representing the current time.
     */
    [[nodiscard]] static auto now() { return clock_type::now(); }

    /**
     * @brief Default constructor.
     *
     * @details All three members are set to the current time point by calling now().
     */
    timer() : start_(now()), stop_(start_), interim_(start_) {}

    /**
     * @brief Get starting time point.
     *
     * @return `std::chrono::time_point<clock_type>` representing the starting time point.
     */
    [[nodiscard]] auto start_time() const noexcept { return start_; }

    /**
     * @brief Get stopping time point.
     *
     * @return `std::chrono::time_point<clock_type>` representing the stopping time point.
     */
    [[nodiscard]] auto stop_time() const noexcept { return stop_; }

    /**
     * @brief Get interim time point.
     *
     * @return `std::chrono::time_point<clock_type>` representing the interim time point.
     */
    [[nodiscard]] auto interim_time() const noexcept { return interim_; }

    /**
     * @brief Set starting time point by calling now().
     */
    void start() { start_ = now(); }

    /**
     * @brief Set stopping time point by calling now().
     */
    void stop() { stop_ = now(); }

    /**
     * @brief Set interim time point by calling now().
     */
    void interim() { interim_ = now(); }

    /**
     * @brief Reset all time points to current time.
     *
     * @details This method resets the start, stop, and interim time points to the current time,
     * effectively resetting the timer without creating a new instance.
     */
    void reset() { start_ = stop_ = interim_ = now(); }

    /**
     * @brief Get elapsed time between starting the timer and the current time.
     *
     * @details This is a convenience method that returns the time difference between the start and
     * the current time points. It is equivalent to calling
     *
     * @code
     * time_passed(t.start_time(), t.now(), to_duration)
     * @endcode
     *
     * where `t` is an instance of `simplemc::timer`.
     *
     * @tparam D Duration type of result (default: simplemc::duration::sec).
     *
     * @param to_duration Instance of the duration type of the result (used for type deduction).
     * @return Time difference between start and now().
     */
    template <typename D = duration::sec>
    [[nodiscard]] auto elapsed([[maybe_unused]] const D& to_duration = D {}) const {
        return time_passed(start_, now(), to_duration);
    }

private:
    std::chrono::time_point<clock_type> start_;
    std::chrono::time_point<clock_type> stop_;
    std::chrono::time_point<clock_type> interim_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_UTILS_TIMER_HPP
