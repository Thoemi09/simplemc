/**
 * @file timer.h
 * @brief Timer built on std::chrono.
 */

#ifndef SIMPLEMC_UTILS_TIMER_H
#define SIMPLEMC_UTILS_TIMER_H

#include <chrono>

namespace simplemc {

/**
 * @brief Type aliases for different std::chrono::duration types.
 */
struct duration {
public:
    using hour = std::chrono::duration<double, std::ratio<3600>>;
    using min = std::chrono::duration<double, std::ratio<60>>;
    using sec = std::chrono::duration<double, std::ratio<1>>;
    using millisec = std::chrono::duration<double, std::milli>;
    using microsec = std::chrono::duration<double, std::micro>;
    using nanosec = std::chrono::duration<double, std::nano>;
};

/**
 * @brief Elapsed time between two time points.
 *
 * @details The third parameter specifies the type of resulting duration.
 * It is only used for type deduction.
 *
 *     auto measured_time = time_passed(tp1, tp2, duration::millisec{});
 *
 * Note that this implementation is different from std::chrono::duration_cast,
 * where the resulting duration type has to be given explictly as a template
 * parameter. This enables us to use a default duration (duration::sec).
 *
 * @tparam C Clock type
 * @tparam D1 Duration type of first point in time
 * @tparam D2 Duration type of second point in time
 * @tparam D Duration type of result (default: duration::sec)
 *
 * @param time_point1 First point in time
 * @param time_point2 Second point in time
 * @param to_duration Instance of the duration type of the result
 * @return Time difference between the two points in time. If t1 > t2, then the
 * result is negative.
 */
template <typename C, typename D1, typename D2, typename D = duration::sec>
auto time_passed(const std::chrono::time_point<C, D1>& time_point1, const std::chrono::time_point<C, D2>& time_point2,
    [[maybe_unused]] const D& to_duration = D {}) {
    return std::chrono::duration_cast<D>(time_point2 - time_point1).count();
}

/**
 * @brief Measure time and record points in time.
 *
 * @details This timer is based on the clocks of std::chrono. The clock can be
 * specified via the template parameter. It can store three different points in
 * time (std::chrono::time_point): start, stop and interim.
 *
 *     timer t;
 *     ...
 *     t.interim();
 *     ...
 *     t.stop();
 *     auto since_start_in_sec = time_passed(t.get_start(), t.get_stop());
 *     auto since_interim_in_min = time_passed(t.get_interim(), t.get_stop(), duration::min{});
 *
 * @tparam Clock Clock from std::chrono (default: std::chrono::steady_clock)
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
     * @return Time point representing the current time.
     */
    static std::chrono::time_point<clock_type> now() { return clock_type::now(); }

    /**
     * @brief Default constructor. All three members are set to the current time point.
     */
    timer() : start_(now()), stop_(start_), interim_(start_) {}

    /**
     * @brief Get starting time point.
     *
     * @return Starting time point.
     */
    std::chrono::time_point<clock_type> start_time() const { return start_; }

    /**
     * @brief Get stopping time point.
     *
     * @return Stopping ime point.
     */
    std::chrono::time_point<clock_type> stop_time() const { return stop_; }

    /**
     * @brief Get interim time point.
     *
     * @return Interim time point.
     */
    std::chrono::time_point<clock_type> interim_time() const { return interim_; }

    /**
     * @brief Set starting time point to now().
     */
    void start() { start_ = now(); }

    /**
     * @brief Set stopping time point to now().
     */
    void stop() { stop_ = now(); }

    /**
     * @brief Set interim time point to now().
     */
    void interim() { interim_ = now(); }

private:
    std::chrono::time_point<clock_type> start_;
    std::chrono::time_point<clock_type> stop_;
    std::chrono::time_point<clock_type> interim_;
};

} // namespace simplemc

#endif // SIMPLEMC_UTILS_TIMER_H
