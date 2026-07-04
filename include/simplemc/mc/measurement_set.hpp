/**
 * @file
 * @brief Set of MC measurements.
 */

#ifndef SIMPLEMC_MC_MEASUREMENT_SET_HPP
#define SIMPLEMC_MC_MEASUREMENT_SET_HPP

#include <simplemc/mc/measurement.hpp>
#include <simplemc/mc/tuple_set.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>

#include <cstddef>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-entities-measurements
 * @{
 */

/**
 * @brief Set of MC measurements.
 *
 * @details The set stores its simplemc::measurement entries in a `std::tuple` (via
 * simplemc::tuple_set), so the measurement *types* are fixed at construction. It also holds a cache of
 * indices for those currently flagged active, rebuilt from the measurement::is_active flags by
 * refresh_active(). Simulation drivers call measure_all() to invoke `measure()` on every active entry.
 *
 * Registration is construction-time (build the whole set in one initializer, deducing the types via
 * CTAD); the active flags stay mutable at runtime via set_active(). Lookup and typed access are
 * provided by the base simplemc::tuple_set.
 *
 * @tparam Ms User measurement types (each satisfying simplemc::mc_measurement).
 */
template <mc_measurement... Ms>
class measurement_set : public tuple_set<measurement<Ms>...> {
public:
    /**
     * @brief Construct the set from its measurements.
     *
     * @param ms Measurements to store, in order.
     */
    explicit measurement_set(measurement<Ms>... ms) : tuple_set<measurement<Ms>...> { std::move(ms)... } {}

    /**
     * @brief Set the active state of a registered measurement.
     *
     * @details It throws a simplemc::simplemc_exception if the name is not registered.
     *
     * @param name Name of the measurement to toggle.
     * @param is_active New active state.
     */
    void set_active(std::string_view name, bool is_active) {
        const auto idx = this->find(name);
        if (!idx) {
            throw simplemc_exception(fmt::format("measurement '{}' is not registered", name));
        }
        this->visit_at(*idx, [&](auto& m) { m.is_active = is_active; });
    }

    /**
     * @brief Cached indices of currently-active measurements.
     *
     * @return Span over the indices of currently-active measurements.
     */
    [[nodiscard]] std::span<const std::size_t> active_indices() const noexcept { return active_; }

    /**
     * @brief Rebuild the active set cache from the measurement::is_active flags.
     */
    void refresh_active() {
        active_.clear();
        active_.reserve(this->size());
        std::size_t i = 0;
        this->for_each([&](const auto& m) {
            if (m.is_active) {
                active_.push_back(i);
            }
            ++i;
        });
    }

    /**
     * @brief Clear the active set cache.
     */
    void clear_active() noexcept { active_.clear(); }

    /**
     * @brief Invoke `measure()` on every entry in the cached active set.
     */
    void measure_all() {
        for (std::size_t i : active_) {
            this->visit_at(i, [](auto& m) { m.measure(); });
        }
    }

    /**
     * @brief Serialize a simplemc::measurement_set: every measurement, keyed by name.
     */
    template <serializer S>
    friend void simplemc_save(S& s, const measurement_set& ms) {
        ms.save_entries(s);
    }

    /**
     * @brief Deserialize a simplemc::measurement_set: every measurement, keyed by name.
     */
    template <serializer S>
    friend void simplemc_load(const S& s, measurement_set& ms) {
        ms.load_entries(s, "measurement");
    }

    /**
     * @brief Serialize the user-input config of a simplemc::measurement_set.
     */
    template <serializer S>
    friend void simplemc_save_input_config(S& s, const measurement_set& ms) {
        ms.save_input_config_entries(s);
    }

    /**
     * @brief Deserialize the user-input config of a simplemc::measurement_set.
     */
    template <serializer S>
    friend void simplemc_load_input_config(const S& s, measurement_set& ms) {
        ms.load_input_config_entries(s, "measurement");
    }

    /**
     * @brief Collect a simplemc::measurement_set from different MPI processes.
     *
     * @param comm simplemc::mpi::communicator object.
     * @param ms Measurement set to reduce in place.
     */
    friend void simplemc_mpi_collect(const mpi::communicator& comm, measurement_set& ms) {
        ms.mpi_collect_entries(comm);
    }

private:
    /// Cached indices of currently-active measurements.
    std::vector<std::size_t> active_;
};

/// Deduction guide: deduce the measurement types from the constructor arguments.
template <mc_measurement... Ms>
measurement_set(measurement<Ms>...) -> measurement_set<Ms...>;

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_MEASUREMENT_SET_HPP
