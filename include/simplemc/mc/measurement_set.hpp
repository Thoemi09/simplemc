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
 * @details The set inherits from simplemc::tuple_set to store simplemc::measurement entries in a
 * `std::tuple`, so the measurement *types* are fixed at construction. It also holds a cache of
 * indices of the measurements currently flagged active, which is rebuilt from the
 * measurement::is_active() flags by refresh_active().
 *
 * Simulation drivers call measure_all(), which invokes `measure()` on every active entry to perform
 * measurements.
 *
 * Measurements are registered at construction-time, e.g.
 *
 * @code{.cpp}
 * simplemc::measurement_set {
 *     simplemc::measurement { foo { }, "foo" },
 *     simplemc::measurement { bar { }, "bar", false }
 * };
 * @endcode
 *
 * The measurement::is_active() flags stay mutable at runtime via set_active(). Names are fixed at
 * construction. Lookup and typed access are provided by the base class simplemc::tuple_set.
 *
 * @tparam Ms User measurement types (each satisfying simplemc::mc_measurement).
 */
template <mc_measurement... Ms>
class measurement_set : public tuple_set<measurement<Ms>...> {
public:
    /**
     * @brief Construct the set from the given measurements.
     *
     * @param ms Measurements to store, in order.
     */
    explicit measurement_set(measurement<Ms>... ms) : tuple_set<measurement<Ms>...> { std::move(ms)... } {}

    /**
     * @brief Set the active state of a registered measurement.
     *
     * @details It throws a simplemc::simplemc_exception if the name is not registered.
     *
     * @param name Name of the measurement whose active state to set.
     * @param is_active New active state.
     */
    void set_active(std::string_view name, bool is_active) {
        this->visit_at(name, [&](auto& m) { m.set_active(is_active); });
    }

    /**
     * @brief Cached indices of currently-active measurements.
     *
     * @return Span over the indices of currently-active measurements.
     */
    [[nodiscard]] std::span<const std::size_t> active_indices() const noexcept { return active_; }

    /**
     * @brief Rebuild the active set cache from the measurement::is_active() flags.
     */
    void refresh_active() {
        active_.clear();
        active_.reserve(this->size());
        std::size_t i = 0;
        this->for_each([&](const auto& m) {
            if (m.is_active()) {
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
     * @brief Serialize a simplemc::measurement_set.
     *
     * @details It dispatches to tuple_set::save_entries, which serializes all measurements in the
     * set.
     *
     * @tparam S Serializer type.
     * @param s Serializer handle.
     * @param ms Measurement set to serialize.
     */
    template <serializer S>
    friend void simplemc_save(S s, const measurement_set& ms) {
        ms.save_entries(s);
    }

    /**
     * @brief Deserialize a simplemc::measurement_set.
     *
     * @details It dispatches to tuple_set::load_entries, which deserializes all measurements in the
     * set.
     *
     * @tparam S Serializer type.
     * @param s Serializer handle.
     * @param ms Measurement set to deserialize.
     */
    template <serializer S>
    friend void simplemc_load(const S& s, measurement_set& ms) {
        ms.load_entries(s, "measurement");
    }

    /**
     * @brief Serialize the user-input config of a simplemc::measurement_set.
     *
     * @details It dispatches to tuple_set::save_input_config_entries, which serializes the
     * input-config of all measurements in the set.
     *
     * @tparam S Serializer type.
     * @param s Serializer handle.
     * @param ms Measurement set to serialize.
     */
    template <serializer S>
    friend void simplemc_save_input_config(S s, const measurement_set& ms) {
        ms.save_input_config_entries(s);
    }

    /**
     * @brief Deserialize the user-input config of a simplemc::measurement_set.
     *
     * @details It dispatches to tuple_set::load_input_config_entries, which deserializes the
     * input-config of all measurements in the set.
     *
     * @tparam S Serializer type.
     * @param s Serializer handle.
     * @param ms Measurement set to deserialize.
     */
    template <serializer S>
    friend void simplemc_load_input_config(const S& s, measurement_set& ms) {
        ms.load_input_config_entries(s, "measurement");
    }

    /**
     * @brief Collect a simplemc::measurement_set from different MPI processes.
     *
     * @details It dispatches to tuple_set::mpi_collect_entries.
     *
     * @param comm simplemc::mpi::communicator object.
     * @param ms Measurement set to reduce in place.
     */
    friend void simplemc_mpi_collect(const mpi::communicator& comm, measurement_set& ms) {
        ms.mpi_collect_entries(comm);
    }

private:
    std::vector<std::size_t> active_;
};

/// Deduction guide: deduce the measurement types from the constructor arguments.
template <mc_measurement... Ms>
measurement_set(measurement<Ms>...) -> measurement_set<Ms...>;

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_MEASUREMENT_SET_HPP
