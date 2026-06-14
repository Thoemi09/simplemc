/**
 * @file
 * @brief Set of MC measurements.
 */

#ifndef SIMPLEMC_MC_MEASUREMENT_SET_HPP
#define SIMPLEMC_MC_MEASUREMENT_SET_HPP

#include <simplemc/mc/measurement.hpp>
#include <simplemc/mc/named_set.hpp>
#include <simplemc/mc/serializer.hpp>
#include <simplemc/mpi/communicator.hpp>
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
 * @details The set holds registered simplemc::measurement objects and a cache of indices for those
 * currently flagged active. The cache is updated by calling refresh_active(), which iterates the
 * stored measurements and rebuilds the cache from their measurement::is_active flags.
 *
 * Simulation drivers call measure_all() to invoke `measure()` on every measurement in the active
 * cache.
 *
 * Registration, lookup and typed access are provided by the base class simplemc::named_set. The same
 * API is shared with simplemc::update_set.
 */
class measurement_set : public named_set<measurement> {
public:
    /**
     * @brief Serializer type used for both checkpoint and input-config serialization.
     */
    using serializer_type = mc_serializer;

    /**
     * @brief Register a simplemc::measurement in the set.
     *
     * @details It calls named_set::add_checked(). A simplemc::simplemc_exception is thrown if the
     * name is already registered.
     *
     * @param m Measurement to register.
     */
    void add(measurement m) { this->add_checked(std::move(m), "measurement"); }

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
        this->entries_[*idx].is_active = is_active;
    }

    /**
     * @brief Cached indices of currently-active measurements.
     *
     * @return Span over the indices of currently-active measurements.
     */
    [[nodiscard]] std::span<const std::size_t> active_indices() const noexcept { return active_; }

    /**
     * @brief Rebuild the active set cache.
     *
     * @details It iterates over all registered measurements and updates the cache according to their
     * measurement::is_active flags.
     */
    void refresh_active() {
        active_.clear();
        active_.reserve(this->entries_.size());
        for (std::size_t i = 0; i < this->entries_.size(); ++i) {
            if (this->entries_[i].is_active) {
                active_.push_back(i);
            }
        }
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
            this->entries_[i].measure();
        }
    }

    /**
     * @brief Serialize a simplemc::measurement_set.
     *
     * @details It serializes all registered measurements by calling named_set::save_entries().
     *
     * @param s Serializer handle.
     * @param ms Measurement set to serialize.
     */
    friend void simplemc_save(serializer_type& s, const measurement_set& ms) { ms.save_entries(s); }

    /**
     * @brief Deserialize a simplemc::measurement_set.
     *
     * @details It deserializes all registered measurements by calling named_set::load_entries().
     *
     * @param s Serializer handle.
     * @param ms Measurement set to deserialize into.
     */
    friend void simplemc_load(const serializer_type& s, measurement_set& ms) { ms.load_entries(s, "measurement"); }

    /**
     * @brief Serialize the user-input config of a simplemc::measurement_set.
     *
     * @details It serializes all registered measurements by calling
     * named_set::save_input_config_entries().
     *
     * @param s Serializer handle.
     * @param ms Measurement set to serialize.
     */
    friend void simplemc_save_input_config(serializer_type& s, const measurement_set& ms) {
        ms.save_input_config_entries(s);
    }

    /**
     * @brief Deserialize the user-input config of a simplemc::measurement_set.
     *
     * @details It deserializes all registered measurements by calling
     * named_set::load_input_config_entries().
     *
     * @param s Serializer handle.
     * @param ms Measurement set to deserialize into.
     */
    friend void simplemc_load_input_config(const serializer_type& s, measurement_set& ms) {
        ms.load_input_config_entries(s, "measurement");
    }

    /**
     * @brief Collect simplemc::measurement_set objects from different MPI processes.
     *
     * @details It simply calls named_set::mpi_collect_entries().
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

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_MEASUREMENT_SET_HPP
