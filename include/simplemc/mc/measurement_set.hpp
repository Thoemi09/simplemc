/**
 * @file
 * @brief Owns a collection of simplemc::measurement entries plus a cached active set.
 */

#ifndef SIMPLEMC_MC_MEASUREMENT_SET_HPP
#define SIMPLEMC_MC_MEASUREMENT_SET_HPP

#include <simplemc/mc/measurement.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace simplemc {

/**
 * @addtogroup simplemc-mc
 * @{
 */

/**
 * @brief Owns a collection of simplemc::measurement entries plus a cached active set.
 *
 * @details The set holds registered measurements (each carrying its own wrapper and metadata) and
 * a cache of indices for those currently flagged active. refresh_active() rebuilds the cache from
 * the per-entry `is_active` flags; measure_all() iterates the cache and calls `measure()` on each
 * active entry.
 *
 * Toggling `is_active` on an entry directly does **not** update the cache; call refresh_active()
 * to apply changes between runs.
 *
 * @tparam S1 Serializer type used for checkpoint serialization.
 * @tparam S2 Serializer type used for input-config serialization.
 */
template <serializer S1 = json_serializer, serializer S2 = json_serializer>
class measurement_set {
public:
    using cp_serializer_type = S1;
    using ic_serializer_type = S2;
    using measurement_type = measurement<S1, S2>;

    /**
     * @brief Register a measurement.
     *
     * @details The simplemc::measurement constructor validated `name` non-empty. This method
     * additionally validates that the name is unique within the set.
     */
    void add(measurement_type m) {
        if (find(m.name).has_value()) {
            throw simplemc_exception(fmt::format("measurement '{}' is already registered", m.name));
        }
        measurements_.push_back(std::move(m));
    }

    /**
     * @brief Toggle the active state of a registered measurement.
     *
     * @details Takes effect on the next refresh_active() (or the next `run()` that calls it).
     */
    void set_active(std::string_view name, bool b) {
        const auto idx = find(name);
        if (!idx) {
            throw simplemc_exception(fmt::format("measurement '{}' is not registered", name));
        }
        measurements_[*idx].is_active = b;
    }

    /**
     * @brief Look up a measurement by name.
     */
    [[nodiscard]] std::optional<std::size_t> find(std::string_view name) const noexcept {
        const auto it = std::ranges::find_if(
            measurements_, [name](const measurement_type& m) { return m.name == name; });
        if (it == measurements_.end()) {
            return std::nullopt;
        }
        return static_cast<std::size_t>(std::distance(measurements_.begin(), it));
    }

    [[nodiscard]] std::size_t size() const noexcept { return measurements_.size(); }
    [[nodiscard]] bool empty() const noexcept { return measurements_.empty(); }

    [[nodiscard]] measurement_type& at(std::size_t i) noexcept { return measurements_[i]; }
    [[nodiscard]] const measurement_type& at(std::size_t i) const noexcept { return measurements_[i]; }
    [[nodiscard]] std::span<const measurement_type> data() const noexcept { return measurements_; }

    /**
     * @brief Cached indices of currently-active measurements.
     */
    [[nodiscard]] const std::vector<std::size_t>& active_indices() const noexcept { return active_; }

    /**
     * @brief Recover a typed pointer to a registered measurement's wrapped user value.
     */
    template <class T>
    [[nodiscard]] T* get(std::string_view name) noexcept {
        const auto idx = find(name);
        return idx ? measurements_[*idx].template get<T>() : nullptr;
    }

    template <class T>
    [[nodiscard]] const T* get(std::string_view name) const noexcept {
        const auto idx = find(name);
        return idx ? measurements_[*idx].template get<T>() : nullptr;
    }

    /**
     * @brief Rebuild the active-set cache from each entry's `is_active` flag.
     */
    void refresh_active() {
        active_.clear();
        active_.reserve(measurements_.size());
        for (std::size_t i = 0; i < measurements_.size(); ++i) {
            if (measurements_[i].is_active) {
                active_.push_back(i);
            }
        }
    }

    /**
     * @brief Clear the active-set cache (useful for `skip_measurements`).
     */
    void clear_active() noexcept { active_.clear(); }

    /**
     * @brief Invoke `measure()` on every entry in the cached active set.
     */
    void measure_all() {
        for (std::size_t i : active_) {
            measurements_[i].measure();
        }
    }

    /**
     * @brief Invoke `measure()` on a single entry by index.
     */
    void measure(std::size_t i) { measurements_[i].measure(); }

    friend void simplemc_save(cp_serializer_type& s, const measurement_set& ms) {
        for (const auto& m : ms.measurements_) {
            auto entry = s[m.name];
            simplemc_save(entry, m);
        }
    }

    friend void simplemc_load(const cp_serializer_type& s, measurement_set& ms) {
        for (auto& m : ms.measurements_) {
            if (!s.has(m.name)) {
                throw simplemc_exception(
                    fmt::format("measurement '{}' not found in serialized data", m.name));
            }
            const auto entry = s[m.name];
            simplemc_load(entry, m);
        }
    }

    friend void simplemc_save_input_config(ic_serializer_type& s, const measurement_set& ms) {
        for (const auto& m : ms.measurements_) {
            auto entry = s[m.name];
            simplemc_save_input_config(entry, m);
        }
    }

    friend void simplemc_load_input_config(const ic_serializer_type& s, measurement_set& ms) {
        for (auto& m : ms.measurements_) {
            if (!s.has(m.name)) {
                throw simplemc_exception(
                    fmt::format("measurement '{}' not found in input config", m.name));
            }
            const auto entry = s[m.name];
            simplemc_load_input_config(entry, m);
        }
    }

    /**
     * @brief All-reduce every registered measurement across MPI ranks.
     *
     * @details Iterates the registered measurements and forwards to the per-`measurement<>`
     * `simplemc_mpi_collect`, which in turn delegates to the wrapper's `mpi_collect()`. Names,
     * ordering, and the active cache are local and not touched; all are assumed identical across
     * ranks by construction.
     *
     * @param comm MPI communicator over which to reduce.
     * @param ms Measurement set to reduce in place.
     */
    friend void simplemc_mpi_collect(const mpi::communicator& comm, measurement_set& ms) {
        for (auto& m : ms.measurements_) {
            simplemc_mpi_collect(comm, m);
        }
    }

private:
    std::vector<measurement_type> measurements_;
    std::vector<std::size_t> active_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_MEASUREMENT_SET_HPP
