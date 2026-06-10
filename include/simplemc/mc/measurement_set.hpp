/**
 * @file
 * @brief Owns a collection of simplemc::measurement entries plus a cached active set.
 */

#ifndef SIMPLEMC_MC_MEASUREMENT_SET_HPP
#define SIMPLEMC_MC_MEASUREMENT_SET_HPP

#include <simplemc/mc/measurement.hpp>
#include <simplemc/mc/named_set.hpp>
#include <simplemc/mc/traits.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>

#include <cstddef>
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
 * active entry. The registration / lookup APIs are shared with simplemc::update_set through
 * simplemc::named_set.
 *
 * Toggling `is_active` on an entry directly does **not** update the cache; call refresh_active()
 * to apply changes between runs.
 *
 * @tparam Traits Traits bundle satisfying simplemc::mc_traits_like (default: simplemc::mc_traits<>).
 */
template <mc_traits_like Traits = mc_traits<>>
class measurement_set : public named_set<measurement<Traits>> {
    using base_type = named_set<measurement<Traits>>;

public:
    using traits_type = Traits;
    using checkpoint_serializer_type = typename Traits::checkpoint_serializer_type;
    using input_config_serializer_type = typename Traits::input_config_serializer_type;
    using measurement_type = measurement<Traits>;

    /**
     * @brief Register a measurement.
     *
     * @details The simplemc::measurement constructor validated `name` non-empty. This method
     * additionally validates that the name is unique within the set.
     */
    void add(measurement_type m) { this->add_checked(std::move(m), "measurement"); }

    /**
     * @brief Toggle the active state of a registered measurement.
     *
     * @details Takes effect on the next refresh_active() (or the next `run()` that calls it).
     */
    void set_active(std::string_view name, bool b) {
        const auto idx = this->find(name);
        if (!idx) {
            throw simplemc_exception(fmt::format("measurement '{}' is not registered", name));
        }
        this->entries_[*idx].is_active = b;
    }

    /**
     * @brief Cached indices of currently-active measurements.
     */
    [[nodiscard]] const std::vector<std::size_t>& active_indices() const noexcept { return active_; }

    /**
     * @brief Rebuild the active-set cache from each entry's `is_active` flag.
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
     * @brief Clear the active-set cache (useful for `skip_measurements`).
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

    friend void simplemc_save(checkpoint_serializer_type& s, const measurement_set& ms) { ms.save_entries(s); }

    friend void simplemc_load(const checkpoint_serializer_type& s, measurement_set& ms) {
        ms.load_entries(s, "measurement");
    }

    friend void simplemc_save_input_config(input_config_serializer_type& s, const measurement_set& ms) {
        ms.save_input_config_entries(s);
    }

    friend void simplemc_load_input_config(const input_config_serializer_type& s, measurement_set& ms) {
        ms.load_input_config_entries(s, "measurement");
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
        ms.mpi_collect_entries(comm);
    }

private:
    std::vector<std::size_t> active_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_MEASUREMENT_SET_HPP
