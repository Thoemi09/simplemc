/**
 * @file
 * @brief Set of MC updates.
 */

#ifndef SIMPLEMC_MC_UPDATE_SET_HPP
#define SIMPLEMC_MC_UPDATE_SET_HPP

#include <simplemc/mc/tuple_set.hpp>
#include <simplemc/mc/update.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-entities-updates
 * @{
 */

/**
 * @brief Set of MC updates.
 *
 * @details The set stores its simplemc::update entries in a `std::tuple` (via simplemc::tuple_set), so
 * the update *types* are fixed at construction. It holds a `std::discrete_distribution` built from the
 * update::weight values; the distribution is (re)built by rebuild_distribution(), which validates the
 * weights and recomputes the detailed-balance update::ratio for every update.
 *
 * Monte Carlo kernels usually:
 *
 * - call select() to sample an update index according to the cached distribution,
 * - reach the corresponding update through tuple_set::visit_at(), and
 * - call `attempt()` and then `accept()`/`reject()` on it.
 *
 * Registration is construction-time (build the whole set in one initializer, deducing the types via
 * CTAD); names/weights and inverse-pairing stay mutable at runtime via set_weight() / link_pair().
 * Lookup and typed access are provided by the base simplemc::tuple_set.
 *
 * @tparam Us User update types (each satisfying simplemc::mc_update).
 */
template <mc_update... Us>
class update_set : public tuple_set<update<Us>...> {
public:
    /**
     * @brief Construct the set from its updates.
     *
     * @param us Updates to store, in order.
     */
    explicit update_set(update<Us>... us) : tuple_set<update<Us>...> { std::move(us)... } {}

    /**
     * @brief Link two registered updates as a detailed-balance inverse pair.
     *
     * @details It cross-links update::inv_name of the two named updates. The detailed-balance
     * update::ratio is not computed here — rebuild_distribution() derives it from the current weights
     * before every run.
     *
     * It throws a simplemc::simplemc_exception if the two names are equal or if either is not
     * registered.
     *
     * @param name_a First update name.
     * @param name_b Second update name.
     */
    void link_pair(std::string_view name_a, std::string_view name_b) {
        if (name_a == name_b) {
            throw simplemc_exception(fmt::format("paired updates must have distinct names (both are '{}')", name_a));
        }
        const auto ia = this->find(name_a);
        const auto ib = this->find(name_b);
        if (!ia) {
            throw simplemc_exception(fmt::format("update '{}' is not registered", name_a));
        }
        if (!ib) {
            throw simplemc_exception(fmt::format("update '{}' is not registered", name_b));
        }
        this->visit_at(*ia, [&](auto& u) { u.inv_name = std::string { name_b }; });
        this->visit_at(*ib, [&](auto& u) { u.inv_name = std::string { name_a }; });
    }

    /**
     * @brief Change the selection weight of a registered update.
     *
     * @details It validates \f$ w \geq 0 \f$, looks up the update by name, and writes the new weight.
     * A simplemc::simplemc_exception is thrown if the name is not registered or the weight is negative.
     *
     * @param name Name of the update whose weight to change.
     * @param w New selection weight (must be \f$ \geq 0 \f$).
     */
    void set_weight(std::string_view name, double w) {
        if (w < 0.0) {
            throw simplemc_exception(fmt::format("update weight must be >= 0 (got {} on '{}')", w, name));
        }
        const auto idx = this->find(name);
        if (!idx) {
            throw simplemc_exception(fmt::format("update '{}' is not registered", name));
        }
        this->visit_at(*idx, [&](auto& u) { u.weight = w; });
    }

    /**
     * @brief Build the selection distribution from the update::weight values and recompute the
     * detailed-balance update::ratio for every update.
     *
     * @details It first recomputes every update::ratio (\f$ 1 \f$ for self-inverse updates, else
     * \f$ w_{\text{inv}} / w \f$) and then initializes the discrete distribution.
     *
     * Because update::weight and update::inv_name are public fields, the invariants enforced by the
     * constructor, set_weight() and link_pair() can be bypassed by direct writes; they are re-validated
     * here (the choke point called before every run, e.g. by simplemc::metropolis_kernel::prepare()).
     *
     * It throws a simplemc::simplemc_exception if a weight is negative, if an update::inv_name is not
     * registered, if the inverse pairing is asymmetric, if exactly one weight of an inverse pair is
     * zero, or if all weights are zero.
     */
    void rebuild_distribution() {
        // recompute detailed-balance ratios and validate weights and inverse pairs
        this->for_each([&](auto& u) {
            if (u.weight < 0.0) {
                throw simplemc_exception(fmt::format("update weight must be >= 0 (got {} on '{}')", u.weight, u.name));
            }
            if (u.inv_name == u.name) {
                u.ratio = 1.0;
                return;
            }
            const auto idx = this->find(u.inv_name);
            if (!idx) {
                throw simplemc_exception(
                    fmt::format("inverse update '{}' of '{}' is not registered", u.inv_name, u.name));
            }
            double w_inv = 0.0;
            std::string_view inv_inv;
            this->visit_at(*idx, [&](const auto& inv) {
                w_inv = inv.weight;
                inv_inv = inv.inv_name;
            });
            if (inv_inv != u.name) {
                throw simplemc_exception(fmt::format("inverse pairing is asymmetric: '{}' names '{}' as its "
                                                     "inverse, but '{}' names '{}'",
                    u.name, u.inv_name, u.inv_name, inv_inv));
            }
            if ((u.weight == 0.0) != (w_inv == 0.0)) {
                throw simplemc_exception(fmt::format("inverse pair must have both weights zero or both non-zero "
                                                     "(got {} on '{}' and {} on '{}')",
                    u.weight, u.name, w_inv, u.inv_name));
            }
            u.ratio = u.weight != 0.0 ? w_inv / u.weight : 1.0;
        });

        // gather weights (at least one must be > 0) and rebuild the distribution
        std::array<double, sizeof...(Us)> weights {};
        std::size_t k = 0;
        this->for_each([&](const auto& u) { weights[k++] = u.weight; });
        if (!std::ranges::any_of(weights, [](double w) { return w > 0.0; })) {
            throw simplemc_exception("at least one update weight must be > 0");
        }
        dist_ = std::discrete_distribution<std::size_t> { weights.begin(), weights.end() };
    }

    /**
     * @brief Sample one update according to the cached selection distribution.
     *
     * @tparam RNG Random number generator type.
     * @param rng Random number generator.
     * @return Index of the selected update.
     */
    template <typename RNG>
    [[nodiscard]] std::size_t select(RNG& rng) {
        return dist_(rng);
    }

    /**
     * @brief Zero the current-run counters of every update via update::reset_run_counters().
     */
    void reset_run_counters() noexcept {
        this->for_each([](auto& u) { u.reset_run_counters(); });
    }

    /**
     * @brief Accumulate the current-run counters of every update via update::accumulate_counters().
     */
    void accumulate_counters() noexcept {
        this->for_each([](auto& u) { u.accumulate_counters(); });
    }

    /**
     * @brief Serialize a simplemc::update_set: every update, keyed by name.
     */
    template <serializer S>
    friend void simplemc_save(S& s, const update_set& us) {
        us.save_entries(s);
    }

    /**
     * @brief Deserialize a simplemc::update_set: every update, keyed by name.
     */
    template <serializer S>
    friend void simplemc_load(const S& s, update_set& us) {
        us.load_entries(s, "update");
    }

    /**
     * @brief Serialize the user-input config of a simplemc::update_set.
     */
    template <serializer S>
    friend void simplemc_save_input_config(S& s, const update_set& us) {
        us.save_input_config_entries(s);
    }

    /**
     * @brief Deserialize the user-input config of a simplemc::update_set.
     */
    template <serializer S>
    friend void simplemc_load_input_config(const S& s, update_set& us) {
        us.load_input_config_entries(s, "update");
    }

    /**
     * @brief Collect a simplemc::update_set from different MPI processes.
     *
     * @details It reduces every update's counters (and value, if supported). Weights, names, the
     * discrete selection distribution, and detailed-balance ratios are local data assumed identical
     * across ranks and are not touched.
     *
     * @param comm simplemc::mpi::communicator object.
     * @param us Update set to reduce in place.
     */
    friend void simplemc_mpi_collect(const mpi::communicator& comm, update_set& us) { us.mpi_collect_entries(comm); }

private:
    std::discrete_distribution<std::size_t> dist_;
};

/// Deduction guide: deduce the update types from the constructor arguments.
template <mc_update... Us>
update_set(update<Us>...) -> update_set<Us...>;

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_UPDATE_SET_HPP
