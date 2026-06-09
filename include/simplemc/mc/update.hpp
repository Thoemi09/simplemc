/**
 * @file
 * @brief Per-update aggregate: wrapped user value plus registration data and counters.
 */

#ifndef SIMPLEMC_MC_UPDATE_HPP
#define SIMPLEMC_MC_UPDATE_HPP

#include <simplemc/mc/basic_update.hpp>
#include <simplemc/mc/concepts.hpp>
#include <simplemc/mpi/all_reduce.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>

#include <concepts>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-mc
 * @{
 */

/**
 * @brief One Monte Carlo update entry: the wrapped user value plus registration data and counters.
 *
 * @details simplemc::update owns a simplemc::basic_update together with the metadata that identifies
 * and configures it (`name`, `inv_name`, `weight`, detailed-balance `ratio`) and the counters that
 * track its behavior over a run (`nprops`, `naccs`, `nimps`) and over the full lifetime
 * (`cumulative_*`). All fields are public so kernels and reporting code can read and write them
 * directly.
 *
 * Construction validates `name` non-empty and `weight >= 0`. `inv_name` defaults to `name` (i.e.
 * self-inverse) and `ratio` defaults to `1.0`; cross-linked pairs are produced by
 * simplemc::update_set::add_pair.
 *
 * @tparam S1 Serializer type used for checkpoint serialization.
 * @tparam S2 Serializer type used for input-config serialization.
 */
template <serializer S1 = json_serializer, serializer S2 = json_serializer>
struct update {
    /**
     * @brief Type used for checkpoint serialization.
     */
    using cp_serializer_type = S1;

    /**
     * @brief Type used for input-config serialization.
     */
    using ic_serializer_type = S2;

    /**
     * @brief The wrapped user update.
     */
    basic_update<S1, S2> wrapped;

    /**
     * @brief Identifier used in lookups and printed reports.
     */
    std::string name;

    /**
     * @brief Identifier of the inverse update. Equal to `name` for self-inverse updates.
     */
    std::string inv_name;

    /**
     * @brief Unnormalized selection weight (\f$ w \geq 0 \f$).
     */
    double weight;

    /**
     * @brief Detailed-balance correction multiplier applied by the kernel to `attempt()`.
     *
     * @details \f$ 1 \f$ for a self-inverse update; \f$ w_{\text{inv}} / w \f$ for a paired update.
     */
    double ratio = 1.0;

    /**
     * @brief Number of times this update has been proposed in the current run.
     */
    std::uint64_t nprops = 0;

    /**
     * @brief Number of times this update has been accepted in the current run.
     */
    std::uint64_t naccs = 0;

    /**
     * @brief Number of times this update has been signaled as impossible in the current run.
     */
    std::uint64_t nimps = 0;

    /**
     * @brief Cumulative number of proposals across runs.
     */
    std::uint64_t cumulative_nprops = 0;

    /**
     * @brief Cumulative number of acceptances across runs.
     */
    std::uint64_t cumulative_naccs = 0;

    /**
     * @brief Cumulative number of impossible signals across runs.
     */
    std::uint64_t cumulative_nimps = 0;

    /**
     * @brief Construct by wrapping a user-defined update value.
     *
     * @details Forwards `u` into an internally-constructed simplemc::basic_update. Validates
     * `name` non-empty and `weight >= 0`. The forwarded type must satisfy simplemc::mc_update and
     * `std::copy_constructible`.
     *
     * @tparam U User update type.
     * @param u User value to wrap.
     * @param name Identifier.
     * @param weight Unnormalized selection weight.
     */
    template <class U>
        requires(!std::same_as<std::remove_cvref_t<U>, update>) &&
                (!std::same_as<std::remove_cvref_t<U>, basic_update<S1, S2>>) &&
                mc_update<std::remove_cvref_t<U>> &&
                std::copy_constructible<std::remove_cvref_t<U>>
    update(U&& u, std::string name, double weight) : update {
        basic_update<S1, S2> { std::forward<U>(u) }, std::move(name), weight
    } {}

    /**
     * @brief Construct from a pre-built simplemc::basic_update wrapper.
     *
     * @details Validates `name` non-empty and `weight >= 0`. Sets `inv_name = name`,
     * `ratio = 1.0`. Counters default to zero.
     *
     * @param w Pre-built wrapper.
     * @param name Identifier.
     * @param weight Unnormalized selection weight.
     */
    update(basic_update<S1, S2> w, std::string name, double weight) :
        wrapped { std::move(w) }, name { std::move(name) }, inv_name { this->name }, weight { weight } {
        if (this->name.empty()) {
            throw simplemc_exception("update name must be non-empty");
        }
        if (this->weight < 0.0) {
            throw simplemc_exception(fmt::format("update weight must be >= 0 (got {} on '{}')", this->weight, this->name));
        }
    }

    /**
     * @brief Zero the current-run counters; leaves the cumulative counters untouched.
     */
    void reset_run_counters() noexcept {
        nprops = 0;
        naccs = 0;
        nimps = 0;
    }

    /**
     * @brief Fold the current-run counters into the cumulative counters and reset the current ones.
     */
    void accumulate_counters() noexcept {
        cumulative_nprops += nprops;
        cumulative_naccs += naccs;
        cumulative_nimps += nimps;
        reset_run_counters();
    }

    /**
     * @brief Recover a pointer to the wrapped user value, checked by type.
     *
     * @tparam T Expected concrete type of the wrapped user value.
     * @return Pointer to the wrapped value, or `nullptr` on type mismatch.
     */
    template <class T>
    [[nodiscard]] T* get() noexcept {
        return wrapped.template get<T>();
    }

    /**
     * @brief Const overload of get().
     */
    template <class T>
    [[nodiscard]] const T* get() const noexcept {
        return wrapped.template get<T>();
    }

    /**
     * @brief Serialize persistent fields of this update.
     *
     * @details Writes `inv_name`, `weight`, `ratio`, the three `cumulative_*` counters, and
     * delegates to the wrapper for the `"user"` sub-key. Schema matches the previous
     * simplemc::update_stats layout exactly (plus the wrapper's `"user"` sub-tree), so existing
     * checkpoints round-trip.
     *
     * @param s Serializer handle.
     * @param u Update to serialize.
     */
    friend void simplemc_save(cp_serializer_type& s, const update& u) {
        s.save_at("inv_name", u.inv_name);
        s.save_at("weight", u.weight);
        s.save_at("ratio", u.ratio);
        s.save_at("cumulative_nprops", u.cumulative_nprops);
        s.save_at("cumulative_naccs", u.cumulative_naccs);
        s.save_at("cumulative_nimps", u.cumulative_nimps);
        u.wrapped.save_at(s, "user");
    }

    /**
     * @brief Deserialize the persistent fields of this update.
     */
    friend void simplemc_load(const cp_serializer_type& s, update& u) {
        s.load_at("inv_name", u.inv_name);
        s.load_at("weight", u.weight);
        s.load_at("ratio", u.ratio);
        s.load_at("cumulative_nprops", u.cumulative_nprops);
        s.load_at("cumulative_naccs", u.cumulative_naccs);
        s.load_at("cumulative_nimps", u.cumulative_nimps);
        u.wrapped.load_at(s, "user");
    }

    /**
     * @brief Serialize the user-input config of this update.
     *
     * @details Writes only `weight` and delegates to the wrapper for the `"user"` sub-key.
     *
     * @param s Serializer handle.
     * @param u Update to serialize.
     */
    friend void simplemc_save_input_config(ic_serializer_type& s, const update& u) {
        s.save_at("weight", u.weight);
        u.wrapped.save_input_config_at(s, "user");
    }

    /**
     * @brief Deserialize the user-input config of this update.
     *
     * @details Each field is optional in the input config (uses `try_load_at`) so callers may omit
     * any fields they do not wish to override.
     */
    friend void simplemc_load_input_config(const ic_serializer_type& s, update& u) {
        s.try_load_at("weight", u.weight);
        u.wrapped.load_input_config_at(s, "user");
    }

    /**
     * @brief All-reduce this update's counters and wrapped payload across MPI ranks.
     *
     * @details Allreduces the six counter fields (`nprops`, `naccs`, `nimps`, plus the three
     * `cumulative_*`) with `MPI_SUM`, then forwards to the wrapper's `mpi_collect()` so the wrapped
     * user value is reduced via its own ADL hook. Identification fields (`name`, `inv_name`,
     * `weight`, `ratio`) are intentionally not touched — they are local registration data assumed
     * identical across ranks.
     *
     * @param comm MPI communicator over which to reduce.
     * @param u Update to reduce in place.
     */
    friend void simplemc_mpi_collect(const mpi::communicator& comm, update& u) {
        mpi::all_reduce_in_place(u.nprops, MPI_SUM, comm);
        mpi::all_reduce_in_place(u.naccs, MPI_SUM, comm);
        mpi::all_reduce_in_place(u.nimps, MPI_SUM, comm);
        mpi::all_reduce_in_place(u.cumulative_nprops, MPI_SUM, comm);
        mpi::all_reduce_in_place(u.cumulative_naccs, MPI_SUM, comm);
        mpi::all_reduce_in_place(u.cumulative_nimps, MPI_SUM, comm);
        u.wrapped.mpi_collect(comm);
    }
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_UPDATE_HPP
