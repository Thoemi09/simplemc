/**
 * @file
 * @brief Type-erased MC update together with some metadata.
 */

#ifndef SIMPLEMC_MC_UPDATE_HPP
#define SIMPLEMC_MC_UPDATE_HPP

#include <simplemc/mc/basic_update.hpp>
#include <simplemc/mc/concepts.hpp>
#include <simplemc/mc/serializer.hpp>
#include <simplemc/mpi/all_reduce.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <fmt/format.h>

#include <concepts>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-mc-entities-updates
 * @{
 */

/**
 * @brief Type-erased MC update together with some metadata.
 *
 * @details simplemc::update owns a simplemc::basic_update together with its metadata:
 *
 * - a unique name that identifies the update,
 * - the name of the inverse update,
 * - an unnormalized selection weight,
 * - a detailed-balance correction ratio, and
 * - counters tracking the number of proposals, acceptances, and impossible signals over the current
 * run and across multiple runs.
 *
 * All fields are public so kernels and reporting code can read and write them directly.
 */
struct update {
    /**
     * @brief Serializer type used for both checkpoint and input-config serialization.
     */
    using serializer_type = mc_serializer;

    /**
     * @brief The wrapped user update.
     */
    basic_update wrapped;

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
     * @brief Constructor wraps a user-defined update value.
     *
     * @details It forwards the given update into an internally-constructed simplemc::basic_update
     * and validates that the name is not empty and that the weight is non-negative. Otherwise, it
     * throws a simplemc::simplemc_exception.
     *
     * The forwarded type must satisfy simplemc::mc_update and `std::copy_constructible`.
     *
     * @tparam U User update type.
     * @param u User update to wrap.
     * @param name Identifier.
     * @param weight Unnormalized selection weight.
     */
    template <typename U>
        requires(!std::same_as<std::remove_cvref_t<U>, update>) &&
        (!std::same_as<std::remove_cvref_t<U>, basic_update>) && mc_update<std::remove_cvref_t<U>> &&
        std::copy_constructible<std::remove_cvref_t<U>>
    update(U&& u, std::string name, double weight) :
        update { basic_update { std::forward<U>(u) }, std::move(name), weight } {}

    /**
     * @brief Construct an update from a pre-built simplemc::basic_update wrapper.
     *
     * @details It validates that the name is not empty and that the weight is non-negative.
     * Otherwise, it throws a simplemc::simplemc_exception.
     *
     * It sets `inv_name = name` and `ratio = 1.0`. The counters default to zero.
     *
     * @param w Pre-built simplemc::basic_update wrapper.
     * @param name Identifier.
     * @param weight Unnormalized selection weight.
     */
    update(basic_update w, std::string name, double weight) :
        wrapped { std::move(w) },
        name { std::move(name) },
        inv_name { this->name },
        weight { weight } {
        if (this->name.empty()) {
            throw simplemc_exception("update name must be non-empty");
        }
        if (this->weight < 0.0) {
            throw simplemc_exception(
                fmt::format("update weight must be >= 0 (got {} on '{}')", this->weight, this->name));
        }
    }

    /**
     * @brief Zero the current-run counters but leave the cumulative counters untouched.
     */
    void reset_run_counters() noexcept {
        nprops = 0;
        naccs = 0;
        nimps = 0;
    }

    /**
     * @brief Add the current-run counters to the cumulative counters and call reset_run_counters().
     */
    void accumulate_counters() noexcept {
        cumulative_nprops += nprops;
        cumulative_naccs += naccs;
        cumulative_nimps += nimps;
        reset_run_counters();
    }

    /**
     * @brief Recover a pointer to the wrapped user update.
     *
     * @details It simply calls basic_update::get<T>().
     *
     * @tparam T Expected type of the wrapped user update.
     * @return Pointer to the wrapped update, or `nullptr` on type mismatch.
     */
    template <typename T>
    [[nodiscard]] T* get() noexcept {
        return wrapped.template get<T>();
    }

    /**
     * @brief Const overload of get().
     */
    template <typename T>
    [[nodiscard]] const T* get() const noexcept {
        return wrapped.template get<T>();
    }

    /**
     * @brief Serialize a simplemc::update.
     *
     * @details It serializes all metadata except `name` and calls basic_update::save_at().
     *
     * @param s Serializer handle.
     * @param u Update to serialize.
     */
    friend void simplemc_save(serializer_type& s, const update& u) {
        s.save_at("inv_name", u.inv_name);
        s.save_at("weight", u.weight);
        s.save_at("ratio", u.ratio);
        s.save_at("cumulative_nprops", u.cumulative_nprops);
        s.save_at("cumulative_naccs", u.cumulative_naccs);
        s.save_at("cumulative_nimps", u.cumulative_nimps);
        u.wrapped.save_at(s, "user");
    }

    /**
     * @brief Deserialize a simplemc::update.
     *
     * @details It deserializes all metadata except `name` and calls basic_update::load_at().
     *
     * @param s Serializer handle.
     * @param u Update to deserialize into.
     */
    friend void simplemc_load(const serializer_type& s, update& u) {
        s.load_at("inv_name", u.inv_name);
        s.load_at("weight", u.weight);
        s.load_at("ratio", u.ratio);
        s.load_at("cumulative_nprops", u.cumulative_nprops);
        s.load_at("cumulative_naccs", u.cumulative_naccs);
        s.load_at("cumulative_nimps", u.cumulative_nimps);
        u.wrapped.load_at(s, "user");
    }

    /**
     * @brief Serialize the user-input config of a simplemc::update.
     *
     * @details It serializes `weight` and calls basic_update::save_input_config_at().
     *
     * @param s Serializer handle.
     * @param u Update to serialize.
     */
    friend void simplemc_save_input_config(serializer_type& s, const update& u) {
        s.save_at("weight", u.weight);
        u.wrapped.save_input_config_at(s, "user");
    }

    /**
     * @brief Deserialize the user-input config of a simplemc::update.
     *
     * @details It deserializes `weight` and calls basic_update::load_input_config_at().
     *
     * @param s Serializer handle.
     * @param u Update to deserialize into.
     */
    friend void simplemc_load_input_config(const serializer_type& s, update& u) {
        s.try_load_at("weight", u.weight);
        u.wrapped.load_input_config_at(s, "user");
    }

    /**
     * @brief Collect simplemc::update objects from different MPI processes.
     *
     * @details It all-reduces the six counter fields and calls basic_update::mpi_collect().
     *
     * @param comm simplemc::mpi::communicator object.
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
