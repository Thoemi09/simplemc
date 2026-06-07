/**
 * @file
 * @brief Polymorphic value type holding a Monte Carlo measurement.
 */

#ifndef SIMPLEMC_MC_MEASUREMENT_HPP
#define SIMPLEMC_MC_MEASUREMENT_HPP

#include <simplemc/mc/concepts.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/serialize/json/json_serializer.hpp>

#include <concepts>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>

namespace simplemc {

/**
 * @addtogroup simplemc-mc
 * @{
 */

/**
 * @brief A Monte Carlo measurement.
 *
 * @details A measurement is an observer of the simulation state that the driver invokes once per
 * measurement cycle to record an observable, accumulate a sample, write to a histogram, etc.
 *
 * The library accepts any user-defined type that satisfies the simplemc::mc_measurement concept (i.e.
 * exposes a callable `measure()` member) — there is no base class to inherit from. A user value is
 * wrapped by construction:
 *
 * @code{.cpp}
 * struct my_observer {
 *     void measure() { ... }
 * };
 *
 * simplemc::measurement m{ my_observer{} };
 * m.measure();
 * @endcode
 *
 * simplemc::measurement is a regular value type: copies are deep, moves are cheap, and it can be
 * stored directly in containers such as `std::vector<%measurement>` to hold a heterogeneous
 * collection of observers. Because copies are deep, the wrapped user type must be copy-constructible;
 * this is checked at construction time.
 *
 * The wrapper is templated on the serializer types `S1` and `S2` used for checkpointing and
 * input-config serialization, respectively:
 * - If the wrapped user type provides a serialization path through `S1`'s `%save_at()` /
 * `%load_at()`, the wrapper's save_at() / load_at() forward to it.
 * - If the wrapped user type satisfies simplemc::has_simplemc_save_input_config with `S2`, the
 * wrapper's save_input_config_at() / load_input_config_at() forward to the user type's ADL hooks.
 * Otherwise they are silent no-ops.
 *
 * See @ref simplemc-mc for a description of the implementation strategy.
 *
 * @note The wrapper takes the user measurement by value and does not expose it back. To access result
 * data (e.g. an accumulator) after a run, have the user type hold a pointer or reference to
 * externally-owned storage rather than the data itself.
 *
 * @tparam S1 Serializer type used for checkpoint serialization.
 * @tparam S2 Serializer type used for input-config serialization.
 */
template <serializer S1 = json_serializer, serializer S2 = json_serializer>
class measurement {
public:
    /**
     * @brief Type used for checkpoint serialization.
     */
    using cp_serializer_type = S1;

    /**
     * @brief Type used for input-config serialization.
     */
    using ic_serializer_type = S2;

    /**
     * @brief Wrap a user-defined measurement.
     *
     * @details The user type is passed by forwarding reference, so lvalues are copied and rvalues
     * are moved into the wrapper.
     *
     * The wrapped type must also be copy-constructible: simplemc::measurement is a value type with
     * deep-copy semantics and the internal model holds an instance by value. This is a storage
     * requirement of the wrapper, not part of the simplemc::mc_measurement role.
     *
     * @tparam M Type satisfying simplemc::mc_measurement and `std::copy_constructible`.
     * @param m Measurement to wrap.
     */
    template <class M>
        requires(!std::same_as<std::remove_cvref_t<M>, measurement>) && mc_measurement<std::remove_cvref_t<M>> &&
        std::copy_constructible<std::remove_cvref_t<M>>
    measurement(M&& m) : pimpl_ { std::make_unique<model<std::remove_cvref_t<M>>>(std::forward<M>(m)) } {}

    /**
     * @brief Copy constructor.
     *
     * @details Performs a deep copy: the new wrapper owns an independent copy of `other`'s wrapped
     * value, produced by the wrapped type's copy constructor. The two wrappers do not share state.
     *
     * @param other Wrapper to copy from.
     */
    measurement(const measurement& other) : pimpl_ { other.pimpl_->clone() } {}

    /**
     * @brief Default move constructor.
     */
    measurement(measurement&& other) noexcept = default;

    /**
     * @brief Copy assignment.
     *
     * @details Performs a deep copy of `other`'s wrapped value into `*this` (via the copy-and-swap
     * pattern), discarding `*this`'s previous wrapped value. After the call, the two wrappers do not
     * share state.
     *
     * @param other Wrapper to copy from.
     * @return Reference to `*this`.
     */
    measurement& operator=(const measurement& other) {
        measurement tmp { other };
        std::swap(pimpl_, tmp.pimpl_);
        return *this;
    }

    /**
     * @brief Default move assignment.
     */
    measurement& operator=(measurement&& other) noexcept = default;

    /**
     * @brief Default destructor.
     */
    ~measurement() = default;

    /**
     * @brief Take a measurement.
     */
    void measure() { pimpl_->measure(); }

    /**
     * @brief Serialize the wrapped user type.
     *
     * @details Forwards to the serializer's `save_at()` method. If the wrapped user type is not
     * serializable through cp_serializer_type, the call is a silent no-op.
     *
     * @param s Serializer handle.
     * @param key Sub-key to write into.
     */
    void save_at(cp_serializer_type& s, std::string_view key) const { pimpl_->save_at(s, key); }

    /**
     * @brief Deserialize the wrapped user type.
     *
     * @details Forwards to the serializer's `load_at()` method. If the wrapped user type is not
     * deserializable through cp_serializer_type, the call is a silent no-op.
     *
     * @param s Serializer handle.
     * @param key Sub-key to read from.
     */
    void load_at(const cp_serializer_type& s, std::string_view key) { pimpl_->load_at(s, key); }

    /**
     * @brief Serialize the wrapped user type as input config.
     *
     * @details Forwards to the wrapped user type's ADL hook. If the wrapped user type does not
     * provide such a hook, the call is a silent no-op — the entry simply contributes nothing to the
     * input-config output.
     *
     * @param s Serializer handle.
     * @param key Sub-key to write into.
     */
    void save_input_config_at(ic_serializer_type& s, std::string_view key) const {
        pimpl_->save_input_config_at(s, key);
    }

    /**
     * @brief Deserialize the wrapped user type from input config.
     *
     * @details Forwards to the wrapped user type's ADL hook. If the wrapped user type does not
     * provide such a hook, the call is a silent no-op — the entry simply contributes nothing to the
     * input-config output.
     *
     * @param s Serializer handle.
     * @param key Sub-key to read from.
     */
    void load_input_config_at(const ic_serializer_type& s, std::string_view key) {
        pimpl_->load_input_config_at(s, key);
    }

private:
    // Hidden polymorphic base — the "concept" half of the concept-model idiom.
    struct interface {
        virtual ~interface() = default;
        virtual void measure() = 0;
        [[nodiscard]] virtual std::unique_ptr<interface> clone() const = 0;
        virtual void save_at(cp_serializer_type& parent, std::string_view key) const = 0;
        virtual void load_at(const cp_serializer_type& parent, std::string_view key) = 0;
        virtual void save_input_config_at(ic_serializer_type& parent, std::string_view key) const = 0;
        virtual void load_input_config_at(const ic_serializer_type& parent, std::string_view key) = 0;
    };

    // Hidden template derived - the "model" half of the concept-model idiom.
    // Note: Each wrapped type M gets its own model type.
    template <class M>
    struct model final : interface {
        M concrete;
        explicit model(M m) : concrete { std::move(m) } {}
        void measure() override { concrete.measure(); }
        [[nodiscard]] std::unique_ptr<interface> clone() const override { return std::make_unique<model>(*this); }
        void save_at(cp_serializer_type& s, std::string_view key) const override {
            if constexpr (requires(cp_serializer_type& p, std::string_view k, const M& v) { p.save_at(k, v); }) {
                s.save_at(key, concrete);
            }
        }
        void load_at(const cp_serializer_type& s, std::string_view key) override {
            if constexpr (requires(const cp_serializer_type& p, std::string_view k, M& v) { p.load_at(k, v); }) {
                s.load_at(key, concrete);
            }
        }
        void save_input_config_at(ic_serializer_type& s, std::string_view key) const override {
            if constexpr (has_simplemc_save_input_config<M, ic_serializer_type>) {
                auto sub = s[key];
                simplemc_save_input_config(sub, concrete);
            }
        }
        void load_input_config_at(const ic_serializer_type& s, std::string_view key) override {
            if constexpr (has_simplemc_load_input_config<M, ic_serializer_type>) {
                const auto sub = s[key];
                simplemc_load_input_config(sub, concrete);
            }
        }
    };

private:
    std::unique_ptr<interface> pimpl_;
};

// Deduction guide: measurement m { my_meas{} } deduces to measurement<json_serializer>.
template <class M>
    requires mc_measurement<std::remove_cvref_t<M>> && std::copy_constructible<std::remove_cvref_t<M>>
measurement(M&&) -> measurement<>;

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_MEASUREMENT_HPP
