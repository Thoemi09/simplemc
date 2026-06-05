/**
 * @file
 * @brief Polymorphic value type holding a Monte Carlo measurement.
 */

#ifndef SIMPLEMC_MC_MEASUREMENT_HPP
#define SIMPLEMC_MC_MEASUREMENT_HPP

#include <simplemc/mc/concepts.hpp>

#include <concepts>
#include <memory>
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
 * See @ref simplemc-mc for a description of the implementation strategy.
 *
 * @note The wrapper takes the user measurement by value and does not expose it back. To access result 
 * data (e.g. an accumulator) after a run, have the user type hold a pointer or reference to 
 * externally-owned storage rather than the data itself.
 */
class measurement {
public:
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

private:
    // Hidden polymorphic base — the "concept" half of the concept-model idiom.
    struct interface {
        virtual ~interface() = default;
        virtual void measure() = 0;
        [[nodiscard]] virtual std::unique_ptr<interface> clone() const = 0;
    };

    // Hidden template derived - the "model" half of the concept-model idiom.
    // Note: Each wrapped type M gets its own model type.
    template <class M>
    struct model final : interface {
        M concrete;
        explicit model(M m) : concrete { std::move(m) } {}
        void measure() override { concrete.measure(); }
        [[nodiscard]] std::unique_ptr<interface> clone() const override { return std::make_unique<model>(*this); }
    };

private:
    std::unique_ptr<interface> pimpl_;
};

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_MC_MEASUREMENT_HPP
