/**
 * @file
 * @brief Discrete distribution using a linear search.
 */

#ifndef SIMPLEMC_RANDOM_DISCRETE_DISTRIBUTION_HPP
#define SIMPLEMC_RANDOM_DISCRETE_DISTRIBUTION_HPP

#include <simplemc/random/generate_discrete.hpp>
#include <simplemc/utils/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <vector>

namespace simplemc {

/**
 * @addtogroup simplemc-random
 * @{
 */

/* Forward declarations. */
template <integer_only T>
class discrete_distribution;

namespace detail {

/**
 * @addtogroup simplemc-random
 * @{
 */

/**
 * @brief Parameter type of simplemc::discrete_distribution.
 *
 * @tparam T Integral type.
 */
template <integer_only T>
class dd_param_type {
public:
    /**
     * @brief Type of distribution.
     */
    using distribution_type = discrete_distribution<T>;

    /**
     * @brief Default constructor.
     */
    dd_param_type();

    /**
     * @brief Construct distribution parameters from a range of weights.
     *
     * @tparam InputIt Iterator type.
     * @param first Beginning of range.
     * @param last End of range.
     */
    template <typename InputIt>
    dd_param_type(InputIt first, InputIt last);

    /**
     * @brief Construct distribution parameters from a list of weights.
     *
     * @param list Initializer list of weights.
     */
    dd_param_type(std::initializer_list<double> list);

    /**
     * @brief Construct distribution parameters such that the weights are given as
     * `w_i = unary_op(xmin + d/2 + i*d)` with `d = (xmax - xmin)/n` and `i = 0, ..., n`.
     *
     * @tparam UnaryOp Unary operation.
     * @param n Number of bins/weights.
     * @param xmin Lower bound.
     * @param xmax Upper bound.
     * @param unary_op Unary operation.
     */
    template <typename UnaryOp>
    dd_param_type(std::size_t n, double xmin, double xmax, UnaryOp unary_op);

    /**
     * @brief Get the probabilities.
     *
     * @return Probability vector.
     */
    [[nodiscard]] const std::vector<double>& probabilities() const { return probs_; }

    /* Friend declarations. */
    friend class simplemc::discrete_distribution<T>;

private:
    /**
     * @brief Normalize probabilities and partial sum accumulated probabilties.
     */
    void initialize();

    std::vector<double> probs_;
    std::vector<double> accprobs_;
};

/// @cond
template <integer_only T>
dd_param_type<T>::dd_param_type() : probs_(1, 1.0) {
    initialize();
}

template <integer_only T>
template <typename InputIt>
dd_param_type<T>::dd_param_type(InputIt first, InputIt last) : probs_(first, last) {
    initialize();
}

template <integer_only T>
dd_param_type<T>::dd_param_type(std::initializer_list<double> list) : probs_(list) {
    initialize();
}

template <integer_only T>
template <typename UnaryOp>
dd_param_type<T>::dd_param_type(std::size_t n, double xmin, double xmax, UnaryOp unary_op) {
    std::size_t nw = (n == 0 ? 1 : n);
    double d = (xmax - xmin) / static_cast<double>(nw);
    probs_.reserve(nw);
    for (std::size_t i = 0; i < nw; ++i) {
        probs_.push_back(unary_op(xmin + static_cast<double>(i) * d + 0.5 * d));
    }
    initialize();
}

template <integer_only T>
void dd_param_type<T>::initialize() {
    if (probs_.size() < 2) {
        probs_.clear();
        probs_.push_back(1.0);
        accprobs_ = probs_;
        return;
    }
    double norm = std::accumulate(probs_.begin(), probs_.end(), 0.0);
    for (auto& p : probs_) {
        p /= norm;
    }
    accprobs_.reserve(probs_.size());
    std::partial_sum(probs_.begin(), probs_.end(), std::back_inserter(accprobs_));
    accprobs_.back() = 1.0;
}
/// @endcond

/**
 * @brief Write a textual representation of distribution parameters to std::ostream.
 *
 * @details Throws an exception, if writing to std::ostream fails.
 *
 * @tparam T Integral type.
 * @param os std::ostream to write to.
 * @param param Distribution parameters to be written.
 * @return Reference to the std::ostream object.
 */
template <integer_only T>
std::ostream& operator<<(std::ostream& os, const dd_param_type<T>& param) {
    auto prec = os.precision();
    auto check_os = [](const std::ostream& os) {
        if (!os) {
            throw simplemc_exception("Error writing dd_param_type to ostream");
        }
    };
    os.precision(std::numeric_limits<double>::max_digits10);
    check_os(os << param.probabilities().size());
    for (const auto& p : param.probabilities()) {
        check_os(os << ' ' << p);
    }
    os.precision(prec);
    return os;
}

/**
 * @brief Read a textual representation of distribution parameters from std::istream.
 *
 * @details Throws an exception, if reading from std::istream fails.
 *
 * @tparam T Integral type.
 * @param is std::istream to read from.
 * @param param Distribution parameters to read into.
 * @return Reference to the std::istream object.
 */
template <integer_only T>
std::istream& operator>>(std::istream& is, dd_param_type<T>& param) {
    auto check_is = [](const std::istream& is) {
        if (!is) {
            throw simplemc_exception("Error reading dd_param_type from istream");
        }
    };
    std::size_t size {};
    check_is(is >> size);
    std::vector<double> probs;
    probs.reserve(size);
    for (std::size_t i = 0; i < size; ++i) {
        double p {};
        check_is(is >> std::ws >> p);
        probs.push_back(p);
    }
    param = dd_param_type<T> { probs.begin(), probs.end() };
    return is;
}

/**
 * @brief Compare two distribution parameters for equality.
 *
 * @tparam T Integral type.
 * @param lhs Left-hand side distribution parameters.
 * @param rhs Right-hand side distribution parameters.
 * @return True if their probabilities are equal.
 */
template <integer_only T>
[[nodiscard]] bool operator==(const dd_param_type<T>& lhs, const dd_param_type<T>& rhs) {
    return lhs.probabilities() == rhs.probabilities();
}

/**
 * @brief Compare two distribution parameters for inequality.
 *
 * @tparam T Integral type.
 * @param lhs Left-hand side distribution parameters.
 * @param rhs Right-hand side distribution parameters.
 * @return True if their probabilities are not equal.
 */
template <integer_only T>
[[nodiscard]] bool operator!=(const dd_param_type<T>& lhs, const dd_param_type<T>& rhs) {
    return !(lhs == rhs);
}

/** @} */

} // namespace detail

/**
 * @brief Discrete distribution based on std::discrete_distribution and
 * boost::discrete_distribution.
 *
 * @details Satisfies the requirements for a C++ RandomNumberDistribution.
 *
 * @tparam T Integral type.
 */
template <integer_only T>
class discrete_distribution {
public:
    /**
     * @brief Type of the random number generated.
     */
    using result_type = T;

    /**
     * @brief Parameter type of the distribution.
     */
    using param_type = detail::dd_param_type<T>;

    /**
     * @brief Default constructor.
     */
    discrete_distribution() = default;

    /**
     * @brief Construct a distribution from a range of weights.
     *
     * @tparam InputIt Iterator type.
     * @param first Beginning of range.
     * @param last End of range.
     */
    template <typename InputIt>
    discrete_distribution(InputIt first, InputIt last) : param_(first, last) {}

    /**
     * @brief Construct a distribution from a list of weights.
     *
     * @param list Initializer list of weights.
     */
    discrete_distribution(std::initializer_list<double> list) : param_(list) {};

    /**
     * @brief Construct a distribution such that the weights are given as
     * `w_i = unary_op(xmin + d/2 + i*d)` with `d = (xmax - xmin)/n` and `i = 0, ..., n`.
     *
     * @tparam UnaryOp Unary operation.
     * @param n Number of bins/weights.
     * @param xmin Lower bound.
     * @param xmax Upper bound.
     * @param unary_op Unary operation.
     */
    template <typename UnaryOp>
    discrete_distribution(std::size_t n, double xmin, double xmax, UnaryOp unary_op) :
        param_(n, xmin, xmax, unary_op) {}

    /**
     * @brief Construct a distribution from given distribution parameters.
     *
     * @param param Parameters specifying the weights.
     */
    explicit discrete_distribution(const param_type& param) : param_(param) {}

    /**
     * @brief Lower bound of the distribution.
     *
     * @return Zero.
     */
    [[nodiscard]] result_type min() const { return 0; }

    /**
     * @brief Upper bound of the distribution.
     *
     * @return Number of weights/probabilities given at construction.
     */
    [[nodiscard]] result_type max() const { return static_cast<result_type>(param_.probs_.size() - 1); }

    /**
     * @brief Get the probabilities.
     *
     * @return Probability vector.
     */
    [[nodiscard]] const std::vector<double>& probabilities() const { return param_.probabilities(); }

    /**
     * @brief Get the distribution parameters.
     *
     * @return Distribution parameters.
     */
    [[nodiscard]] const param_type& param() const { return param_; }

    /**
     * @brief Set the distribution parameters
     *
     * @param param Distribution paramters.
     */
    void param(const param_type& param) { param_ = param; }

    /**
     * @brief Reset internal state of distribution.
     */
    void reset() {}

    /**
     * @brief Generate random integer.
     *
     * @tparam RNG Random number generator.
     * @param rng RNG object.
     * @return Random integer distributed according to weights/probabilities.
     */
    template <typename RNG>
    result_type operator()(RNG& rng) const {
        return detail::generate_discrete_ls<result_type>(rng, param_.accprobs_);
    }

    /**
     * @brief Generate random integer distributed according to given distribution parameters.
     *
     * @tparam RNG Random number generator.
     * @param rng RNG object.
     * @param param Distribution parameters specifying probabilities.
     * @return Random integer distributed according to the given distribution parameters.
     */
    template <typename RNG>
    result_type operator()(RNG& rng, const param_type& param) const {
        return detail::generate_discrete_ls<result_type>(rng, param.accprobs_);
    }

private:
    param_type param_;
};

/**
 * @brief Write a textual representation of a distribution to std::ostream.
 *
 * @details Throws an exception, if writing to std::ostream fails.
 *
 * @tparam T Integral type.
 * @param os std::ostream to write to.
 * @param dd Distribution to be written.
 * @return Reference to the std::ostream object.
 */
template <integer_only T>
std::ostream& operator<<(std::ostream& os, const discrete_distribution<T>& dd) {
    if (!(os << dd.param())) {
        throw simplemc_exception("Error writing discrete_distribution to ostream");
    }
    return os;
}

/**
 * @brief Read a textual representations of a distribution from std::istream.
 *
 * @details Throws an exception, if reading from std::istream fails.
 *
 * @tparam T Integral type.
 * @param is std::istream to read from.
 * @param dd Distribution to be read into.
 * @return Reference to the std::istream object.
 */
template <integer_only T>
std::istream& operator>>(std::istream& is, discrete_distribution<T>& dd) {
    typename discrete_distribution<T>::param_type parm;
    if (is >> parm) {
        dd.param(parm);
    } else {
        throw simplemc_exception("Error reading discrete_distribution from istream");
    }
    return is;
}

/**
 * @brief Compare two distributions for equality.
 *
 * @tparam T Integral type.
 * @param lhs Left-hand side distribution.
 * @param rhs Right-hand side distribution.
 * @return True if the parameters of the distributions are the same.
 */
template <integer_only T>
[[nodiscard]] bool operator==(const discrete_distribution<T>& lhs, const discrete_distribution<T>& rhs) {
    return lhs.param() == rhs.param();
}

/**
 * @brief Compare two distributions for inequality.
 *
 * @tparam T Integral type.
 * @param lhs Left-hand side distribution.
 * @param rhs Right-hand side distribution.
 * @return True if the parameters of the distributions are distinct.
 */
template <integer_only T>
[[nodiscard]] bool operator!=(const discrete_distribution<T>& lhs, const discrete_distribution<T>& rhs) {
    return !(lhs == rhs);
}

/** @} */

} // namespace simplemc

#endif // SIMPLEMC_RANDOM_DISCRETE_DISTRIBUTION_HPP
