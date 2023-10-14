/**
 * @file discrete_distribution.hpp
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

/* Forward declarations. */
template <integer_for_random T>
class discrete_distribution;

namespace detail {

/**
 * @brief Parameter type of discrete_distribution.
 *
 * @tparam T Integral type.
 */
template <integer_for_random T>
class dd_param_type {
public:
    /**
     * @brief Type of distribution.
     */
    using distribution_type = discrete_distribution<T>;

    /**
     * @brief Default constructor for dd_param_type.
     */
    dd_param_type();

    /**
     * @brief Construct a dd_param_type from a range of weights.
     *
     * @tparam InputIt Iterator type.
     * @param first Beginning of range.
     * @param last End of range.
     */
    template <typename InputIt>
    dd_param_type(InputIt first, InputIt last);

    /**
     * @brief Construct a dd_param_type from a list of weights.
     *
     * @param list Initializer list of weights.
     */
    dd_param_type(std::initializer_list<double> list);

    /**
     * @brief Construct a dd_param_type such that the weights are given as
     * w_i = unary_op(xmin + d/2 + i*d) with d = (xmax - xmin)/n and i = 0, ..., n.
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
     * @brief Get probabilities.
     *
     * @return Probability vector.
     */
    const std::vector<double>& probabilities() const { return probs_; }

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

template <integer_for_random T>
dd_param_type<T>::dd_param_type() : probs_(1, 1.0) {
    initialize();
}

template <integer_for_random T>
template <typename InputIt>
dd_param_type<T>::dd_param_type(InputIt first, InputIt last) : probs_(first, last) {
    initialize();
}

template <integer_for_random T>
dd_param_type<T>::dd_param_type(std::initializer_list<double> list) : probs_(list) {
    initialize();
}

template <integer_for_random T>
template <typename UnaryOp>
dd_param_type<T>::dd_param_type(std::size_t n, double xmin, double xmax, UnaryOp unary_op) {
    std::size_t nw = (n == 0 ? 1 : n);
    double d = (xmax - xmin) / nw;
    probs_.reserve(nw);
    for (std::size_t i = 0; i < nw; ++i) {
        probs_.push_back(unary_op(xmin + i * d + 0.5 * d));
    }
    initialize();
}

template <integer_for_random T>
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

/**
 * @brief Write textual representation of dd_param_type to ostream. Throws an exception, if
 * writing to ostream fails.
 *
 * @tparam T Integral type.
 * @param os std::ostream.
 * @param param Parameters.
 * @return Reference to the std::ostream object.
 */
template <integer_for_random T>
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
 * @brief Read textual representation of dd_param_type from istream. Throws an exception, if
 * reading from istream fails.
 *
 * @tparam T Integral type.
 * @param is std::istream.
 * @param param Parameters.
 * @return Reference to the std::istream object.
 */
template <integer_for_random T>
std::istream& operator>>(std::istream& is, dd_param_type<T>& param) {
    auto check_is = [](const std::istream& is) {
        if (!is) {
            throw simplemc_exception("Error reading dd_param_type from istream");
        }
    };
    std::size_t size;
    check_is(is >> size);
    std::vector<double> probs;
    probs.reserve(size);
    for (std::size_t i = 0; i < size; ++i) {
        double p;
        check_is(is >> std::ws >> p);
        probs.push_back(p);
    }
    param = dd_param_type<T> { probs.begin(), probs.end() };
    return is;
}

/**
 * @brief Compare two dd_param_type objects for equality.
 *
 * @tparam T Integral type.
 * @param lhs Parameters #1.
 * @param rhs Parameters #2.
 * @return True if their probabilities are equal.
 */
template <integer_for_random T>
bool operator==(const dd_param_type<T>& lhs, const dd_param_type<T>& rhs) {
    return lhs.probabilities() == rhs.probabilities();
}

/**
 * @brief Compare two dd_param_type objects for inequality.
 *
 * @tparam T Integral type.
 * @param lhs Parameters #1.
 * @param rhs Parameters #2.
 * @return True if their probabilities are not equal.
 */
template <integer_for_random T>
bool operator!=(const dd_param_type<T>& lhs, const dd_param_type<T>& rhs) {
    return !(lhs == rhs);
}

} // namespace detail

/**
 * @brief Discrete distribution based on std::discrete_distribution and
 * boost::discrete_distribution. Only to be used with 64-bit RNGs.
 *
 * @details Satisfies the requirements for a C++ RandomNumberDistribution.
 *
 * @tparam T Integral type.
 */
template <integer_for_random T>
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
     * @brief Default constructor for a discrete distribution.
     */
    discrete_distribution() = default;

    /**
     * @brief Construct a discrete distribution from a range of weights.
     *
     * @tparam InputIt Iterator type.
     * @param first Beginning of range.
     * @param last End of range.
     */
    template <typename InputIt>
    discrete_distribution(InputIt first, InputIt last) : param_(first, last) {}

    /**
     * @brief Construct a discrete distribution from a list of weights.
     *
     * @param list Initializer list of weights.
     */
    discrete_distribution(std::initializer_list<double> list) : param_(list) {};

    /**
     * @brief Construct a discrete_distribution such that the weights are given
     * as w_i = unary_op(xmin + d/2 + i*d) with d = (xmax - xmin)/n and
     * i = 0, ..., n.
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
     * @brief Construct a discrete distribution from a given param_type.
     *
     * @param param Parameter specifying the weights.
     */
    explicit discrete_distribution(const param_type& param) : param_(param) {}

    /**
     * @brief Lower bound of the distribution.
     *
     * @return Zero.
     */
    result_type min() const { return 0; }

    /**
     * @brief Upper bound of the distribution.
     *
     * @return Number of weights/probabilities given at construction.
     */
    result_type max() const { return static_cast<result_type>(param_.probs_.size() - 1); }

    /**
     * @brief Get probabilities.
     *
     * @return Probability vector.
     */
    const std::vector<double>& probabilities() const { return param_.probabilities(); }

    /**
     * @brief Get parameter set of the distribution.
     *
     * @return Parameters.
     */
    const param_type& param() const { return param_; }

    /**
     * @brief Set the parameters for this distribution.
     *
     * @param param Parameter object.
     */
    void param(const param_type& param) { param_ = param; }

    /**
     * @brief Reset internal state of distribution.
     */
    void reset() {}

    /**
     * @brief Generate random integer.
     *
     * @tparam Engine RNG.
     * @param eng RNG.
     * @return Random integer distributed according to weights/probabilities.
     */
    template <typename Engine>
    result_type operator()(Engine& eng) const {
        return detail::generate_discrete_ls<result_type>(eng, param_.accprobs_);
    }

    /**
     * @brief Generate random integer distributed according to param.
     *
     * @tparam Engine RNG.
     * @param eng RNG.
     * @param param Parameter object specifying probabilities.
     * @return Random integer distributed according to param.
     */
    template <typename Engine>
    result_type operator()(Engine& eng, const param_type& param) const {
        return detail::generate_discrete_ls<result_type>(eng, param.accprobs_);
    }

private:
    param_type param_;
};

/**
 * @brief Write a textual representation of a discrete_distribution to ostream. Throws an
 * exception, if writing to ostream fails.
 *
 * @tparam T Integral type.
 * @param os Reference to ostream.
 * @param dd Distribution to be written.
 * @return Reference to ostream.
 */
template <integer_for_random T>
std::ostream& operator<<(std::ostream& os, const discrete_distribution<T>& dd) {
    if (!(os << dd.param())) {
        throw simplemc_exception("Error writing discrete_distribution to ostream");
    }
    return os;
}

/**
 * @brief Restore discrete_distribution from istream. Throws an exception, if reading from
 * istream fails.
 *
 * @tparam T Integral type.
 * @param is Reference to istream.
 * @param dd Distribution to be read into.
 * @return Reference to istream.
 */
template <integer_for_random T>
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
 * @brief Compare two discrete_distributions for equality.
 *
 * @tparam T Integral type.
 * @param lhs Distribution #1.
 * @param rhs Distribution #2.
 * @return True if the parameters of the distributions are the same.
 */
template <integer_for_random T>
bool operator==(const discrete_distribution<T>& lhs, const discrete_distribution<T>& rhs) {
    return lhs.param() == rhs.param();
}

/**
 * @brief Compare two discrete_distributions for inequality.
 *
 * @tparam T Integral type.
 * @param lhs Distribution #1.
 * @param rhs Distribution #2.
 * @return True if the parameters of the distributions are distinct.
 */
template <integer_for_random T>
bool operator!=(const discrete_distribution<T>& lhs, const discrete_distribution<T>& rhs) {
    return !(lhs == rhs);
}

} // namespace simplemc

#endif // SIMPLEMC_RANDOM_DISCRETE_DISTRIBUTION_H
