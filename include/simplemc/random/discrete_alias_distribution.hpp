/**
 * @file discrecte_alias_distribution.hpp
 * @brief Discrete distribution based on Walker-Alias-Algorithm.
 */

#ifndef SIMPLEMC_RANDOM_DISCRETE_ALIAS_DISTRIBUTION_HPP
#define SIMPLEMC_RANDOM_DISCRETE_ALIAS_DISTRIBUTION_HPP

#include <simplemc/random/generate_discrete.hpp>
#include <simplemc/utils/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <initializer_list>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace simplemc {

/* Forward declarations. */
template <integer_only T>
class discrete_alias_distribution;

namespace detail {

/**
 * @brief Parameter type of discrete_alias_distribution.
 *
 * @tparam T Integral type.
 */
template <integer_only T>
class dad_param_type {
public:
    /**
     * @brief Type of distribution.
     */
    using distribution_type = discrete_alias_distribution<T>;

    /**
     * @brief Default constructor for dad_param_type.
     */
    dad_param_type();

    /**
     * @brief Constructs a dad_param_type from a range of weights.
     *
     * @tparam InputIt Iterator type.
     * @param first Beginning of range.
     * @param last End of range.
     */
    template <typename InputIt>
    dad_param_type(InputIt first, InputIt last);

    /**
     * @brief Constructs a dad_param_type from a list of weights.
     *
     * @param list Initializer list of weights.
     */
    dad_param_type(std::initializer_list<double> list);

    /**
     * @brief Constructs a dad_param_type such that the weights are given as
     * w_i = unary_op(xmin + d/2 + i*d) with d = (xmax - xmin)/n and
     * i = 0, ..., n.
     *
     * @tparam UnaryOp Unary operation.
     * @param n Number of bin/weights.
     * @param xmin Lower bound.
     * @param xmax Upper bound.
     * @param unary_op Unary operation.
     */
    template <typename UnaryOp>
    dad_param_type(std::size_t n, double xmin, double xmax, UnaryOp unary_op);

    /**
     * @brief Get probabilities.
     *
     * @return Probability vector.
     */
    [[nodiscard]] const std::vector<double>& probabilities() const { return probs_; }

    /* Friend declarations. */
    friend class simplemc::discrete_alias_distribution<T>;

private:
    /**
     * @brief Normalize probabilities.
     */
    void normalize();

    std::vector<double> probs_;
};

template <integer_only T>
dad_param_type<T>::dad_param_type() : probs_(1, 1.0) {}

template <integer_only T>
template <typename InputIt>
dad_param_type<T>::dad_param_type(InputIt first, InputIt last) : probs_(first, last) {
    normalize();
}

template <integer_only T>
dad_param_type<T>::dad_param_type(std::initializer_list<double> list) : probs_(list) {
    normalize();
}

template <integer_only T>
template <typename UnaryOp>
dad_param_type<T>::dad_param_type(std::size_t n, double xmin, double xmax, UnaryOp unary_op) {
    std::size_t nw = n == 0 ? 1 : n;
    double d = (xmax - xmin) / static_cast<double>(nw);
    probs_.reserve(nw);
    for (std::size_t i = 0; i < nw; ++i) {
        probs_.push_back(unary_op(xmin + static_cast<double>(i) * d + 0.5 * d));
    }
    normalize();
}

template <integer_only T>
void dad_param_type<T>::normalize() {
    if (probs_.size() < 2) {
        probs_.clear();
        probs_.push_back(1.0);
        return;
    }
    double norm = std::accumulate(probs_.begin(), probs_.end(), 0.0);
    for (auto& p : probs_) {
        p /= norm;
    }
}

/**
 * @brief Write textual representation of dad_param_type to ostream.
 *
 * @details Throws an exception, if writing to ostream fails.
 *
 * @param os std::ostream.
 * @param param Parameters.
 * @return Reference to the std::ostream object.
 */
template <integer_only T>
std::ostream& operator<<(std::ostream& os, const dad_param_type<T>& param) {
    auto prec = os.precision();
    auto check_os = [](const std::ostream& os) {
        if (!os) {
            throw simplemc_exception("Error writing dad_param_type to ostream");
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
 * @brief Read textual representation of dad_param_type from istream.
 *
 * @details Throws an exception, if reading from istream fails.
 *
 * @tparam T Integral type.
 * @param is std::istream.
 * @param param Parameters.
 * @return Reference to the std::istream object.
 */
template <integer_only T>
std::istream& operator>>(std::istream& is, dad_param_type<T>& param) {
    auto check_is = [](const std::istream& is) {
        if (!is) {
            throw simplemc_exception("Error reading dad_param_type from istream");
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
    param = dad_param_type<T> { probs.begin(), probs.end() };
    return is;
}

/**
 * @brief Compare two dad_param_type objects for equality.
 *
 * @tparam T Integral type.
 * @param lhs Parameters #1.
 * @param rhs Parameters #2.
 * @return `true` if their probabilities are equal.
 */
template <integer_only T>
[[nodiscard]] bool operator==(const dad_param_type<T>& lhs, const dad_param_type<T>& rhs) {
    return lhs.probabilities() == rhs.probabilities();
}

/**
 * @brief Compare two dad_param_type objects for inequality.
 *
 * @tparam T Integral type.
 * @param lhs Parameters #1.
 * @param rhs Parameters #2.
 * @return `true` if their probabilities are not equal.
 */
template <integer_only T>
[[nodiscard]] bool operator!=(const dad_param_type<T>& lhs, const dad_param_type<T>& rhs) {
    return !(lhs == rhs);
}

} // namespace detail

/**
 * @brief Discrete distribution using the Walker-Alias-Algorithm based on
 * std::discrete_distribution and boost::discrete_distribution.
 *
 * @details Satisfies the requirements for a C++ RandomNumberDistribution.
 *
 * @tparam T Integral type.
 */
template <integer_only T>
class discrete_alias_distribution {
public:
    /**
     * @brief Type of the random number generated.
     */
    using result_type = std::size_t;

    /**
     * @brief Parameter type of the distribution.
     */
    using param_type = detail::dad_param_type<T>;

    /**
     * @brief Default constructor for a discrete_alias_distribution.
     */
    discrete_alias_distribution();

    /**
     * @brief Construct a discrete_alias_distribution from a range of weights.
     *
     * @tparam InputIt Iterator type.
     * @param first Beginning of range.
     * @param last End of range.
     */
    template <typename InputIt>
    discrete_alias_distribution(InputIt first, InputIt last);

    /**
     * @brief Construct a discrete_alias_distribution from a list of weights.
     *
     * @param list Initializer list of weights.
     */
    discrete_alias_distribution(std::initializer_list<double> list);

    /**
     * @brief Construct a discrete_alias_distribution such that the weights are
     * given as w_i = unary_op(xmin + d/2 + i*d) with d = (xmax - xmin)/n and
     * i = 0, ..., n.
     *
     * @tparam UnaryOp Unary operation.
     * @param n Number of bin/weights.
     * @param xmin Lower bound.
     * @param xmax Upper bound.
     * @param unary_op Unary operation.
     */
    template <typename UnaryOp>
    discrete_alias_distribution(std::size_t n, double xmin, double xmax, UnaryOp unary_op);

    /**
     * @brief Construct a discrete_alias_distribution from a given param_type.
     *
     * @param param Parameter specifying the weights.
     */
    explicit discrete_alias_distribution(const param_type& param);

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
     * @brief Get probabilities.
     *
     * @return Probability vector.
     */
    [[nodiscard]] const std::vector<double>& probabilities() const { return param_.probabilities(); }

    /**
     * @brief Get parameter set of the distribution.
     *
     * @return Parameters.
     */
    [[nodiscard]] const param_type& param() const { return param_; }

    /**
     * @brief Set the parameters for this distribution.
     *
     * @param param Parameter object.
     */
    void param(const param_type& param) {
        param_ = param;
        initialize();
    }

    /**
     * @brief Reset internal state of distribution.
     */
    void reset() {}

    /**
     * @brief Generate random integer.
     *
     * @tparam RNG Random number generator.
     * @param rng RNG object.
     * @return Random int distributed according to weights/probabilities.
     */
    template <typename RNG>
    result_type operator()(RNG& rng) const;

    /**
     * @brief Generate random integer distributed according to param.
     *
     * @details Uses a linear search instead of Walker-Alias.
     *
     * @tparam RNG Random number generator.
     * @param rng RNG object.
     * @param param Parameter object specifying probabilities.
     * @return Random int distributed according to param.
     */
    template <typename RNG>
    result_type operator()(RNG& rng, const param_type& param) const;

private:
    /**
     * @brief Initialize probablities and alias table.
     */
    void initialize();

    param_type param_;
    std::vector<double> probs_;
    std::vector<int> alias_;
};

template <integer_only T>
discrete_alias_distribution<T>::discrete_alias_distribution() : param_() {
    initialize();
}

template <integer_only T>
template <typename InputIt>
discrete_alias_distribution<T>::discrete_alias_distribution(InputIt first, InputIt last) : param_(first, last) {
    initialize();
}

template <integer_only T>
discrete_alias_distribution<T>::discrete_alias_distribution(std::initializer_list<double> list) : param_(list) {
    initialize();
}

template <integer_only T>
template <typename UnaryOp>
discrete_alias_distribution<T>::discrete_alias_distribution(std::size_t n, double xmin, double xmax, UnaryOp unary_op) :
    param_(n, xmin, xmax, unary_op) {
    initialize();
}

template <integer_only T>
discrete_alias_distribution<T>::discrete_alias_distribution(const param_type& param) : param_(param) {
    initialize();
}

template <integer_only T>
void discrete_alias_distribution<T>::initialize() {
    auto size = param_.probs_.size();
    probs_ = param_.probs_;
    alias_.resize(size, 0);

    // helper vectors/arrays
    std::vector<int> above;
    std::vector<int> below;
    above.reserve(size);
    below.reserve(size);
    for (std::size_t i = 0; i < size; ++i) {
        probs_[i] *= size;
        if (probs_[i] < 1.0) {
            below.push_back(static_cast<int>(i));
        } else {
            above.push_back(static_cast<int>(i));
        }
    }

    // create tables
    while (below.size() != 0 && above.size() != 0) {
        auto j = below.back();
        auto k = above.back();
        alias_[j] = k;
        below.pop_back();
        probs_[k] += probs_[j] - 1;
        if (probs_[k] < 1) {
            above.pop_back();
            below.push_back(k);
        }
    }
    for (std::size_t i = 0; i < below.size(); ++i) {
        probs_[below[i]] = 1;
    }
    for (std::size_t i = 0; i < above.size(); ++i) {
        probs_[above[i]] = 1;
    }
}

template <integer_only T>
template <typename Engine>
discrete_alias_distribution<T>::result_type discrete_alias_distribution<T>::operator()(Engine& eng) const {
    auto j = std::uniform_int_distribution<std::size_t>(0, probs_.size() - 1)(eng);
    auto r2 = std::uniform_real_distribution<double>(0, 1)(eng);
    return (r2 <= probs_[j] ? j : alias_[j]);
}

template <integer_only T>
template <typename Engine>
discrete_alias_distribution<T>::result_type discrete_alias_distribution<T>::operator()(
    Engine& eng, const param_type& param) const {
    std::vector<double> accprobs;
    accprobs.reserve(param.probs_.size());
    std::partial_sum(param.probs_.begin(), param.probs_.end(), std::back_inserter(accprobs));
    accprobs.back() = 1.0;
    return detail::generate_discrete_ls<result_type>(eng, accprobs);
}

/**
 * @brief Write a textual representation of a discrete_alias_distribution to ostream.
 *
 * @details Throws an exception, if writing to ostream fails.
 *
 * @tparam T Integral type.
 * @param os Reference to ostream.
 * @param dd Distribution to be written.
 * @return Reference to ostream.
 */
template <integer_only T>
std::ostream& operator<<(std::ostream& os, const discrete_alias_distribution<T>& dad) {
    if (!(os << dad.param())) {
        throw simplemc_exception("Error writing discrete_alias_distribution to ostream");
    }
    return os;
}

/**
 * @brief Restore discrete_alias_distribution from istream.
 *
 * @details Throws an exception, if reading from istream fails.
 *
 * @tparam T Integral type.
 * @param is Reference to istream.
 * @param dd Distribution to be read into.
 * @return Reference to istream.
 */
template <integer_only T>
std::istream& operator>>(std::istream& is, discrete_alias_distribution<T>& dad) {
    typename discrete_alias_distribution<T>::param_type parm;
    if (is >> parm) {
        dad.param(parm);
    } else {
        throw simplemc_exception("Error reading discrete_alias_distribution from istream");
    }
    return is;
}

/**
 * @brief Compare two discrete_alias_distributions for equality.
 *
 * @tparam T Integral type.
 * @param lhs Distribution #1.
 * @param rhs Distribution #2.
 * @return `true` if the parameters of the distributions are the same.
 */
template <integer_only T>
[[nodiscard]] bool operator==(const discrete_alias_distribution<T>& lhs, const discrete_alias_distribution<T>& rhs) {
    return lhs.param() == rhs.param();
}

/**
 * @brief Compare two discrete_alias_distributions for inequality.
 *
 * @tparam T Integral type.
 * @param lhs Distribution #1.
 * @param rhs Distribution #2.
 * @return `true` if the parameters of the distributions are distinct.
 */
template <integer_only T>
[[nodiscard]] bool operator!=(const discrete_alias_distribution<T>& lhs, const discrete_alias_distribution<T>& rhs) {
    return !(lhs == rhs);
}

} // namespace simplemc

#endif // SIMPLEMC_RANDOM_DISCRETE_ALIAS_DISTRIBUTION_HPP
