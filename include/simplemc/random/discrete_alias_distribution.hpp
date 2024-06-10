/**
 * @file
 * @brief Discrete distribution based on the Walker-Alias-Algorithm.
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

/**
 * @ingroup simplemc-random-dists
 * @brief Discrete distribution using the Walker-Alias algorithm based on
 * `std::discrete_distribution`.
 *
 * @details It satisfies the C++ named requirements: <a href="https://en.cppreference.com/w/cpp/
 * named_req/RandomNumberDistribution">RandomNumberDistribution</a> and uses the
 * <a href="https://en.wikipedia.org/wiki/Alias_method">Walker-Alias</a> algorithm to generate random
 * numbers.
 *
 * For more information, please take a look at <a href="https://en.cppreference.com/w/cpp/numeric/
 * random/discrete_distribution">std::discrete_distribution</a>.
 *
 * @tparam T Integral type.
 */
template <integer_only T>
class discrete_alias_distribution {
public:
    /**
     * @brief Type of the random number generated.
     */
    using result_type = T;

    /**
     * @brief Parameter type of a simplemc::discrete_alias_distribution.
     *
     * @details It stores the probabilities of the distribution.
     */
    class param_type {
    public:
        /**
         * @brief Type of distribution to which the parameter type belongs.
         */
        using distribution_type = discrete_alias_distribution<T>;

        /**
         * @brief Default constructor sets a single probability to 1.0.
         */
        param_type() : probs_(1, 1.0) {}

        /**
         * @brief Constructs distribution parameters from a range of weights.
         *
         * @details It normalizes the weights and stores the probabilities.
         *
         * @tparam InputIt Iterator type.
         * @param first Beginning of the range.
         * @param last End of the range.
         */
        template <typename InputIt>
        param_type(InputIt first, InputIt last) : probs_(first, last) {
            normalize();
        }

        /**
         * @brief Constructs distribution parameters from a list of weights.
         *
         * @details It normalizes the weights and stores the probabilities.
         *
         * @param list Initializer list of weights.
         */
        param_type(std::initializer_list<double> list) : probs_(list) { normalize(); }

        /**
         * @brief Constructs distribution parameters such that the weights are given as
         * `w_i = unary_op(xmin + d/2 + i*d)` with `d = (xmax - xmin)/n` and `i = 0, ..., n`.
         *
         * @tparam UnaryOp Unary operation.
         * @param n Number of weights.
         * @param xmin Lower bound.
         * @param xmax Upper bound.
         * @param unary_op Unary operation.
         */
        template <typename UnaryOp>
        param_type(std::size_t n, double xmin, double xmax, UnaryOp unary_op) {
            std::size_t nw = n == 0 ? 1 : n;
            double d = (xmax - xmin) / static_cast<double>(nw);
            probs_.reserve(nw);
            for (std::size_t i = 0; i < nw; ++i) {
                probs_.push_back(unary_op(xmin + static_cast<double>(i) * d + 0.5 * d));
            }
            normalize();
        }

        /**
         * @brief Get the probabilities.
         *
         * @return Probability vector.
         */
        [[nodiscard]] const std::vector<double>& probabilities() const { return probs_; }

        /**
         * @brief Compare two simplemc::discrete_alias_distribution::param_type objects for equality.
         *
         * @tparam T Integral type.
         * @param lhs Left-hand side distribution.
         * @param rhs Right-hand side distribution.
         * @return True if their probabilities are equal.
         */
        [[nodiscard]] friend bool operator==(const param_type& lhs, const param_type& rhs) {
            return lhs.probs_ == rhs.probs_;
        }

        /**
         * @brief Compare two simplemc::discrete_alias_distribution::param_type objects for
         * inequality.
         *
         * @tparam T Integral type.
         * @param lhs Left-hand side distribution.
         * @param rhs Right-hand side distribution.
         * @return True if their probabilities are not equal.
         */
        [[nodiscard]] friend bool operator!=(const param_type& lhs, const param_type& rhs) { return !(lhs == rhs); }

        /* Friend declarations. */
        friend class discrete_alias_distribution;

    private:
        // Normalize probabilities.
        void normalize() {
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

        std::vector<double> probs_;
    };

    /**
     * @brief Default constructor uses a single weight of 1.0 such that the distribution will always
     * return 0.
     */
    discrete_alias_distribution() : param_() { initialize(); }

    /**
     * @brief Construct a distribution from a range of weights.
     *
     * @tparam InputIt Iterator type.
     * @param first Beginning of the range.
     * @param last End of the range.
     */
    template <typename InputIt>
    discrete_alias_distribution(InputIt first, InputIt last) : param_(first, last) {
        initialize();
    }

    /**
     * @brief Construct a distribution from a list of weights.
     *
     * @param list Initializer list of weights.
     */
    discrete_alias_distribution(std::initializer_list<double> list) : param_(list) { initialize(); }

    /**
     * @brief Construct a distribution such that the weights are given as
     * `w_i = unary_op(xmin + d/2 + i*d)` with `d = (xmax - xmin)/n` and `i = 0, ..., n`.
     *
     * @tparam UnaryOp Unary operation.
     * @param n Number of weights.
     * @param xmin Lower bound.
     * @param xmax Upper bound.
     * @param unary_op Unary operation.
     */
    template <typename UnaryOp>
    discrete_alias_distribution(std::size_t n, double xmin, double xmax, UnaryOp unary_op) :
        param_(n, xmin, xmax, unary_op) {
        initialize();
    }

    /**
     * @brief Construct a distribution from given distribution parameters.
     *
     * @param param Parameters specifying the weights/probabilities.
     */
    explicit discrete_alias_distribution(const param_type& param) : param_(param) { initialize(); }

    /**
     * @brief Lower bound of the distribution.
     *
     * @return Minimum value generated by the distribution (= 0).
     */
    [[nodiscard]] result_type min() const { return 0; }

    /**
     * @brief Upper bound of the distribution.
     *
     * @return Maximum value generated by the distribution.
     */
    [[nodiscard]] result_type max() const { return static_cast<result_type>(param_.probs_.size() - 1); }

    /**
     * @brief Get the probabilities of the distribution.
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
     * @brief Set the distribution parameters.
     *
     * @param param Distribution parameters.
     */
    void param(const param_type& param) {
        param_ = param;
        initialize();
    }

    /**
     * @brief Reset the internal state of distribution.
     */
    void reset() {}

    /**
     * @brief Generate a random integer.
     *
     * @tparam RNG Random number generator type.
     * @param rng RNG object.
     * @return Random integer distributed according to the weights/probabilities of the distribution.
     */
    template <typename RNG>
    result_type operator()(RNG& rng) const {
        auto j = std::uniform_int_distribution<std::size_t>(0, probs_.size() - 1)(rng);
        auto r2 = std::uniform_real_distribution<double>(0, 1)(rng);
        return (r2 <= probs_[j] ? j : alias_[j]);
    }

    /**
     * @brief Generate a random integer distributed according to given distribution parameters.
     *
     * @note It uses a linear search instead of the Walker-Alias method.
     *
     * @tparam RNG Random number generator type.
     * @param rng RNG object.
     * @param param Distribution parameters specifying weights/probabilities.
     * @return Random integer distributed according to the given distribution parameters.
     */
    template <typename RNG>
    result_type operator()(RNG& rng, const param_type& param) const {
        std::vector<double> accprobs;
        accprobs.reserve(param.probs_.size());
        std::partial_sum(param.probs_.begin(), param.probs_.end(), std::back_inserter(accprobs));
        accprobs.back() = 1.0;
        return detail::generate_discrete_ls<result_type>(rng, accprobs);
    }

    /**
     * @brief Compare two simplemc::discrete_alias_distribution objects for equality.
     *
     * @param lhs Left-hand side distribution.
     * @param rhs Right-hand side distribution.
     * @return True if the parameters of the distributions are the same.
     */
    [[nodiscard]] friend bool operator==(
        const discrete_alias_distribution& lhs, const discrete_alias_distribution& rhs) {
        return lhs.param_ == rhs.param_;
    }

    /**
     * @brief Compare two simplemc::discrete_alias_distribution objects for inequality.
     *
     * @param lhs Left-hand side distribution.
     * @param rhs Right-hand side distribution.
     * @return True if the parameters of the distributions are distinct.
     */
    [[nodiscard]] friend bool operator!=(
        const discrete_alias_distribution& lhs, const discrete_alias_distribution& rhs) {
        return !(lhs == rhs);
    }

    /**
     * @brief Write a textual representation of a simplemc::discrete_alias_distribution to
     * `std::ostream`.
     *
     * @details Throws an exception, if writing to `std::ostream` fails.
     *
     * @param os `std::ostream` to write to.
     * @param dad Distribution to be written.
     * @return Reference to the `std::ostream` object.
     */
    friend std::ostream& operator<<(std::ostream& os, const discrete_alias_distribution& dad) {
        auto prec = os.precision();
        auto check_os = [](const std::ostream& os) {
            if (!os) {
                throw simplemc_exception("Error writing a simplemc::discrete_alias_distribution to ostream");
            }
        };
        os.precision(std::numeric_limits<double>::max_digits10);
        check_os(os << dad.param().probabilities().size());
        for (const auto& p : dad.param().probabilities()) {
            check_os(os << ' ' << p);
        }
        os.precision(prec);
        return os;
    }

    /**
     * @brief Read a textual representation of a simplemc::discrete_alias_distribution from
     * `std::istream`.
     *
     * @details Throws an exception, if reading from `std::istream` fails.
     *
     * @tparam T Integral type.
     * @param is `std::istream` to read from.
     * @param dad Distribution to be read into.
     * @return Reference to the `std::istream` object.
     */
    friend std::istream& operator>>(std::istream& is, discrete_alias_distribution& dad) {
        auto check_is = [](const std::istream& is) {
            if (!is) {
                throw simplemc_exception("Error reading a simplemc::discrete_alias_distribution from istream");
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
        typename discrete_alias_distribution<T>::param_type parm { probs.begin(), probs.end() };
        dad.param(parm);
        return is;
    }

private:
    // Initialize the probablities and the alias table.
    void initialize() {
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
        for (int i : below) {
            probs_[i] = 1;
        }
        for (int i : above) {
            probs_[i] = 1;
        }
    }

    param_type param_;
    std::vector<double> probs_;
    std::vector<int> alias_;
};

} // namespace simplemc

#endif // SIMPLEMC_RANDOM_DISCRETE_ALIAS_DISTRIBUTION_HPP
