/**
 * @file traits.hpp
 * @brief Traits, type defs and other things related to accumulators.
 */

#ifndef SIMPLEMC_ACCS_TRAITS_HPP
#define SIMPLEMC_ACCS_TRAITS_HPP

#include <simplemc/numeric/eigen.hpp>
#include <simplemc/utils/concepts.hpp>

#include <cstdint>

namespace simplemc {

/**
 * @brief Traits for accumulators.
 *
 * @tparam T Type of accumulated values (either double or std::complex<double>).
 */
template <double_or_complex T>
struct acc_traits {
    /**
     * @brief Type of accumulated value.
     */
    using value_type = T;

    /**
     * @brief Type for counting.
     */
    using count_type = std::uint64_t;

    /**
     * @brief Type for storing accumulated values.
     */
    using storage_type = Eigen::ArrayX<value_type>;

    /**
     * @brief Construct a storage with NaNs.
     *
     * @param size Size of storage.
     * @return Storage with NaNs.
     */
    static storage_type make_nans(long size) {
        const auto nan = std::numeric_limits<double>::quiet_NaN();
        if constexpr (std::is_same_v<value_type, double>) {
            return storage_type::Constant(size, nan);
        } else {
            return storage_type::Constant(size, std::complex<double> { nan, nan });
        }
    }

    /**
     * @brief Calculate the sample mean.
     *
     * @param data Accumulated values.
     * @param count Number of accumulated values.
     * @return Sample mean.
     */
    static storage_type mean(const storage_type& data, count_type count) {
        if (count == 0) {
            return make_nans(data.size());
        }
        return data / static_cast<value_type>(count);
    }

    // /**
    //  * @brief Calculate the sample variance.
    //  *
    //  * @param data Accumulated values.
    //  * @param data2 Accumulated squared values.
    //  * @param count Number of accumulated values.
    //  * @return Sample variance.
    //  */
    // static storage_type variance(const storage_type& data, const storage_type& data2, count_type count) {
    //     if (count <= 1) {
    //         return make_nans(data.size());
    //     }
    //     auto m = data / static_cast<value_type>(count);
    //     auto re = data.real();
    //     auto im = data.imag();
    //     return (data2 - static_cast<value_type>(count) * m * m) / static_cast<value_type>(count - 1);
    // }
};

} // namespace simplemc

#endif // SIMPLEMC_ACCS_TRAITS_HPP
