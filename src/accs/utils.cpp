/**
 * @file utils.cpp
 * @brief Utility functions for simplemc-accs.
 */

#include <simplemc/accs/utils.hpp>

namespace simplemc::accs {

template <double_or_complex T>
Eigen::ArrayX<T> make_nans(long size) {
    constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
    if constexpr (std::is_same_v<T, double>) {
        return Eigen::ArrayX<T>::Constant(size, nan);
    } else {
        return Eigen::ArrayX<T>::Constant(size, std::complex<double> { nan, nan });
    }
}

template <double_or_complex T, varalg A>
Eigen::ArrayX<T> mean(const Eigen::ArrayX<T>& data, std::uint64_t count, T shift) {
    if (count == 0) {
        return make_nans<T>(data.size());
    }
    if constexpr (A == varalg::standard) {
        return data / static_cast<T>(count) + shift;
    } else {
        return data + shift;
    }
}

/* Explicit template instantiations. */
template Eigen::ArrayX<double> make_nans<double>(long);
template Eigen::ArrayX<std::complex<double>> make_nans(long);

template Eigen::ArrayX<double> mean<double>(const Eigen::ArrayX<double>&, std::uint64_t, double);
template Eigen::ArrayX<std::complex<double>> mean(
    const Eigen::ArrayX<std::complex<double>>&, std::uint64_t, std::complex<double>);
template Eigen::ArrayX<double> mean<double, varalg::welford>(const Eigen::ArrayX<double>&, std::uint64_t, double);
template Eigen::ArrayX<std::complex<double>> mean<std::complex<double>, varalg::welford>(
    const Eigen::ArrayX<std::complex<double>>&, std::uint64_t, std::complex<double>);

} // namespace simplemc::accs
