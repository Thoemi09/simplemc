/**
 * @file mean_acc.cpp
 * @brief Mean accumulator for calculating arithmetic means.
 */

#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

namespace simplemc {

template <double_or_complex T, accs::varalg A>
mean_acc<T, A>::mean_acc(size_type size, value_type shift) :
    data_(storage_type::Zero(size)),
    count_(0),
    idx_(0),
    shift_(shift) {
    if (size <= 0) {
        throw simplemc_exception("Size <= 0 in mean accumulator", "mean_acc::mean_acc");
    }
}

template <double_or_complex T, accs::varalg A>
mean_acc<T, A>::mean_acc(const storage_type& data, count_type count, value_type shift) :
    data_(data),
    count_(count),
    idx_(0),
    shift_(shift) {
    if (data_.size() == 0) {
        throw simplemc_exception("Size == 0 in mean accumulator", "mean_acc::mean_acc");
    }
}

template <double_or_complex T, accs::varalg A>
void mean_acc<T, A>::set(const storage_type& data, count_type count, value_type shift) {
    if (data.size() == 0) {
        throw simplemc_exception("Size == 0 in mean accumulator", "mean_acc::set");
    }
    data_ = data;
    count_ = count;
    idx_ = 0;
    shift_ = shift;
}

template <double_or_complex T, accs::varalg A>
void mean_acc<T, A>::reset(size_type size, value_type shift) {
    if (size <= 0) {
        size = data_.size();
    }
    if (!simplemc::isfinite(shift)) {
        shift = shift_;
    }
    data_ = storage_type::Zero(size);
    count_ = 0;
    idx_ = 0;
    shift_ = shift;
}

template <double_or_complex T, accs::varalg A>
mean_acc<T, A>& mean_acc<T, A>::operator<<(const mean_acc& acc) {
    assert(size() == acc.size());
    data_ += (acc.data_ + acc.shift_ - shift_);
    count_ += acc.count_;
    return *this;
}

/* Explicit template instantiations. */
template class mean_acc<double>;
template class mean_acc<std::complex<double>>;
template class mean_acc<double, accs::varalg::welford>;
template class mean_acc<std::complex<double>, accs::varalg::welford>;

} // namespace simplemc
