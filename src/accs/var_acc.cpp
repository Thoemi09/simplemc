/**
 * @file var_acc.cpp
 * @brief Mean accumulator for calculating arithmetic means.
 */

#include <simplemc/accs/var_acc.hpp>
#include <simplemc/numeric/utils.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

namespace simplemc {

template <accs::varalg A>
var_acc<double, A>::var_acc(size_type size, value_type shift) :
    data_(storage_type::Zero(size)),
    data2_(storage_type::Zero(size)),
    count_(0),
    idx_(0),
    shift_(shift) {
    if (size <= 0) {
        throw simplemc_exception("Size <= 0 in mean accumulator", "var_acc::var_acc");
    }
}

template <accs::varalg A>
var_acc<double, A>::var_acc(const storage_type& data, const storage_type& data2, count_type count, value_type shift) :
    data_(data),
    data2_(data2),
    count_(count),
    idx_(0),
    shift_(shift) {
    if (data_.size() == 0) {
        throw simplemc_exception("Size == 0 in mean accumulator", "var_acc::var_acc");
    }
    if (data_.size() != data2_.size()) {
        throw simplemc_exception("Sizes of data storages do not match", "var_acc::var_acc");
    }
}

template <accs::varalg A>
void var_acc<double, A>::set(const storage_type& data, const storage_type& data2, count_type count, value_type shift) {
    if (data.size() == 0) {
        throw simplemc_exception("Size == 0 in mean accumulator", "var_acc::set");
    }
    if (data.size() != data2.size()) {
        throw simplemc_exception("Sizes of data storages do not match", "var_acc::var_acc");
    }
    data_ = data;
    data2_ = data2;
    count_ = count;
    idx_ = 0;
    shift_ = shift;
}

template <accs::varalg A>
void var_acc<double, A>::reset(size_type size, value_type shift) {
    if (size <= 0) {
        size = data_.size();
    }
    if (!simplemc::isfinite(shift)) {
        shift = shift_;
    }
    data_ = storage_type::Zero(size);
    data2_ = storage_type::Zero(size);
    count_ = 0;
    idx_ = 0;
    shift_ = shift;
}

template <accs::varalg A>
var_acc<double, A>& var_acc<double, A>::operator<<(const var_acc& acc) {
    assert(size() == acc.size());
    assert(shift() == acc.shift());
    data_ += acc.data_;
    data2_ += acc.data2_;
    count_ += acc.count_;
    return *this;
}

/* Explicit template instantiations. */
template class var_acc<double>;
//template class var_acc<std::complex<double>>;
template class var_acc<double, accs::varalg::welford>;
//template class var_acc<std::complex<double>, accs::varalg::welford>;

} // namespace simplemc
