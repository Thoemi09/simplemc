#include "./test_utils.hpp"

#include <fmt/ranges.h>

void histogram01::add(double value) {
    int idx = static_cast<int>(value / step_);
    hist_[idx] += 1.0;
    nsamples_ += 1;
}

// Print the histogram.
void histogram01::print() const {
    fmt::print("Histogram:\n{}\n", hist_);
}

// Check that the histogram is uniform within some tolerance.
[[nodiscard]] bool histogram01::check_uniform(double tol) const {
    double exact = static_cast<double>(nsamples_) / nbins_;
    double err = tol * exact;
    double lower = exact - err;
    double upper = exact + err;
    bool res = true;
    for (int i = 0; i < nbins_; i++) {
        if (hist_[i] < lower || hist_[i] > upper) {
            res = false;
            break;
        }
    }
    return res;
}
