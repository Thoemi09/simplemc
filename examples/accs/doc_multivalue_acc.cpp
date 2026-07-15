// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <fmt/ranges.h>
#include <simplemc/accs.hpp>
#include <simplemc/numeric/eigen.hpp>

#include <vector>

int main() {
    // data to be sampled
    std::vector<double> data = { 1.0, 2.0, 3.0, 4.0, 5.0 };

    // accumulate samples into the first and third elements of a mean accumulator of size 3
    simplemc::mean_acc_dynamic<double> acc(3);
    for (auto& val : data) {
        auto mva = acc.make_mva();
        mva[0] << val;
        mva[2] << val;
        mva.commit();
    }

    // print the mean of the accumulated data
    fmt::println("Mean: {}", simplemc::make_span(acc.mean()));
}
