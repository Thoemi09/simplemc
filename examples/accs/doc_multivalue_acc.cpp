#include <fmt/ranges.h>
#include <simplemc/accs.hpp>

#include <vector>

int main() {
    // data to be sampled
    std::vector<double> data = { 1.0, 2.0, 3.0, 4.0, 5.0 };

    // accumulate samples into the first and third element of a mean accumulator of size 3
    simplemc::mean_acc_dynamic<double> acc(3);
    for (auto& val : data) {
        auto mva = acc.make_mva();
        mva[0] << val;
        mva[2] << val;
        mva.increment_count();
    }

    // print the mean of the accumulated data
    fmt::print("Mean: {}\n", acc.mean());
}
