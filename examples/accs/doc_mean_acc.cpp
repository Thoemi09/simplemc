#include <fmt/base.h>
#include <simplemc/accs/mean_acc.hpp>

#include <vector>

int main() {
    // data to be sampled
    std::vector<double> data = { 1.0, 2.0, 3.0, 4.0, 5.0 };

    // accumulate samples into a (scalar) mean accumulator
    simplemc::mean_acc<double> acc {};
    for (const auto& val : data) {
        acc << val;
    }

    // print the mean of the accumulated data
    fmt::println("Mean: {}", acc.mean());
}
