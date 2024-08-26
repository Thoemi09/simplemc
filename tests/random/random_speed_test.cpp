#include <fmt/ranges.h>
#include <simplemc/random.hpp>
#include <simplemc/utils.hpp>

int main() {
    using namespace simplemc;
    std::vector<double> weights { 1, 3, 5, 0, 2, 6, 18, 2, 11, 8 };
    // std::vector<double> weights { 1, 6, 3 };
    std::vector<int> hist(weights.size(), 0);
    xoshiro256p xop;
    int n = 1000000;
    discrete_distribution<int> smc_dist(weights.begin(), weights.end());
    timer t;
    t.start();
    for (int i = 0; i < n; ++i) {
        hist[smc_dist(xop)] += 1;
    }
    t.stop();
    fmt::print("Runtime simplemc::discrete_distribution = {}\n", time_passed(t.start_time(), t.stop_time()));
    fmt::print("{}\n", hist);
    std::vector<int> hist2(weights.size(), 0);
    std::discrete_distribution<int> std_dist(weights.begin(), weights.end());
    t.start();
    for (int i = 0; i < n; ++i) {
        hist2[std_dist(xop)] += 1;
    }
    t.stop();
    fmt::print("Runtime std::discrete_distribution = {}\n", time_passed(t.start_time(), t.stop_time()));
    fmt::print("{}\n", hist2);
    std::vector<int> hist3(weights.size(), 0);
    discrete_alias_distribution<int> smc_al_dist(weights.begin(), weights.end());
    t.start();
    for (int i = 0; i < n; ++i) {
        hist3[smc_al_dist(xop)] += 1;
    }
    t.stop();
    fmt::print("Runtime simplemc::discrete_alias_distribution = {}\n", time_passed(t.start_time(), t.stop_time()));
    fmt::print("{}\n", hist3);
}
