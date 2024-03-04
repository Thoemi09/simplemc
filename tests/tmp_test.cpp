#include "./test_utils.hpp"
#include "./accs/stochastic_process.hpp"

#include <simplemc/accs/utils.hpp>
#include <simplemc/utils/format.hpp>
#include <simplemc/utils/to_string.hpp>
#include <fmt/ranges.h>
#include <range/v3/all.hpp>

#include <algorithm>
#include <vector>

int main() {
    std::vector<int> v{ 1, 2, 1, 4, 5 };
    std::cout << std::ranges::is_sorted(v) << std::endl;
}