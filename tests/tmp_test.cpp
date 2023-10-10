#include "range/v3/algorithm/transform.hpp"
#include <simplemc/json/range.hpp>

#include <fmt/ranges.h>
#include <range/v3/range.hpp>
#include <range/v3/algorithm.hpp>
#include <range/v3/view.hpp>

#include <vector>

int main() {
    auto view = ranges::views::ints(0, 10);
    fmt::print("{}\n", view);
}