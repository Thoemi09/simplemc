#include <fmt/ranges.h>
#include <vector>
#include <span>
#include <ranges>

int main() {
    std::vector<int> v { 1, 2, 3 };
    fmt::print("{}\n", std::span(v));
    fmt::print("{}\n", std::ranges::iota_view{1, 10});
}
