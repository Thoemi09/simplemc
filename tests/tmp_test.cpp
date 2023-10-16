#include "fmt/core.h"
#include <fmt/ranges.h>
#include <simplemc/grids.hpp>
#include <range/v3/all.hpp>

int main() {
    simplemc::power_grid pg { 1, 4, 3, 2 };
    fmt::print("grid = {}\n", pg.view());
}
