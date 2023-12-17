#include <fmt/ranges.h>
#include <vector>

int main() {
    std::vector<int> v{1, 2, 3};
    fmt::print("Range: {}\n", v);
    return 0;
}
