#include <fmt/base.h>
#include <simplemc/utils.hpp>

#include <array>

int main() {
    // open file and write to it --> it is automatically closed when going out of scope
    {
        auto file = simplemc::open_file("doc_file_io.txt", "w");
        fmt::println(file, "Line no. {}", 1);
        fmt::println(file, "Line no. {}", 2);
    }

    // open file and read from it --> it is automatically closed when going out of scope
    {
        auto file = simplemc::open_file("doc_file_io.txt", "r");
        fmt::println("File contents:");
        std::array<char, 256> buffer {};
        while (std::fgets(buffer.data(), sizeof(buffer), file)) {
            fmt::print("{}", buffer.data());
        }
    }
}
