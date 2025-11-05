#include <fmt/base.h>
#include <simplemc/utils.hpp>

int main() {
    // simplemc::generic_error with function name
    try {
        throw simplemc::generic_error("generic_error", "Test message", "main");
    } catch (const simplemc::generic_error& e) {
        fmt::println("Caught exception: {}", e.what());
    }

    // simplemc::simplemc_exception with automatic source location capture
    try {
        throw simplemc::simplemc_exception("Test message");
    } catch (const simplemc::simplemc_exception& e) {
        fmt::println("Caught exception: {}", e.what());
    }
}
