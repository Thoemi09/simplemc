/**
 * @file
 * @brief Unit tests for simplemc-json.
 */

#include "../test_utils.hpp"

#include <simplemc/json.hpp>

#include <span>

namespace foo {

// Custom class for testing JSON serialization.
struct bar : public simplemc::basic_json {
    int x = 10;
    void write_json(nlohmann::json& j) const override { j["x"] = x; }
    void read_json(const nlohmann::json& j) override { j.at("x").get_to(x); }
};

} // namespace foo

// Check JSON serialization/deserialization of an object.
template <typename T1, typename T2>
void check_json(const T1& orig, T2& copy) {
    nlohmann::json j;
    j["orig"] = orig;
    j.at("orig").get_to(copy);
    j["copy"] = copy;
    ASSERT_EQ(j["orig"], j["copy"]);
}

// Check JSON serialization/deserialization of an object (needs to be default constructible).
template <typename T>
void check_json(const T& orig) {
    T copy;
    check_json(orig, copy);
}

// Check if file IO in text mode works.
void check_file_io(const nlohmann::json j, const std::string& fname, int w) {
    simplemc::write_json_file(j, fname, w);
    nlohmann::json j2;
    ASSERT_NE(j, j2);
    simplemc::read_json_file(j2, fname);
    ASSERT_EQ(j, j2);
}

// Check if file IO in binary mode works.
void check_file_io(const nlohmann::json j, const std::string& fname, simplemc::json_binary_mode m) {
    simplemc::write_json_file(j, fname, m);
    nlohmann::json j2;
    ASSERT_NE(j, j2);
    simplemc::read_json_file(j2, fname, m);
    ASSERT_EQ(j, j2);
}

// Test JSON serialization of complex types.
TEST(SimplemcJson, ComplexSerialization) {
    std::complex<double> cdb { 1.0, 2.0 };
    check_json(cdb);
}

// Test JSON serialization of ranges.
TEST(SimplemcJson, RangeSerialization) {
    nlohmann::json j;
    std::array<int, 3> arr { 1, 2, 3 }, arr_rev {};
    auto arr_span = std::span(arr);
    simplemc::range_to_json(j["arr"], arr_span);
    simplemc::range_from_json(j["arr"], arr_rev | ranges::views::reverse);
    check_range_equal(arr | ranges::views::reverse, arr_rev);
}

// Test JSON file IO.
TEST(SimplemcJson, FileIO) {
    nlohmann::json j;
    j["int"] = 1;
    j["short"] = 3.14;
    j["string"] = "my string";
    j["complex"] = std::complex<double> { 1.0, 2.0 };
    j["array"] = std::array<int, 3> { 1, 2, 3 };
    check_file_io(j, "text_0.json", 0);
    check_file_io(j, "text_4.json", 4);
    check_file_io(j, "binary.bson", simplemc::json_binary_mode::bson);
    check_file_io(j, "binary.cbor", simplemc::json_binary_mode::cbor);
    check_file_io(j, "binary.msgpack", simplemc::json_binary_mode::msgpack);
    check_file_io(j, "binary.ubjson", simplemc::json_binary_mode::ubjson);
}

// Test deriving from basic_json.
TEST(SimplemcJson, BasicJson) {
    foo::bar b;
    b.x = 100;
    check_json(b);
}
