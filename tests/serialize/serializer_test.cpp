/**
 * @file
 * @brief Unit tests for the simplemc-serialize core + JSON backend.
 *
 * @details Covers:
 *  - `serializer` / `deserializer` concept conformance (json_serializer, json_deserializer,
 *    and a mock backend that's not JSON-shaped).
 *  - Primitive and `std::complex` / `std::vector` / `Eigen::Vector*` / `Eigen::Matrix*` round-trips
 *    through the nlohmann-fallback path.
 *  - ADL dispatch via friend `simplemc_save` (intrusive) and namespace-scope `simplemc_save` /
 *    `simplemc_load` (non-intrusive).
 *  - `operator[]` navigation produces the expected JSON tree.
 *  - `has` / `array_size` schema-tolerance queries.
 *  - All four binary file IO modes still round-trip via the relocated file_io.
 */

#include "../test_utils.hpp"

#include <simplemc/serialize.hpp>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <complex>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// ===== Test types =====================================================================

namespace test_types {

// Type with an intrusive friend simplemc_save / simplemc_load (private member access).
struct intrusive_point {
private:
    double x_ = 0;
    double y_ = 0;

public:
    intrusive_point() = default;
    intrusive_point(double x, double y) : x_ { x }, y_ { y } {}

    [[nodiscard]] double x() const { return x_; }
    [[nodiscard]] double y() const { return y_; }

    bool operator==(const intrusive_point&) const = default;

    template <simplemc::serializer S>
    friend void simplemc_save(S& s, const intrusive_point& p) {
        s.save_at("x", p.x_);
        s.save_at("y", p.y_);
    }
    template <simplemc::deserializer S>
    friend void simplemc_load(const S& s, intrusive_point& p) {
        s.load_at("x", p.x_);
        s.load_at("y", p.y_);
    }
};

// Type with public fields and namespace-scope simplemc_save / simplemc_load (non-intrusive).
struct nonintrusive_box {
    int width = 0;
    int height = 0;

    bool operator==(const nonintrusive_box&) const = default;
};

template <simplemc::serializer S>
void simplemc_save(S& s, const nonintrusive_box& b) {
    s.save_at("width", b.width);
    s.save_at("height", b.height);
}
template <simplemc::deserializer S>
void simplemc_load(const S& s, nonintrusive_box& b) {
    s.load_at("width", b.width);
    s.load_at("height", b.height);
}

// Composite type that uses both customization mechanisms recursively.
struct composite {
    intrusive_point pt;
    nonintrusive_box bx;
    std::vector<double> data;

    bool operator==(const composite&) const = default;
};

template <simplemc::serializer S>
void simplemc_save(S& s, const composite& c) {
    s.save_at("pt", c.pt);
    s.save_at("bx", c.bx);
    s.save_at("data", c.data);
}
template <simplemc::deserializer S>
void simplemc_load(const S& s, composite& c) {
    s.load_at("pt", c.pt);
    s.load_at("bx", c.bx);
    s.load_at("data", c.data);
}

// Mock output serializer — proves the serializer concept doesn't bake in JSON details.
class mock_output {
public:
    using key_type = std::string;
    using value_type = double; // simplified: only doubles for the test

    std::map<key_type, value_type>* current_;
    std::map<key_type, value_type> own_;

    mock_output() : current_ { &own_ } {}

    mock_output save_at(std::string_view key, double v) {
        (*current_)[std::string { key }] = v;
        return *this;
    }
    // For composite types, mock_output supports recursive ADL dispatch through key-prefixing.
    // For this test we only care about concept satisfaction.

    mock_output operator[](std::string_view /*key*/) { return *this; }

    [[nodiscard]] bool has(std::string_view key) const { return current_->contains(std::string { key }); }
};
static_assert(simplemc::serializer<mock_output, double>);

} // namespace test_types

// ===== Concept conformance ============================================================

TEST(SerializerCore, ConceptConformance) {
    using simplemc::deserializer;
    using simplemc::serializer;
    using simplemc::json_deserializer;
    using simplemc::json_serializer;

    static_assert(serializer<json_serializer>);
    static_assert(deserializer<json_deserializer>);
    SUCCEED();
}

// ===== Primitive + std + Eigen round-trips through nlohmann fallback ==================

TEST(SerializerJson, PrimitiveRoundtrip) {
    simplemc::json_serializer s {};
    s.save_at("int", 42);
    s.save_at("double", 3.14);
    s.save_at("bool", true);
    s.save_at("string", std::string { "hi" });

    s.write_to_file("primitive.json");

    simplemc::json_deserializer d { "primitive.json" };
    int i = 0;
    double dbl = 0;
    bool b = false;
    std::string str;
    d.load_at("int", i);
    d.load_at("double", dbl);
    d.load_at("bool", b);
    d.load_at("string", str);

    ASSERT_EQ(i, 42);
    ASSERT_DOUBLE_EQ(dbl, 3.14);
    ASSERT_EQ(b, true);
    ASSERT_EQ(str, "hi");
}

TEST(SerializerJson, ComplexRoundtrip) {
    const std::complex<double> z { 1.5, -2.5 };
    simplemc::json_serializer s {};
    s.save_at("z", z);
    s.write_to_file("complex.json");

    simplemc::json_deserializer d { "complex.json" };
    std::complex<double> back;
    d.load_at("z", back);
    check_equal(back, z);
}

TEST(SerializerJson, VectorRoundtrip) {
    const std::vector<double> v { 1.0, 2.0, 3.0, 4.0 };
    simplemc::json_serializer s {};
    s.save_at("v", v);
    s.write_to_file("vector.json");

    simplemc::json_deserializer d { "vector.json" };
    std::vector<double> back;
    d.load_at("v", back);
    check_equal(back, v);
}

TEST(SerializerJson, EigenVectorRoundtrip) {
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    simplemc::json_serializer s {};
    s.save_at("v", v);
    s.write_to_file("eigen_vec.json");

    simplemc::json_deserializer d { "eigen_vec.json" };
    Eigen::VectorXd back;
    d.load_at("v", back);
    ASSERT_EQ(back.size(), v.size());
    for (long i = 0; i < v.size(); ++i) {
        ASSERT_DOUBLE_EQ(back(i), v(i));
    }
}

TEST(SerializerJson, EigenMatrixRoundtrip) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    simplemc::json_serializer s {};
    s.save_at("m", m);
    s.write_to_file("eigen_mat.json");

    simplemc::json_deserializer d { "eigen_mat.json" };
    Eigen::MatrixXd back;
    d.load_at("m", back);
    ASSERT_EQ(back.rows(), m.rows());
    ASSERT_EQ(back.cols(), m.cols());
    for (long r = 0; r < m.rows(); ++r) {
        for (long c = 0; c < m.cols(); ++c) {
            ASSERT_DOUBLE_EQ(back(r, c), m(r, c));
        }
    }
}

// ===== ADL dispatch ===================================================================

TEST(SerializerJson, AdlDispatch_FriendPath) {
    using test_types::intrusive_point;
    const intrusive_point p { 1.25, -3.5 };

    simplemc::json_serializer s {};
    s.save_at("p", p);

    // verify the JSON shape: { "p": { "x": 1.25, "y": -3.5 } }
    const auto& j = s.tree();
    ASSERT_TRUE(j.contains("p"));
    ASSERT_TRUE(j["p"].contains("x"));
    ASSERT_TRUE(j["p"].contains("y"));
    ASSERT_DOUBLE_EQ(j["p"]["x"].get<double>(), 1.25);
    ASSERT_DOUBLE_EQ(j["p"]["y"].get<double>(), -3.5);

    s.write_to_file("intrusive.json");
    simplemc::json_deserializer d { "intrusive.json" };
    intrusive_point back;
    d.load_at("p", back);
    ASSERT_EQ(back, p);
}

TEST(SerializerJson, AdlDispatch_FreeFunctionPath) {
    using test_types::nonintrusive_box;
    const nonintrusive_box b { 4, 5 };

    simplemc::json_serializer s {};
    s.save_at("b", b);
    s.write_to_file("nonintrusive.json");

    simplemc::json_deserializer d { "nonintrusive.json" };
    nonintrusive_box back;
    d.load_at("b", back);
    ASSERT_EQ(back, b);
}

TEST(SerializerJson, AdlDispatch_Composite) {
    using namespace test_types;
    const composite c { intrusive_point { 1.0, 2.0 }, nonintrusive_box { 10, 20 }, { 0.5, 1.5, 2.5 } };

    simplemc::json_serializer s {};
    s.save_at("c", c);
    s.write_to_file("composite.json");

    simplemc::json_deserializer d { "composite.json" };
    composite back;
    d.load_at("c", back);
    ASSERT_EQ(back, c);
}

// ===== operator[] navigation ==========================================================

TEST(SerializerJson, OperatorBracket_BuildsSubTree) {
    simplemc::json_serializer s {};
    s.save_at("step", 1);
    s["results"].save_at("mean", 0.5);
    s["results"].save_at("var", 0.1);

    const auto& j = s.tree();
    ASSERT_EQ(j["step"].get<int>(), 1);
    ASSERT_DOUBLE_EQ(j["results"]["mean"].get<double>(), 0.5);
    ASSERT_DOUBLE_EQ(j["results"]["var"].get<double>(), 0.1);
}

TEST(SerializerJson, OperatorBracket_ChainedNavigation) {
    simplemc::json_serializer s {};
    s["a"]["b"]["c"].save_at("x", 42);

    const auto& j = s.tree();
    ASSERT_EQ(j["a"]["b"]["c"]["x"].get<int>(), 42);
}

TEST(SerializerJson, OperatorBracket_SharesTree) {
    simplemc::json_serializer s {};
    auto results = s["results"];
    results.save_at("mean", 0.5);
    results.save_at("var", 0.1);

    // s sees the changes because the tree is shared
    const auto& j = s.tree();
    ASSERT_DOUBLE_EQ(j["results"]["mean"].get<double>(), 0.5);
    ASSERT_DOUBLE_EQ(j["results"]["var"].get<double>(), 0.1);
}

TEST(SerializerJson, OperatorBracket_ReadSideThrowsOnMissing) {
    nlohmann::json j;
    j["existing"] = 1;
    auto d = simplemc::json_deserializer::from_tree(j);
    EXPECT_THROW({ auto sub = d["missing"]; }, simplemc::simplemc_exception);
}

// ===== Schema-tolerance queries =======================================================

TEST(SerializerJson, SchemaTolerance_HasAndArraySize) {
    nlohmann::json j;
    j["present"] = 1;
    j["list"] = { 10, 20, 30 };
    auto d = simplemc::json_deserializer::from_tree(j);

    EXPECT_TRUE(d.has("present"));
    EXPECT_FALSE(d.has("missing"));
    EXPECT_EQ(d.array_size("list"), 3u);
}

TEST(SerializerJson, SchemaTolerance_ArrayShape) {
    nlohmann::json j;
    j["mat"] = nlohmann::json::array({ nlohmann::json::array({ 1, 2, 3 }), nlohmann::json::array({ 4, 5, 6 }) });
    auto d = simplemc::json_deserializer::from_tree(j);

    auto shape = d.array_shape("mat");
    ASSERT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], 2u);
    EXPECT_EQ(shape[1], 3u);
}

// ===== File IO binary modes (relocated coverage from old json/file_io tests) =========

// Check if file IO round-trips a JSON object under the given options.
static void check_file_io(const nlohmann::json& j, const std::string& fname,
                          const simplemc::json_io_options& opts) {
    simplemc::write_json_file(j, fname, opts);
    nlohmann::json j2;
    ASSERT_NE(j, j2);
    simplemc::read_json_file(j2, fname, opts);
    ASSERT_EQ(j, j2);
}

TEST(SerializerJson, FileIOModes) {
    using simplemc::json_file_mode;
    nlohmann::json j;
    j["int"] = 1;
    j["double"] = 3.14;
    j["string"] = "my string";
    j["complex"] = std::complex<double> { 1.0, 2.0 };
    j["array"] = std::vector<int> { 1, 2, 3 };
    check_file_io(j, "text_0.json", { .mode = json_file_mode::text, .indent = 0 });
    check_file_io(j, "text_4.json", { .mode = json_file_mode::text, .indent = 4 });
    check_file_io(j, "binary.bson", { .mode = json_file_mode::bson });
    check_file_io(j, "binary.cbor", { .mode = json_file_mode::cbor });
    check_file_io(j, "binary.msgpack", { .mode = json_file_mode::msgpack });
    check_file_io(j, "binary.ubjson", { .mode = json_file_mode::ubjson });
}

// ===== Range helpers (relocated from json/range.hpp tests) =====================

TEST(SerializerJson, RangeAdapters) {
    nlohmann::json j;
    std::array<int, 3> arr { 1, 2, 3 };
    std::array<int, 3> arr_rev {};

    simplemc::range_to_json(j["arr"], arr);
    simplemc::range_from_json(j["arr"], arr_rev | simplemc::ranges::views::reverse);
    check_equal(arr | simplemc::ranges::views::reverse, arr_rev);
}

// ===== save_to_file / load_from_file one-shots =========================================

TEST(SerializerJson, SaveLoadToFile_WithSimplemcSave) {
    using test_types::intrusive_point;
    const intrusive_point p { 7.0, 9.0 };
    simplemc::json_serializer::save_to_file("p.json", p);

    intrusive_point back;
    simplemc::json_deserializer::load_from_file("p.json", back);
    ASSERT_EQ(back, p);
}

TEST(SerializerJson, SaveLoadToFile_NlohmannFallback) {
    const std::vector<double> v { 1.0, 2.5, -3.0 };
    simplemc::json_serializer::save_to_file("v.json", v);

    std::vector<double> back;
    simplemc::json_deserializer::load_from_file("v.json", back);
    check_equal(back, v);
}
