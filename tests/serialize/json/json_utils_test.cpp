#include "../../test_utils.hpp"

#include <simplemc/serialize/json/complex.hpp>
#include <simplemc/serialize/json/eigen.hpp>
#include <simplemc/serialize/json/file_io.hpp>
#include <simplemc/serialize/json/utils.hpp>
#include <simplemc/utils/ranges.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include <array>
#include <complex>
#include <string>
#include <vector>

// Round-trip a raw nlohmann::json under the given options.
static void check_file_io(const nlohmann::json& j, const std::string& fname, const simplemc::json_io_options& opts) {
    simplemc::write_json_file(j, fname, opts);
    nlohmann::json j2;
    ASSERT_NE(j, j2);
    simplemc::read_json_file(j2, fname, opts);
    ASSERT_EQ(j, j2);
}

// Test writing/reading a JSON file in various modes.
TEST(JSONUtils, FileIOModes) {
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

// Test root-level (non-object) document round-trip through file IO.
TEST(JSONUtils, FileIORootArray) {
    const nlohmann::json j = std::vector<double> { 1.0, 2.5, -3.0 };
    simplemc::write_json_file(j, "root_array.json");
    nlohmann::json j2;
    simplemc::read_json_file(j2, "root_array.json");
    ASSERT_EQ(j2, j);
}

// Test JSON serialization of ranges.
TEST(JSONUtils, RangeAdapters) {
    nlohmann::json j;
    std::array<int, 3> arr { 1, 2, 3 };
    std::array<int, 3> arr_rev {};

    simplemc::range_to_json(j["arr"], arr);
    simplemc::range_from_json(j["arr"], arr_rev | simplemc::ranges::views::reverse);
    check_equal(arr | simplemc::ranges::views::reverse, arr_rev);
}

// Test JSON serialization of complex numbers.
TEST(JSONUtils, ComplexAdlSerializer) {
    const std::complex<double> z { 1.5, -2.5 };
    nlohmann::json j = z;
    // shape: [real, imag]
    ASSERT_TRUE(j.is_array());
    ASSERT_EQ(j.size(), 2u);
    EXPECT_DOUBLE_EQ(j[0].get<double>(), 1.5);
    EXPECT_DOUBLE_EQ(j[1].get<double>(), -2.5);

    std::complex<double> z2;
    j.get_to(z2);
    check_equal(z2, z);
}

// Test JSON serialization of dynamic-size Eigen vectors.
TEST(JSONUtils, EigenVectorXdAdlSerializer) {
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    nlohmann::json j = v;

    Eigen::VectorXd v2;
    j.get_to(v2);
    ASSERT_EQ(v2.size(), v.size());
    check_near(v2, v);
}

// Test JSON serialization of dynamic-size Eigen matrices.
TEST(JSONUtils, EigenMatrixXdAdlSerializer) {
    Eigen::MatrixXd m(2, 3);
    m << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    nlohmann::json j = m;

    Eigen::MatrixXd m2;
    j.get_to(m2);
    ASSERT_EQ(m2.rows(), m.rows());
    ASSERT_EQ(m2.cols(), m.cols());
    check_near(m2, m);
}

// Test JSON serialization of fixed-size Eigen vectors.
TEST(JSONUtils, EigenFixedVector3dAdlSerializer) {
    const Eigen::Vector3d v { 1.0, 2.0, 3.0 };
    nlohmann::json j = v;

    Eigen::Vector3d v2;
    j.get_to(v2);
    check_near(v2, v);
}

// Test JSON serialization of fixed-size Eigen matrices.
TEST(JSONUtils, EigenFixedMatrix23AdlSerializer) {
    Eigen::Matrix<double, 2, 3> m;
    m << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    nlohmann::json j = m;

    Eigen::Matrix<double, 2, 3> m2;
    j.get_to(m2);
    check_near(m2, m);
}

// Test that deserializing into a fixed-size Eigen type with mismatched dimensions throws.
TEST(JSONUtils, EigenFixedMismatchThrows) {
    // serialize a 3-vector, attempt to load into a 2-vector
    const Eigen::Vector3d v3 { 1.0, 2.0, 3.0 };
    const nlohmann::json j_vec = v3;
    Eigen::Vector2d v2;
    EXPECT_THROW(j_vec.get_to(v2), simplemc::simplemc_exception);

    // serialize a 2x3 matrix, attempt to load into a 3x2 matrix
    Eigen::Matrix<double, 2, 3> m23;
    m23 << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    const nlohmann::json j_mat = m23;
    Eigen::Matrix<double, 3, 2> m32;
    EXPECT_THROW(j_mat.get_to(m32), simplemc::simplemc_exception);
}
