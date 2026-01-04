#include "../test_utils.hpp"

#include <simplemc/numeric/eigen.hpp>
#include <simplemc/numeric/linear_map.hpp>
#include <simplemc/numeric/utils.hpp>

#include <complex>
#include <numbers>

// Test some useful concepts.
TEST(SimplemcNumeric, UtilityConcepts) {
    static_assert(simplemc::static_extent<Eigen::Dynamic>);
    static_assert(simplemc::static_extent<1>);
    static_assert(simplemc::static_extent<200>);
    static_assert(!simplemc::static_extent<0>);
    static_assert(!simplemc::static_extent<-10>);

    static_assert(simplemc::eigen_matrix<Eigen::Matrix<double, 1, 1>>);
    static_assert(simplemc::eigen_matrix<Eigen::Matrix<std::complex<double>, 1, 1>>);
    static_assert(simplemc::eigen_vector<Eigen::VectorXd> && simplemc::eigen_matrix<Eigen::VectorXd>);
    static_assert(simplemc::eigen_matrix<Eigen::VectorXcd> && simplemc::eigen_vector<Eigen::VectorXcd>);
    static_assert(simplemc::eigen_matrix<Eigen::MatrixXd>);
    static_assert(!simplemc::eigen_matrix<Eigen::MatrixXi>);
    static_assert(!simplemc::eigen_vector<Eigen::ArrayXd>);

    static_assert(simplemc::eigen_matrix_dbl<Eigen::MatrixXd>);
    static_assert(simplemc::eigen_matrix_cplx<Eigen::MatrixXcd>);
    static_assert(!simplemc::eigen_matrix_dbl<Eigen::MatrixXcd>);
    static_assert(!simplemc::eigen_matrix_cplx<Eigen::MatrixXd>);
    static_assert(simplemc::eigen_vector_dbl<Eigen::VectorXd>);
    static_assert(simplemc::eigen_vector_cplx<Eigen::VectorXcd>);
    static_assert(!simplemc::eigen_vector_dbl<Eigen::VectorXcd>);
    static_assert(!simplemc::eigen_vector_cplx<Eigen::VectorXd>);
}

// Test the make_nans functions.
TEST(SimplemcNumeric, MakeNans) {
    using namespace simplemc;
    auto vec_d = nans_vector<Eigen::VectorXd>(10);
    auto vec_sd = nans_vector<Eigen::Vector<double, 10>>();
    auto vec_c = nans_vector<Eigen::VectorXcd>(10);
    auto vec_sc = nans_vector<Eigen::Vector<std::complex<double>, 10>>();
    for (int i = 0; i < 10; ++i) {
        check_isnan(vec_d[i]);
        check_isnan(vec_sd[i]);
        check_isnan(vec_c[i]);
        check_isnan(vec_sc[i]);
    }

    auto mat_d = nans_matrix<Eigen::MatrixXd>(5, 5);
    auto mat_sd = nans_matrix<Eigen::Matrix<double, 5, 5>>();
    auto mat_c = nans_matrix<Eigen::MatrixXcd>(5, 5);
    auto mat_sc = nans_matrix<Eigen::Matrix<std::complex<double>, 5, 5>>();
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            check_isnan(mat_d(i, j));
            check_isnan(mat_sd(i, j));
            check_isnan(mat_c(i, j));
            check_isnan(mat_sc(i, j));
        }
    }
}

// Test some utilty functions.
TEST(SimplemcNumeric, UtilityFunctions) {
    // abs_diff and rel_diff for arithmetic types
    ASSERT_DOUBLE_EQ(simplemc::abs_diff(1e-5, 1e-5), 0);
    ASSERT_DOUBLE_EQ(simplemc::rel_diff(1e-5, 1e-5), 0);
    ASSERT_DOUBLE_EQ(simplemc::abs_diff(1e-5, 2e-5), 1e-5);
    ASSERT_DOUBLE_EQ(simplemc::rel_diff(1e-5, 2e-5), 1);

    // abs_diff and rel_diff for complex types
    std::complex<double> z1 { 1.0, 2.0 };
    std::complex<double> z2 { 1.0, 2.0 };
    ASSERT_DOUBLE_EQ(simplemc::abs_diff(z1, z2), 0);
    ASSERT_DOUBLE_EQ(simplemc::rel_diff(z1, z2), 0);
    std::complex<double> z3 { 2.0, 2.0 };
    ASSERT_DOUBLE_EQ(simplemc::abs_diff(z1, z3), 1.0);
    ASSERT_TRUE(simplemc::rel_diff(z1, z3) > 0);

    // rel_diff with zero values (edge case)
    ASSERT_TRUE(simplemc::rel_diff(0.0, 1e-10) > 0);
    ASSERT_TRUE(simplemc::rel_diff(1e-10, 0.0) > 0);
    std::complex<double> z_zero { 0.0, 0.0 };
    std::complex<double> z_small { 1e-10, 0.0 };
    ASSERT_TRUE(simplemc::rel_diff(z_zero, z_small) > 0);
    ASSERT_TRUE(simplemc::rel_diff(z_small, z_zero) > 0);

    // isfinite for arithmetic types
    ASSERT_TRUE(simplemc::isfinite(1.0 / 1.0));
    ASSERT_FALSE(simplemc::isfinite(1.0 / 0.0));
    ASSERT_TRUE(simplemc::isfinite(42));
    ASSERT_TRUE(simplemc::isfinite(-3.14f));

    // isfinite for complex types
    constexpr auto inf = std::numeric_limits<double>::infinity();
    constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
    std::complex<double> c_finite { 1.0, 2.0 };
    std::complex<double> c_nan_imag { 1.0, nan };
    std::complex<double> c_nan_real { nan, 2.0 };
    std::complex<double> c_inf_imag { 1.0, inf };
    std::complex<double> c_inf_real { inf, 2.0 };
    ASSERT_TRUE(simplemc::isfinite(c_finite));
    ASSERT_FALSE(simplemc::isfinite(c_nan_imag));
    ASSERT_FALSE(simplemc::isfinite(c_nan_real));
    ASSERT_FALSE(simplemc::isfinite(c_inf_imag));
    ASSERT_FALSE(simplemc::isfinite(c_inf_real));
}

// Test map_to_interval and map_to_interval_lb functions.
TEST(SimplemcNumeric, MapToInterval) {
    using std::numbers::pi;
    double tol = 1e-10;
    double lb = -pi;
    double ub = pi;
    ASSERT_NEAR(0.0, simplemc::map_to_interval(0.0, lb, ub), tol);
    ASSERT_NEAR(ub, simplemc::map_to_interval(ub, lb, ub), tol);
    ASSERT_NEAR(ub, simplemc::map_to_interval(lb, lb, ub), tol);
    ASSERT_NEAR(ub, simplemc::map_to_interval(3 * pi, lb, ub), tol);
    ASSERT_NEAR(ub, simplemc::map_to_interval(-3 * pi, lb, ub), tol);
    ASSERT_NEAR(-0.7, simplemc::map_to_interval(-0.7 - 4 * pi, lb, ub), tol);
    ASSERT_NEAR(-0.7 + pi, simplemc::map_to_interval(-0.7 - 3 * pi, lb, ub), tol);
    ASSERT_NEAR(-0.7, simplemc::map_to_interval(-0.7 + 12 * pi, lb, ub), tol);
    ASSERT_NEAR(0.0, simplemc::map_to_interval_lb(0.0, lb, ub), tol);
    ASSERT_NEAR(lb, simplemc::map_to_interval_lb(ub, lb, ub), tol);
    ASSERT_NEAR(lb, simplemc::map_to_interval_lb(lb, lb, ub), tol);
    ASSERT_NEAR(lb, simplemc::map_to_interval_lb(3 * pi, lb, ub), tol);
    ASSERT_NEAR(lb, simplemc::map_to_interval_lb(-3 * pi, lb, ub), tol);
    ASSERT_NEAR(-0.7, simplemc::map_to_interval_lb(-0.7 - 4 * pi, lb, ub), tol);
    ASSERT_NEAR(-0.7 + pi, simplemc::map_to_interval_lb(-0.7 - 3 * pi, lb, ub), tol);
    ASSERT_NEAR(-0.7, simplemc::map_to_interval_lb(-0.7 + 12 * pi, lb, ub), tol);
    lb = 1.1;
    ub = 2.3;
    ASSERT_NEAR(1.7, simplemc::map_to_interval(1.7, lb, ub), tol);
    ASSERT_NEAR(ub, simplemc::map_to_interval(ub, lb, ub), tol);
    ASSERT_NEAR(ub, simplemc::map_to_interval(lb, lb, ub), tol);
    ASSERT_NEAR(1.2, simplemc::map_to_interval(1.2 + 3 * 1.2, lb, ub), tol);
    ASSERT_NEAR(1.9, simplemc::map_to_interval(1.9 - 3 * 1.2, lb, ub), tol);
    ASSERT_NEAR(1.7, simplemc::map_to_interval_lb(1.7, lb, ub), tol);
    ASSERT_NEAR(lb, simplemc::map_to_interval_lb(ub, lb, ub), tol);
    ASSERT_NEAR(lb, simplemc::map_to_interval_lb(lb, lb, ub), tol);
    ASSERT_NEAR(1.2, simplemc::map_to_interval_lb(1.2 + 3 * 1.2, lb, ub), tol);
    ASSERT_NEAR(1.9, simplemc::map_to_interval_lb(1.9 - 3 * 1.2, lb, ub), tol);
    lb = -2.3;
    ub = -1.1;
    ASSERT_NEAR(-1.7, simplemc::map_to_interval(-1.7, lb, ub), tol);
    ASSERT_NEAR(ub, simplemc::map_to_interval(ub, lb, ub), tol);
    ASSERT_NEAR(ub, simplemc::map_to_interval(lb, lb, ub), tol);
    ASSERT_NEAR(-1.2, simplemc::map_to_interval(-1.2 + 3 * 1.2, lb, ub), tol);
    ASSERT_NEAR(-1.9, simplemc::map_to_interval(-1.9 - 3 * 1.2, lb, ub), tol);
    ASSERT_NEAR(-1.7, simplemc::map_to_interval_lb(-1.7, lb, ub), tol);
    ASSERT_NEAR(lb, simplemc::map_to_interval_lb(ub, lb, ub), tol);
    ASSERT_NEAR(lb, simplemc::map_to_interval_lb(lb, lb, ub), tol);
    ASSERT_NEAR(-1.2, simplemc::map_to_interval_lb(-1.2 + 3 * 1.2, lb, ub), tol);
    ASSERT_NEAR(-1.9, simplemc::map_to_interval_lb(-1.9 - 3 * 1.2, lb, ub), tol);
}

// Test within_bounds function.
TEST(SimplemcNumeric, WithinBounds) {
    double min_diff = 1e-3;
    double low = 4.2;
    double up = 5.0;
    double val_in1 = 4.5;
    double val_in2 = low + 1e-4;
    double val_in3 = up - 1e-4;
    ASSERT_TRUE(simplemc::within_bounds(val_in1, low, up, min_diff));
    ASSERT_FALSE(simplemc::within_bounds(val_in2, low, up, min_diff));
    ASSERT_FALSE(simplemc::within_bounds(val_in3, low, up, min_diff));
}

// Test the make_span function for Eigen types.
TEST(SimplemcNumeric, MakeSpan) {
    Eigen::Vector3d v3 = Eigen::Vector3d::Random();
    auto sp_v3 = simplemc::make_span(v3);
    ASSERT_EQ(sp_v3.size(), v3.size());
    ASSERT_EQ(sp_v3.data(), &v3(0));
    Eigen::VectorXd vn = Eigen::VectorXd::Random(10);
    auto sp_vn = simplemc::make_span(vn);
    ASSERT_EQ(sp_vn.size(), vn.size());
    ASSERT_EQ(sp_vn.data(), &vn(0));
    Eigen::Matrix3cd m3 = Eigen::Matrix3cd::Random();
    auto sp_m3 = simplemc::make_span(m3);
    ASSERT_EQ(sp_m3.size(), m3.size());
    ASSERT_EQ(sp_m3.data(), &m3(0, 0));
    Eigen::MatrixXi mn = Eigen::MatrixXi::Random(10, 10);
    auto sp_mn = simplemc::make_span(mn);
    ASSERT_EQ(sp_mn.size(), mn.size());
    ASSERT_EQ(sp_mn.data(), &mn(0, 0));
    const Eigen::Array44f a44 = Eigen::Array44f::Random();
    auto sp_a44 = simplemc::make_span(a44);
    ASSERT_EQ(sp_a44.size(), a44.size());
    ASSERT_EQ(sp_a44.data(), &a44(0, 0));
}

// Test some Eigen specific extensions.
TEST(SimplemcNumeric, EigenFunctions) {
    // isfinite
    constexpr auto inf = std::numeric_limits<double>::infinity();
    constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
    Eigen::Vector3d v { 1.0, 2.0, 3.0 };
    Eigen::Vector3d v_inf { 1.0, 2.0, inf };
    Eigen::Vector3d v_nan { 1.0, nan, 3.0 };
    ASSERT_TRUE(simplemc::isfinite(v));
    ASSERT_FALSE(simplemc::isfinite(v_inf));
    ASSERT_FALSE(simplemc::isfinite(v_nan));
    Eigen::MatrixXd mat = Eigen::MatrixXd::Random(10, 15);
    ASSERT_TRUE(simplemc::isfinite(mat));
    mat(5, 5) = inf;
    ASSERT_FALSE(simplemc::isfinite(mat));

    // abs_diff and rel_diff for vectors
    Eigen::Vector3d v1 { 1.0, 2.0, 3.0 };
    Eigen::Vector3d v2 { 1.0, 2.0, 3.0 + 1e-5 };
    check_near(simplemc::abs_diff(v1, v2), 1e-5);
    ASSERT_TRUE(simplemc::rel_diff(v1, v2) < 1e-5);

    // abs_diff and rel_diff for matrices
    Eigen::Matrix2d m1 = Eigen::Matrix2d::Identity();
    Eigen::Matrix2d m2 = Eigen::Matrix2d::Identity();
    ASSERT_DOUBLE_EQ(simplemc::abs_diff(m1, m2), 0.0);
    ASSERT_DOUBLE_EQ(simplemc::rel_diff(m1, m2), 0.0);
    m2(0, 0) = 1.0 + 1e-6;
    ASSERT_TRUE(simplemc::abs_diff(m1, m2) > 0);
    ASSERT_TRUE(simplemc::rel_diff(m1, m2) > 0);

    // rel_diff with zero norm (edge case)
    Eigen::Vector2d v_zero = Eigen::Vector2d::Zero();
    Eigen::Vector2d v_small { 1e-10, 0.0 };
    ASSERT_TRUE(simplemc::rel_diff(v_zero, v_small) > 0);
    ASSERT_TRUE(simplemc::rel_diff(v_small, v_zero) > 0);
}

// Test conversion between cartesian and polar coordinates.
TEST(SimplemcNumeric, PolarCartesianConversion) {
    // 1D (identity transformation)
    Eigen::Matrix<double, 1, 1> v1_c { 3.14 };
    Eigen::Matrix<double, 1, 1> v1_p { 3.14 };
    check_range_near(simplemc::cartesian_to_polar(v1_c), v1_p);
    check_range_near(simplemc::polar_to_cartesian(v1_p), v1_c);

    // 2D
    Eigen::Vector2d v2_p { std::numbers::sqrt2, std::numbers::pi / 4 };
    Eigen::Vector2d v2_c { 1.0, 1.0 };
    check_range_near(simplemc::polar_to_cartesian(v2_p), v2_c);
    check_range_near(simplemc::cartesian_to_polar(v2_c), v2_p);

    // 3D
    Eigen::Vector3d v3_c { 1.0, 0.0, 0.0 };
    Eigen::Vector3d v3_p { 1.0, std::numbers::pi / 2, 0.0 };
    check_range_near(simplemc::cartesian_to_polar(v3_c), v3_p);
    check_range_near(simplemc::polar_to_cartesian(v3_p), v3_c);
}

// Test the angle between two vectors.
TEST(SimplemcNumeric, AngleBetweenVectors) {
    using std::numbers::pi;

    // 2D
    Eigen::Vector2d v1 { 1, 0 };
    Eigen::Vector2d v2 { 1, 0 };
    ASSERT_DOUBLE_EQ(simplemc::angle(v1, v2), 0);
    v2 = { -1, 0 };
    ASSERT_DOUBLE_EQ(simplemc::angle(v1, v2), pi);
    v2 = { 0, 1 };
    ASSERT_DOUBLE_EQ(simplemc::angle(v1, v2), pi / 2);
    v2 = { 1, 1 };
    ASSERT_DOUBLE_EQ(simplemc::angle(v1, v2), pi / 4);
    v2 = { 1, -1 };
    ASSERT_DOUBLE_EQ(simplemc::angle(v1, v2), pi / 4);
    v2 = { -1, 1 };
    ASSERT_DOUBLE_EQ(simplemc::angle(v1, v2), 3 * pi / 4);
    v2 = { -1, -1 };
    ASSERT_DOUBLE_EQ(simplemc::angle(v1, v2), 3 * pi / 4);

    // 3D
    Eigen::Vector3d v3 { 1, 0, 0 };
    Eigen::Vector3d v4 { 1, 0, 0 };
    ASSERT_DOUBLE_EQ(simplemc::angle(v3, v4), 0);
    v4 = { -1, 0, 0 };
    ASSERT_DOUBLE_EQ(simplemc::angle(v3, v4), pi);
    v4 = { 0, 0, 1 };
    ASSERT_DOUBLE_EQ(simplemc::angle(v3, v4), pi / 2);
    v4 = { 1, 1, 0 };
    ASSERT_DOUBLE_EQ(simplemc::angle(v3, v4), pi / 4);
    v4 = { 1, 0, 1 };
    ASSERT_DOUBLE_EQ(simplemc::angle(v3, v4), pi / 4);
}

// Test the check for singular matrix.
TEST(SimplemcNumeric, IsMatrixSingular) {
    // 1D
    ASSERT_TRUE(simplemc::is_matrix_singular(Eigen::Matrix<double, 1, 1> { 0 }));
    ASSERT_FALSE(simplemc::is_matrix_singular(Eigen::Matrix<double, 1, 1> { 1e-12 }));

    // 2D
    auto mat_2d = Eigen::Matrix2d { { 1, 0 }, { 1, 0 } };
    ASSERT_TRUE(simplemc::is_matrix_singular(mat_2d));
    auto mat_eps = Eigen::Matrix2d { { 0, 0 }, { 0, 1e-12 } };
    ASSERT_FALSE(simplemc::is_matrix_singular(mat_2d + mat_eps));
    ASSERT_TRUE(simplemc::is_matrix_singular(mat_2d + mat_eps - mat_eps));
}

// Test linear maps.
TEST(SimplemcNumeric, LinearMap) {
    constexpr auto inf = std::numeric_limits<double>::infinity();
    constexpr auto minus_inf = -inf;
    using lm = simplemc::linear_map;
    auto check_map = [](const lm& map, std::array<double, 2> int1, std::array<double, 2> int2) {
        ASSERT_EQ(int2[0], map.map(int1[0]));
        ASSERT_EQ(int2[1], map.map(int1[1]));
        ASSERT_EQ(int1[0], map.inverse_map(int2[0]));
        ASSERT_EQ(int1[1], map.inverse_map(int2[1]));
    };
    check_map(lm { { -1, 1 }, { 0, 80 } }, { -1, 1 }, { 0, 80 });
    check_map(lm { { minus_inf, 1 }, { minus_inf, 80 } }, { minus_inf, 1 }, { minus_inf, 80 });
    check_map(lm { { -1, inf }, { 0, inf } }, { -1, inf }, { 0, inf });
    check_map(lm { { minus_inf, 1 }, { minus_inf, 80 } }, { minus_inf, 1 }, { minus_inf, 80 });
    check_map(lm { { minus_inf, inf }, { minus_inf, inf } }, { minus_inf, inf }, { minus_inf, inf });
}

// Test linear map getters.
TEST(SimplemcNumeric, LinearMapGetters) {
    simplemc::linear_map map { { -1.0, 1.0 }, { 0.0, 80.0 } };

    // domain() getter
    auto dom = map.domain();
    ASSERT_DOUBLE_EQ(dom[0], -1.0);
    ASSERT_DOUBLE_EQ(dom[1], 1.0);

    // range() getter
    auto rg = map.range();
    ASSERT_DOUBLE_EQ(rg[0], 0.0);
    ASSERT_DOUBLE_EQ(rg[1], 80.0);

    // params() getter
    auto [alpha, beta] = map.params();
    ASSERT_DOUBLE_EQ(alpha, 40.0);
    ASSERT_DOUBLE_EQ(beta, 40.0);

    // verify mapping is consistent with params
    ASSERT_DOUBLE_EQ(map.map(0.0), alpha * 0.0 + beta);
    ASSERT_DOUBLE_EQ(map.map(0.5), alpha * 0.5 + beta);
}

// Test linear map set() method and exceptions.
TEST(SimplemcNumeric, LinearMapSetAndExceptions) {
    // set() method
    simplemc::linear_map map {};
    map.set({ 0.0, 10.0 }, { 0.0, 100.0 });
    ASSERT_DOUBLE_EQ(map.map(5.0), 50.0);
    ASSERT_DOUBLE_EQ(map.inverse_map(50.0), 5.0);

    // exception for invalid domain (a >= b)
    ASSERT_THROW(simplemc::linear_map({ 1.0, 1.0 }, { 0.0, 10.0 }), simplemc::simplemc_exception);
    ASSERT_THROW(simplemc::linear_map({ 2.0, 1.0 }, { 0.0, 10.0 }), simplemc::simplemc_exception);

    // exception for invalid range (c >= d)
    ASSERT_THROW(simplemc::linear_map({ 0.0, 1.0 }, { 10.0, 10.0 }), simplemc::simplemc_exception);
    ASSERT_THROW(simplemc::linear_map({ 0.0, 1.0 }, { 20.0, 10.0 }), simplemc::simplemc_exception);

    // exception for incompatible infinite intervals
    constexpr auto inf = std::numeric_limits<double>::infinity();
    ASSERT_THROW(simplemc::linear_map({ -inf, inf }, { 0.0, 1.0 }), simplemc::simplemc_exception);
    ASSERT_THROW(simplemc::linear_map({ 0.0, 1.0 }, { -inf, inf }), simplemc::simplemc_exception);
    ASSERT_THROW(simplemc::linear_map({ -inf, 1.0 }, { 0.0, inf }), simplemc::simplemc_exception);
}
