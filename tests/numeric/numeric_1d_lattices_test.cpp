#include "../test_utils.hpp"

#include <simplemc/numeric/bravais_lattice_1d.hpp>

#include <numbers>

// Test 1D linear lattice.
TEST(SimplemcNumeric, LinearLattice) {
    using namespace simplemc;
    using std::numbers::pi;
    double tol = 1e-7;
    auto lat = make_linear_lattice(2);
    Eigen::Matrix<double, 1, 1> exp_real_lat;
    exp_real_lat << 2.0;
    check_range_near(make_span(lat.real_lattice_vectors()), make_span(exp_real_lat), tol);
    Eigen::Matrix<double, 1, 1> exp_rec_lat;
    exp_rec_lat << pi;
    check_range_near(make_span(lat.reciprocal_lattice_vectors()), make_span(exp_rec_lat), tol);
    double exp_real_vol = 2.0;
    ASSERT_NEAR(exp_real_vol, lat.real_cell_volume(), tol);
    double exp_rec_vol = pi;
    ASSERT_NEAR(exp_rec_vol, lat.reciprocal_cell_volume(), tol);
    ASSERT_EQ(std::string("linear_1d"), std::string(lattice_tag_to_string(lat.param().tag)));
    ASSERT_EQ(lattice_tag::linear_1d, string_to_lattice_tag("linear_1d"));
}
