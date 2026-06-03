/**
 * @file
 * @brief Per-library-type round-trip tests for simplemc-serialize-json.
 *
 * @details Exercises the library-shipped `simplemc_save` / `simplemc_load` overloads against the
 * JSON backend: accumulators (mean_acc, var_acc, covar_acc, batch_acc), grids (linear, power,
 * symmetric_power, nd_grid), bravais_lattice, and RNG state (xoshiro256, splitmix64). Each test
 * builds an instance, exercises it briefly (where meaningful), round-trips through file IO, and
 * asserts state recovery.
 */

#include "../../test_utils.hpp"

#include <simplemc/accs/autocorr_acc.hpp>
#include <simplemc/accs/batch_acc.hpp>
#include <simplemc/accs/block_acc.hpp>
#include <simplemc/accs/covar_acc.hpp>
#include <simplemc/accs/mean_acc.hpp>
#include <simplemc/accs/var_acc.hpp>
#include <simplemc/grids/custom_grid.hpp>
#include <simplemc/grids/linear_grid.hpp>
#include <simplemc/grids/nd_grid.hpp>
#include <simplemc/grids/power_grid.hpp>
#include <simplemc/grids/symmetric_power_grid.hpp>
#include <simplemc/numeric/bravais_lattice.hpp>
#include <simplemc/random/splitmix64.hpp>
#include <simplemc/random/xoshiro256.hpp>
#include <simplemc/serialize.hpp>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <complex>
#include <random>
#include <vector>

// ===== Accumulators ====================================================================

TEST(SerializersJson, MeanAccScalar) {
    simplemc::mean_acc<double, simplemc::varalg::welford> a {};
    a << 1.0 << 2.0 << 3.0 << 4.0;

    simplemc::json_serializer s;
    simplemc_save(s, a);
    s.write_to_file("mean_acc.json");

    simplemc::mean_acc<double, simplemc::varalg::welford> b {};
    simplemc::json_deserializer d { "mean_acc.json" };
    simplemc_load(d, b);

    EXPECT_EQ(b.count(), a.count());
    check_near(b.mean(), a.mean());
}

TEST(SerializersJson, MeanAccVector) {
    using vec_t = Eigen::Vector3d;
    simplemc::mean_acc<vec_t, simplemc::varalg::welford> a {};
    a << vec_t { 1.0, 2.0, 3.0 } << vec_t { 4.0, 5.0, 6.0 };

    simplemc::json_serializer s;
    simplemc_save(s, a);
    s.write_to_file("mean_acc_vec.json");

    simplemc::mean_acc<vec_t, simplemc::varalg::welford> b {};
    simplemc::json_deserializer d { "mean_acc_vec.json" };
    simplemc_load(d, b);

    EXPECT_EQ(b.count(), a.count());
    check_near(b.mean(), a.mean());
}

TEST(SerializersJson, VarAccScalar) {
    simplemc::var_acc<double, simplemc::varalg::welford> a {};
    a << 1.0 << 2.0 << 3.0 << 4.0 << 5.0;

    simplemc::json_serializer s;
    simplemc_save(s, a);
    s.write_to_file("var_acc.json");

    simplemc::var_acc<double, simplemc::varalg::welford> b {};
    simplemc::json_deserializer d { "var_acc.json" };
    simplemc_load(d, b);

    EXPECT_EQ(b.count(), a.count());
    check_near(b.mean(), a.mean());
    check_near(b.variance(), a.variance());
}

TEST(SerializersJson, VarAccVector) {
    using vec_t = Eigen::Vector3d;
    simplemc::var_acc<vec_t, simplemc::varalg::welford> a {};
    a << vec_t { 1.0, 2.0, 3.0 } << vec_t { 4.0, 5.0, 6.0 } << vec_t { -1.0, 0.5, 2.0 };

    simplemc::json_serializer s;
    simplemc_save(s, a);
    s.write_to_file("var_acc_vec.json");

    simplemc::var_acc<vec_t, simplemc::varalg::welford> b {};
    simplemc::json_deserializer d { "var_acc_vec.json" };
    simplemc_load(d, b);

    EXPECT_EQ(b.count(), a.count());
    check_near(b.mean(), a.mean());
    check_near(b.variance(), a.variance());
}

TEST(SerializersJson, VarAccComplexScalar) {
    using cplx = std::complex<double>;
    simplemc::var_acc<cplx, simplemc::varalg::welford> a {};
    a << cplx { 1.0, -1.0 } << cplx { 2.0, 0.5 } << cplx { 3.0, 1.5 } << cplx { -1.0, 2.0 };

    simplemc::json_serializer s;
    simplemc_save(s, a);
    s.write_to_file("var_acc_cplx.json");

    simplemc::var_acc<cplx, simplemc::varalg::welford> b {};
    simplemc::json_deserializer d { "var_acc_cplx.json" };
    simplemc_load(d, b);

    EXPECT_EQ(b.count(), a.count());
    check_near(b.mean(), a.mean());
    check_near(b.variance(), a.variance());
    check_near(b.variance_of_real_data(), a.variance_of_real_data());
    check_near(b.variance_of_imag_data(), a.variance_of_imag_data());
    check_near(b.covariance_of_real_and_imag_data(), a.covariance_of_real_and_imag_data());
}

TEST(SerializersJson, VarAccComplexVector) {
    using vec_t = Eigen::Vector2cd;
    using cplx = std::complex<double>;
    simplemc::var_acc<vec_t, simplemc::varalg::standard> a {};
    a << vec_t { cplx { 1.0, 2.0 }, cplx { -1.0, 0.5 } };
    a << vec_t { cplx { 3.0, -2.0 }, cplx { 2.0, 1.0 } };
    a << vec_t { cplx { 0.5, 0.5 }, cplx { 1.5, -0.5 } };

    simplemc::json_serializer s;
    simplemc_save(s, a);
    s.write_to_file("var_acc_cplx_vec.json");

    simplemc::var_acc<vec_t, simplemc::varalg::standard> b {};
    simplemc::json_deserializer d { "var_acc_cplx_vec.json" };
    simplemc_load(d, b);

    EXPECT_EQ(b.count(), a.count());
    check_near(b.mean(), a.mean());
    check_near(b.variance(), a.variance());
    check_near(b.variance_of_real_data(), a.variance_of_real_data());
    check_near(b.variance_of_imag_data(), a.variance_of_imag_data());
}

TEST(SerializersJson, CovarAccScalar) {
    simplemc::covar_acc<double, simplemc::varalg::welford> a {};
    a << 1.0 << 2.0 << 3.0 << 4.0;

    simplemc::json_serializer s;
    simplemc_save(s, a);
    s.write_to_file("covar_acc.json");

    simplemc::covar_acc<double, simplemc::varalg::welford> b {};
    simplemc::json_deserializer d { "covar_acc.json" };
    simplemc_load(d, b);

    EXPECT_EQ(b.count(), a.count());
    check_near(b.mean(), a.mean());
}

TEST(SerializersJson, CovarAccVector) {
    using vec_t = Eigen::Vector2d;
    simplemc::covar_acc<vec_t, simplemc::varalg::welford> a {};
    a << vec_t { 1.0, 2.0 } << vec_t { 4.0, 5.0 } << vec_t { -1.0, 0.5 };

    simplemc::json_serializer s;
    simplemc_save(s, a);
    s.write_to_file("covar_acc_vec.json");

    simplemc::covar_acc<vec_t, simplemc::varalg::welford> b {};
    simplemc::json_deserializer d { "covar_acc_vec.json" };
    simplemc_load(d, b);

    EXPECT_EQ(b.count(), a.count());
    check_near(b.mean(), a.mean());
    check_near(b.covariance(), a.covariance());
}

TEST(SerializersJson, CovarAccComplexScalar) {
    using cplx = std::complex<double>;
    simplemc::covar_acc<cplx, simplemc::varalg::welford> a {};
    a << cplx { 1.0, -1.0 } << cplx { 2.0, 0.5 } << cplx { 3.0, 1.5 } << cplx { -1.0, 2.0 };

    simplemc::json_serializer s;
    simplemc_save(s, a);
    s.write_to_file("covar_acc_cplx.json");

    simplemc::covar_acc<cplx, simplemc::varalg::welford> b {};
    simplemc::json_deserializer d { "covar_acc_cplx.json" };
    simplemc_load(d, b);

    EXPECT_EQ(b.count(), a.count());
    check_near(b.mean(), a.mean());
}

TEST(SerializersJson, CovarAccComplexVector) {
    using vec_t = Eigen::Vector2cd;
    using cplx = std::complex<double>;
    simplemc::covar_acc<vec_t, simplemc::varalg::standard> a {};
    a << vec_t { cplx { 1.0, 2.0 }, cplx { -1.0, 0.5 } };
    a << vec_t { cplx { 3.0, -2.0 }, cplx { 2.0, 1.0 } };
    a << vec_t { cplx { 0.5, 0.5 }, cplx { 1.5, -0.5 } };

    simplemc::json_serializer s;
    simplemc_save(s, a);
    s.write_to_file("covar_acc_cplx_vec.json");

    simplemc::covar_acc<vec_t, simplemc::varalg::standard> b {};
    simplemc::json_deserializer d { "covar_acc_cplx_vec.json" };
    simplemc_load(d, b);

    EXPECT_EQ(b.count(), a.count());
    check_near(b.mean(), a.mean());
}

TEST(SerializersJson, BatchAccScalar) {
    simplemc::batch_acc<double, simplemc::varalg::welford> a { 1, 8 };
    for (int i = 0; i < 50; ++i) {
        a << static_cast<double>(i);
    }

    simplemc::json_serializer s;
    simplemc_save(s, a);
    s.write_to_file("batch_acc.json");

    simplemc::batch_acc<double, simplemc::varalg::welford> b { 1, 8 };
    simplemc::json_deserializer d { "batch_acc.json" };
    simplemc_load(d, b);

    EXPECT_EQ(b.count(), a.count());
    check_near(b.mean(), a.mean());
}

TEST(SerializersJson, BlockAccVar) {
    simplemc::block_acc<simplemc::var_acc<double, simplemc::varalg::welford>> a { 4 };
    for (int i = 0; i < 17; ++i) {
        a << static_cast<double>(i);
    }

    simplemc::json_serializer s;
    simplemc_save(s, a);
    s.write_to_file("block_var_acc.json");

    simplemc::block_acc<simplemc::var_acc<double, simplemc::varalg::welford>> b { 4 };
    simplemc::json_deserializer d { "block_var_acc.json" };
    simplemc_load(d, b);

    EXPECT_EQ(b.block_size(), a.block_size());
    EXPECT_EQ(b.count(), a.count());
    EXPECT_EQ(b.total_count(), a.total_count());
    check_near(b.mean(), a.mean());
    check_near(b.variance(), a.variance());
}

TEST(SerializersJson, BlockAccCovar) {
    using vec_t = Eigen::Vector2d;
    simplemc::block_acc<simplemc::covar_acc<vec_t, simplemc::varalg::welford>> a { 3, 2 };
    for (int i = 0; i < 13; ++i) {
        a << vec_t { static_cast<double>(i), static_cast<double>(2 * i + 1) };
    }

    simplemc::json_serializer s;
    simplemc_save(s, a);
    s.write_to_file("block_covar_acc.json");

    simplemc::block_acc<simplemc::covar_acc<vec_t, simplemc::varalg::welford>> b { 3, 2 };
    simplemc::json_deserializer d { "block_covar_acc.json" };
    simplemc_load(d, b);

    EXPECT_EQ(b.block_size(), a.block_size());
    EXPECT_EQ(b.count(), a.count());
    check_near(b.mean(), a.mean());
    check_near(b.covariance(), a.covariance());
}

TEST(SerializersJson, BlockAccVarComplex) {
    using cplx = std::complex<double>;
    simplemc::block_acc<simplemc::var_acc<cplx, simplemc::varalg::welford>> a { 4 };
    for (int i = 0; i < 17; ++i) {
        a << cplx { static_cast<double>(i), static_cast<double>(-i) };
    }

    simplemc::json_serializer s;
    simplemc_save(s, a);
    s.write_to_file("block_var_acc_cplx.json");

    simplemc::block_acc<simplemc::var_acc<cplx, simplemc::varalg::welford>> b { 4 };
    simplemc::json_deserializer d { "block_var_acc_cplx.json" };
    simplemc_load(d, b);

    EXPECT_EQ(b.block_size(), a.block_size());
    EXPECT_EQ(b.count(), a.count());
    check_near(b.mean(), a.mean());
    check_near(b.variance(), a.variance());
}

TEST(SerializersJson, AutocorrAccVar) {
    simplemc::autocorr_acc<simplemc::var_acc<double, simplemc::varalg::welford>> a {};
    // Use enough samples to grow several levels (default factor=2, min_levels=2).
    for (int i = 0; i < 1000; ++i) {
        a << static_cast<double>(i);
    }

    simplemc::json_serializer s;
    simplemc_save(s, a);
    s.write_to_file("autocorr_var_acc.json");

    simplemc::autocorr_acc<simplemc::var_acc<double, simplemc::varalg::welford>> b {};
    simplemc::json_deserializer d { "autocorr_var_acc.json" };
    simplemc_load(d, b);

    EXPECT_EQ(b.factor(), a.factor());
    EXPECT_EQ(b.min_levels(), a.min_levels());
    EXPECT_EQ(b.num_levels(), a.num_levels());
    for (std::size_t l = 0; l < a.num_levels(); ++l) {
        EXPECT_EQ(b.count(l), a.count(l));
        EXPECT_EQ(b.block_size(l), a.block_size(l));
    }
    check_near(b.mean(), a.mean());
    check_near(b.variance(), a.variance());
}

TEST(SerializersJson, AutocorrAccCovar) {
    using vec_t = Eigen::Vector2d;
    simplemc::autocorr_acc<simplemc::covar_acc<vec_t, simplemc::varalg::welford>> a { 2 };
    for (int i = 0; i < 512; ++i) {
        a << vec_t { static_cast<double>(i), static_cast<double>(0.5 * i) };
    }

    simplemc::json_serializer s;
    simplemc_save(s, a);
    s.write_to_file("autocorr_covar_acc.json");

    simplemc::autocorr_acc<simplemc::covar_acc<vec_t, simplemc::varalg::welford>> b { 2 };
    simplemc::json_deserializer d { "autocorr_covar_acc.json" };
    simplemc_load(d, b);

    EXPECT_EQ(b.factor(), a.factor());
    EXPECT_EQ(b.min_levels(), a.min_levels());
    EXPECT_EQ(b.num_levels(), a.num_levels());
    check_near(b.mean(), a.mean());
    check_near(b.covariance(), a.covariance());
}

TEST(SerializersJson, AutocorrAccVarComplex) {
    using cplx = std::complex<double>;
    simplemc::autocorr_acc<simplemc::var_acc<cplx, simplemc::varalg::welford>> a {};
    for (int i = 0; i < 512; ++i) {
        a << cplx { static_cast<double>(i), static_cast<double>(-i) };
    }

    simplemc::json_serializer s;
    simplemc_save(s, a);
    s.write_to_file("autocorr_var_acc_cplx.json");

    simplemc::autocorr_acc<simplemc::var_acc<cplx, simplemc::varalg::welford>> b {};
    simplemc::json_deserializer d { "autocorr_var_acc_cplx.json" };
    simplemc_load(d, b);

    EXPECT_EQ(b.factor(), a.factor());
    EXPECT_EQ(b.min_levels(), a.min_levels());
    EXPECT_EQ(b.num_levels(), a.num_levels());
    check_near(b.mean(), a.mean());
    check_near(b.variance(), a.variance());
}

// ===== Grids ===========================================================================

TEST(SerializersJson, LinearGrid) {
    simplemc::linear_grid g { 0.0, 10.0, 11 };
    simplemc::json_serializer s;
    simplemc_save(s, g);
    s.write_to_file("linear_grid.json");

    simplemc::linear_grid h;
    simplemc::json_deserializer d { "linear_grid.json" };
    simplemc_load(d, h);

    EXPECT_DOUBLE_EQ(h.first(), g.first());
    EXPECT_DOUBLE_EQ(h.last(), g.last());
    EXPECT_EQ(h.size(), g.size());
}

TEST(SerializersJson, PowerGrid) {
    simplemc::power_grid g { 0.0, 1.0, 11, 2.0 };
    simplemc::json_serializer s;
    simplemc_save(s, g);
    s.write_to_file("power_grid.json");

    simplemc::power_grid h;
    simplemc::json_deserializer d { "power_grid.json" };
    simplemc_load(d, h);

    EXPECT_DOUBLE_EQ(h.first(), g.first());
    EXPECT_DOUBLE_EQ(h.last(), g.last());
    EXPECT_EQ(h.size(), g.size());
    EXPECT_DOUBLE_EQ(h.power(), g.power());
}

TEST(SerializersJson, CustomGrid) {
    simplemc::custom_grid g { std::vector<double> { 0.0, 0.1, 0.5, 1.7, 3.14, 10.0 } };
    simplemc::json_serializer s;
    simplemc_save(s, g);
    s.write_to_file("custom_grid.json");

    simplemc::custom_grid h;
    simplemc::json_deserializer d { "custom_grid.json" };
    simplemc_load(d, h);

    EXPECT_DOUBLE_EQ(h.first(), g.first());
    EXPECT_DOUBLE_EQ(h.last(), g.last());
    EXPECT_EQ(h.size(), g.size());
    check_near(h.grid_points(), g.grid_points());
}

TEST(SerializersJson, SymmetricPowerGrid) {
    simplemc::symmetric_power_grid g { 0.0, 1.0, 9, 2.0 };
    simplemc::json_serializer s;
    simplemc_save(s, g);
    s.write_to_file("sym_power_grid.json");

    simplemc::symmetric_power_grid h;
    simplemc::json_deserializer d { "sym_power_grid.json" };
    simplemc_load(d, h);

    EXPECT_DOUBLE_EQ(h.first(), g.first());
    EXPECT_DOUBLE_EQ(h.last(), g.last());
    EXPECT_EQ(h.size(), g.size());
    EXPECT_DOUBLE_EQ(h.grid1().power(), g.grid1().power());
}

TEST(SerializersJson, NdGrid) {
    simplemc::nd_grid<simplemc::linear_grid, simplemc::power_grid> g {
        simplemc::linear_grid { 0.0, 1.0, 5 },
        simplemc::power_grid { 0.0, 10.0, 11, 2.0 },
    };
    simplemc::json_serializer s;
    simplemc_save(s, g);
    s.write_to_file("nd_grid.json");

    simplemc::nd_grid<simplemc::linear_grid, simplemc::power_grid> h;
    simplemc::json_deserializer d { "nd_grid.json" };
    simplemc_load(d, h);

    EXPECT_DOUBLE_EQ(std::get<0>(h.grids()).first(), 0.0);
    EXPECT_DOUBLE_EQ(std::get<0>(h.grids()).last(), 1.0);
    EXPECT_EQ(std::get<0>(h.grids()).size(), 5);
    EXPECT_DOUBLE_EQ(std::get<1>(h.grids()).first(), 0.0);
    EXPECT_DOUBLE_EQ(std::get<1>(h.grids()).last(), 10.0);
    EXPECT_EQ(std::get<1>(h.grids()).size(), 11);
    EXPECT_DOUBLE_EQ(std::get<1>(h.grids()).power(), 2.0);
}

// ===== Bravais lattice =================================================================

TEST(SerializersJson, BravaisLattice2D) {
    Eigen::Matrix<double, 2, 2> A;
    A << 1.0, 0.0, 0.0, 2.0;
    simplemc::bravais_lattice<2> lat { A };
    simplemc::json_serializer s;
    simplemc_save(s, lat);
    s.write_to_file("lattice_2d.json");

    simplemc::bravais_lattice<2> lat2;
    simplemc::json_deserializer d { "lattice_2d.json" };
    simplemc_load(d, lat2);

    check_near(lat2.real_lattice_vectors(), lat.real_lattice_vectors());
    check_near(lat2.reciprocal_lattice_vectors(), lat.reciprocal_lattice_vectors());
    EXPECT_DOUBLE_EQ(lat2.real_cell_volume(), lat.real_cell_volume());
}

// ===== RNG state =======================================================================

TEST(SerializersJson, Xoshiro256Plus) {
    simplemc::xoshiro256p rng { 42ULL };
    // advance the state
    for (int i = 0; i < 100; ++i) {
        (void)rng();
    }
    simplemc::json_serializer s;
    simplemc_save(s, rng);
    s.write_to_file("xoshiro.json");

    simplemc::xoshiro256p rng2 { 0ULL };
    simplemc::json_deserializer d { "xoshiro.json" };
    simplemc_load(d, rng2);

    EXPECT_EQ(rng, rng2);
    // verify they continue to produce the same sequence
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(rng(), rng2());
    }
}

TEST(SerializersJson, Splitmix64) {
    simplemc::splitmix64 rng { 12345ULL };
    for (int i = 0; i < 100; ++i) {
        (void)rng();
    }
    auto state_before = rng.internal_state();

    simplemc::json_serializer s;
    simplemc_save(s, rng);
    s.write_to_file("splitmix.json");

    simplemc::splitmix64 rng2 { 0ULL };
    simplemc::json_deserializer d { "splitmix.json" };
    simplemc_load(d, rng2);

    EXPECT_EQ(rng2.internal_state(), state_before);
}
