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

    simplemc::json_serializer::save_to_file("mean_acc.json", a);

    simplemc::mean_acc<double, simplemc::varalg::welford> b {};
    simplemc::json_deserializer::load_from_file("mean_acc.json", b);

    EXPECT_EQ(b.count(), a.count());
    check_near(b.mean(), a.mean());
}

TEST(SerializersJson, MeanAccVector) {
    using vec_t = Eigen::Vector3d;
    simplemc::mean_acc<vec_t, simplemc::varalg::welford> a {};
    a << vec_t { 1.0, 2.0, 3.0 } << vec_t { 4.0, 5.0, 6.0 };

    simplemc::json_serializer::save_to_file("mean_acc_vec.json", a);

    simplemc::mean_acc<vec_t, simplemc::varalg::welford> b {};
    simplemc::json_deserializer::load_from_file("mean_acc_vec.json", b);

    EXPECT_EQ(b.count(), a.count());
    check_near(b.mean(), a.mean());
}

TEST(SerializersJson, VarAccScalar) {
    simplemc::var_acc<double, simplemc::varalg::welford> a {};
    a << 1.0 << 2.0 << 3.0 << 4.0 << 5.0;

    simplemc::json_serializer::save_to_file("var_acc.json", a);

    simplemc::var_acc<double, simplemc::varalg::welford> b {};
    simplemc::json_deserializer::load_from_file("var_acc.json", b);

    EXPECT_EQ(b.count(), a.count());
    check_near(b.mean(), a.mean());
    check_near(b.variance(), a.variance());
}

TEST(SerializersJson, CovarAccScalar) {
    simplemc::covar_acc<double, simplemc::varalg::welford> a {};
    a << 1.0 << 2.0 << 3.0 << 4.0;

    simplemc::json_serializer::save_to_file("covar_acc.json", a);

    simplemc::covar_acc<double, simplemc::varalg::welford> b {};
    simplemc::json_deserializer::load_from_file("covar_acc.json", b);

    EXPECT_EQ(b.count(), a.count());
    check_near(b.mean(), a.mean());
}

TEST(SerializersJson, BatchAccScalar) {
    simplemc::batch_acc<double, simplemc::varalg::welford> a { 1, 8 };
    for (int i = 0; i < 50; ++i) {
        a << static_cast<double>(i);
    }

    simplemc::json_serializer::save_to_file("batch_acc.json", a);

    simplemc::batch_acc<double, simplemc::varalg::welford> b { 1, 8 };
    simplemc::json_deserializer::load_from_file("batch_acc.json", b);

    EXPECT_EQ(b.count(), a.count());
    check_near(b.mean(), a.mean());
}

// ===== Grids ===========================================================================

TEST(SerializersJson, LinearGrid) {
    simplemc::linear_grid g { 0.0, 10.0, 11 };
    simplemc::json_serializer::save_to_file("linear_grid.json", g);

    simplemc::linear_grid h;
    simplemc::json_deserializer::load_from_file("linear_grid.json", h);

    EXPECT_DOUBLE_EQ(h.first(), g.first());
    EXPECT_DOUBLE_EQ(h.last(), g.last());
    EXPECT_EQ(h.size(), g.size());
}

TEST(SerializersJson, PowerGrid) {
    simplemc::power_grid g { 0.0, 1.0, 11, 2.0 };
    simplemc::json_serializer::save_to_file("power_grid.json", g);

    simplemc::power_grid h;
    simplemc::json_deserializer::load_from_file("power_grid.json", h);

    EXPECT_DOUBLE_EQ(h.first(), g.first());
    EXPECT_DOUBLE_EQ(h.last(), g.last());
    EXPECT_EQ(h.size(), g.size());
    EXPECT_DOUBLE_EQ(h.power(), g.power());
}

TEST(SerializersJson, SymmetricPowerGrid) {
    simplemc::symmetric_power_grid g { 0.0, 1.0, 9, 2.0 };
    simplemc::json_serializer::save_to_file("sym_power_grid.json", g);

    simplemc::symmetric_power_grid h;
    simplemc::json_deserializer::load_from_file("sym_power_grid.json", h);

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
    simplemc::json_serializer::save_to_file("nd_grid.json", g);

    simplemc::nd_grid<simplemc::linear_grid, simplemc::power_grid> h;
    simplemc::json_deserializer::load_from_file("nd_grid.json", h);

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
    simplemc::json_serializer::save_to_file("lattice_2d.json", lat);

    simplemc::bravais_lattice<2> lat2;
    simplemc::json_deserializer::load_from_file("lattice_2d.json", lat2);

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
    simplemc::json_serializer::save_to_file("xoshiro.json", rng);

    simplemc::xoshiro256p rng2 { 0ULL };
    simplemc::json_deserializer::load_from_file("xoshiro.json", rng2);

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

    simplemc::json_serializer::save_to_file("splitmix.json", rng);

    simplemc::splitmix64 rng2 { 0ULL };
    simplemc::json_deserializer::load_from_file("splitmix.json", rng2);

    EXPECT_EQ(rng2.internal_state(), state_before);
}
