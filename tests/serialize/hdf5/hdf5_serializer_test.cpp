#include "../serialize_test_utils.hpp"
#include "../test_types.hpp"

#include <simplemc/random/splitmix64.hpp>
#include <simplemc/random/xoshiro256.hpp>
#include <simplemc/serialize/hdf5.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <Eigen/Dense>
#include <fmt/format.h>

#include <complex>
#include <filesystem>
#include <vector>

namespace {

// Build a unique temp filename based on the current GoogleTest test name so that parallel runs do
// not collide.
std::filesystem::path temp_h5_path() {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    return std::filesystem::temp_directory_path() /
        fmt::format("simplemc_h5_{}_{}.h5", info->test_suite_name(), info->name());
}

// Round-trip src through a temp HDF5 file into dst. The file is removed afterwards.
template <class T>
void roundtrip_hdf5(const T& src, T& dst) {
    const auto path = temp_h5_path();
    {
        simplemc::hdf5_serializer s { path, simplemc::hdf5_file_mode::truncate };
        s.save_at("root", src);
    }
    {
        const simplemc::hdf5_serializer d { path, simplemc::hdf5_file_mode::read };
        d.load_at("root", dst);
    }
    std::filesystem::remove(path);
}

} // namespace

// Test serializer/deserializer concept conformance.
TEST(SerializerHdf5, ConceptConformance) {
    using simplemc::deserializer;
    using simplemc::hdf5_serializer;
    using simplemc::serializer;

    static_assert(serializer<hdf5_serializer>);
    static_assert(deserializer<hdf5_serializer>);
    SUCCEED();
}

// Test ADL dispatch to friend simplemc_save / simplemc_load functions.
TEST(SerializerHdf5, AdlDispatchFriendPath) {
    using test_types::intrusive_point;
    const intrusive_point a { 1.25, -3.5 };
    intrusive_point b;
    roundtrip_hdf5(a, b);
    ASSERT_EQ(b, a);
}

// Test ADL dispatch to namespace-scope simplemc_save / simplemc_load functions.
TEST(SerializerHdf5, AdlDispatchFreeFunctionPath) {
    using test_types::nonintrusive_box;
    const nonintrusive_box a { .width = 4, .height = 5 };
    nonintrusive_box b;
    roundtrip_hdf5(a, b);
    ASSERT_EQ(b, a);
}

// Test ADL dispatch to a composite type.
TEST(SerializerHdf5, AdlDispatchComposite) {
    using namespace test_types;
    const composite a { .pt = intrusive_point { 1.0, 2.0 },
        .bx = nonintrusive_box { .width = 10, .height = 20 },
        .data = { 0.5, 1.5, 2.5 } };
    composite b;
    roundtrip_hdf5(a, b);
    ASSERT_EQ(b, a);
}

// Test that operator[] navigation builds a nested group structure.
TEST(SerializerHdf5, OperatorBracketBuildsSubTree) {
    const auto path = temp_h5_path();
    {
        simplemc::hdf5_serializer s { path, simplemc::hdf5_file_mode::truncate };
        s.save_at("step", 1);
        s["results"].save_at("mean", 0.5);
        s["results"].save_at("var", 0.1);
    }

    const simplemc::hdf5_serializer d { path, simplemc::hdf5_file_mode::read };
    const auto& f = d.file();
    ASSERT_TRUE(f.exist("/step"));
    ASSERT_TRUE(f.exist("/results"));
    ASSERT_TRUE(f.exist("/results/mean"));
    ASSERT_TRUE(f.exist("/results/var"));
    EXPECT_EQ(f.getObjectType("/step"), HighFive::ObjectType::Dataset);
    EXPECT_EQ(f.getObjectType("/results"), HighFive::ObjectType::Group);
    EXPECT_EQ(f.getObjectType("/results/mean"), HighFive::ObjectType::Dataset);
    EXPECT_EQ(f.getObjectType("/results/var"), HighFive::ObjectType::Dataset);

    int step = 0;
    double mean = 0.0;
    double var = 0.0;
    d.load_at("step", step);
    d["results"].load_at("mean", mean);
    d["results"].load_at("var", var);
    EXPECT_EQ(step, 1);
    EXPECT_DOUBLE_EQ(mean, 0.5);
    EXPECT_DOUBLE_EQ(var, 0.1);
    std::filesystem::remove(path);
}

// Test chaining of operator[] calls.
TEST(SerializerHdf5, OperatorBracketChainedNavigation) {
    const auto path = temp_h5_path();
    {
        simplemc::hdf5_serializer s { path, simplemc::hdf5_file_mode::truncate };
        s["a"]["b"]["c"].save_at("x", 42);
    }
    const simplemc::hdf5_serializer d { path, simplemc::hdf5_file_mode::read };
    const auto& f = d.file();
    ASSERT_TRUE(f.exist("/a"));
    ASSERT_TRUE(f.exist("/a/b"));
    ASSERT_TRUE(f.exist("/a/b/c"));
    ASSERT_TRUE(f.exist("/a/b/c/x"));
    EXPECT_EQ(f.getObjectType("/a"), HighFive::ObjectType::Group);
    EXPECT_EQ(f.getObjectType("/a/b"), HighFive::ObjectType::Group);
    EXPECT_EQ(f.getObjectType("/a/b/c"), HighFive::ObjectType::Group);
    EXPECT_EQ(f.getObjectType("/a/b/c/x"), HighFive::ObjectType::Dataset);

    int x = 0;
    d["a"]["b"]["c"].load_at("x", x);
    EXPECT_EQ(x, 42);
    std::filesystem::remove(path);
}

// Test that operator[] on the read side throws if the key is missing.
TEST(SerializerHdf5, OperatorBracketReadSideThrowsOnMissing) {
    const auto path = temp_h5_path();
    {
        simplemc::hdf5_serializer s { path, simplemc::hdf5_file_mode::truncate };
        s.save_at("existing", 1);
    }
    const simplemc::hdf5_serializer d { path, simplemc::hdf5_file_mode::read };
    EXPECT_THROW({ auto sub = d["missing"]; }, simplemc::simplemc_exception);
    std::filesystem::remove(path);
}

// Test load_at on a missing key throws.
TEST(SerializerHdf5, LoadAtThrowsOnMissing) {
    const auto path = temp_h5_path();
    {
        simplemc::hdf5_serializer s { path, simplemc::hdf5_file_mode::truncate };
        s.save_at("present", 1);
    }
    const simplemc::hdf5_serializer d { path, simplemc::hdf5_file_mode::read };
    int x = 0;
    EXPECT_THROW(d.load_at("missing", x), simplemc::simplemc_exception);
    std::filesystem::remove(path);
}

// Test path tracking of operator[] calls.
TEST(SerializerHdf5, PathTracksPosition) {
    const auto path = temp_h5_path();
    simplemc::hdf5_serializer s { path, simplemc::hdf5_file_mode::truncate };
    EXPECT_EQ(s.path(), "/");
    auto sub = s["a"]["b"];
    EXPECT_EQ(sub.path(), "/a/b");
    std::filesystem::remove(path);
}

// Test presence of keys via has().
TEST(SerializerHdf5, HasKey) {
    const auto path = temp_h5_path();
    {
        simplemc::hdf5_serializer s { path, simplemc::hdf5_file_mode::truncate };
        s.save_at("present", 1);
        s["group"].save_at("inside", 2);
    }
    const simplemc::hdf5_serializer d { path, simplemc::hdf5_file_mode::read };
    EXPECT_TRUE(d.has("present"));
    EXPECT_TRUE(d.has("group"));
    EXPECT_FALSE(d.has("missing"));
    EXPECT_TRUE(d["group"].has("inside"));
    EXPECT_FALSE(d["group"].has("missing"));
    std::filesystem::remove(path);
}

// Test serializing/deserializing a mean_acc with real scalar sample type.
TEST(SerializerHdf5, MeanAccScalar) {
    simplemc::mean_acc<double> a;
    a << 1.0 << 2.0 << 3.0 << 4.0;
    simplemc::mean_acc<double> b;
    roundtrip_hdf5(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a mean_acc with real vector sample type.
TEST(SerializerHdf5, MeanAccVector) {
    using vec_t = Eigen::Vector3d;
    simplemc::mean_acc<vec_t> a;
    a << vec_t { 1.0, 2.0, 3.0 } << vec_t { 4.0, 5.0, 6.0 };
    simplemc::mean_acc<vec_t> b;
    roundtrip_hdf5(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a var_acc with real scalar sample type.
TEST(SerializerHdf5, VarAccScalar) {
    simplemc::var_acc<double> a;
    a << 1.0 << 2.0 << 3.0 << 4.0 << 5.0;
    simplemc::var_acc<double> b;
    roundtrip_hdf5(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a var_acc with real vector sample type.
TEST(SerializerHdf5, VarAccVector) {
    using vec_t = Eigen::Vector3d;
    simplemc::var_acc<vec_t> a;
    a << vec_t { 1.0, 2.0, 3.0 } << vec_t { 4.0, 5.0, 6.0 } << vec_t { -1.0, 0.5, 2.0 };
    simplemc::var_acc<vec_t> b;
    roundtrip_hdf5(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a var_acc with complex scalar sample type.
TEST(SerializerHdf5, VarAccComplexScalar) {
    using cplx = std::complex<double>;
    simplemc::var_acc<cplx> a;
    a << cplx { 1.0, -1.0 } << cplx { 2.0, 0.5 } << cplx { 3.0, 1.5 } << cplx { -1.0, 2.0 };
    simplemc::var_acc<cplx> b;
    roundtrip_hdf5(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a var_acc with complex vector sample type.
TEST(SerializerHdf5, VarAccComplexVector) {
    using vec_t = Eigen::Vector2cd;
    using cplx = std::complex<double>;
    simplemc::var_acc<vec_t> a;
    a << vec_t { cplx { 1.0, 2.0 }, cplx { -1.0, 0.5 } };
    a << vec_t { cplx { 3.0, -2.0 }, cplx { 2.0, 1.0 } };
    a << vec_t { cplx { 0.5, 0.5 }, cplx { 1.5, -0.5 } };
    simplemc::var_acc<vec_t> b;
    roundtrip_hdf5(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a covar_acc with real scalar sample type.
TEST(SerializerHdf5, CovarAccScalar) {
    simplemc::covar_acc<double> a;
    a << 1.0 << 2.0 << 3.0 << 4.0;
    simplemc::covar_acc<double> b;
    roundtrip_hdf5(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a covar_acc with real vector sample type.
TEST(SerializerHdf5, CovarAccVector) {
    using vec_t = Eigen::Vector2d;
    simplemc::covar_acc<vec_t> a;
    a << vec_t { 1.0, 2.0 } << vec_t { 4.0, 5.0 } << vec_t { -1.0, 0.5 };
    simplemc::covar_acc<vec_t> b;
    roundtrip_hdf5(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a batch_acc with real scalar sample type.
TEST(SerializerHdf5, BatchAccScalar) {
    simplemc::batch_acc<double> a { 1, 8 };
    for (int i = 0; i < 50; ++i) {
        a << static_cast<double>(i);
    }
    simplemc::batch_acc<double> b { 1, 8 };
    roundtrip_hdf5(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a block_acc wrapping a var_acc with real scalar sample type.
TEST(SerializerHdf5, BlockAccVar) {
    simplemc::block_acc<simplemc::var_acc<double>> a { 4 };
    for (int i = 0; i < 17; ++i) {
        a << static_cast<double>(i);
    }
    simplemc::block_acc<simplemc::var_acc<double>> b { 4 };
    roundtrip_hdf5(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing an autocorr_acc wrapping a var_acc with real scalar sample type.
TEST(SerializerHdf5, AutocorrAccVar) {
    simplemc::autocorr_acc<simplemc::var_acc<double>> a;
    for (int i = 0; i < 1000; ++i) {
        a << static_cast<double>(i);
    }
    simplemc::autocorr_acc<simplemc::var_acc<double>> b;
    roundtrip_hdf5(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a linear_grid.
TEST(SerializerHdf5, LinearGrid) {
    simplemc::linear_grid a { 0.0, 10.0, 11 };
    simplemc::linear_grid b;
    roundtrip_hdf5(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a power_grid.
TEST(SerializerHdf5, PowerGrid) {
    simplemc::power_grid a { 0.0, 1.0, 11, 2.0 };
    simplemc::power_grid b;
    roundtrip_hdf5(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a custom_grid.
TEST(SerializerHdf5, CustomGrid) {
    simplemc::custom_grid a { std::vector<double> { 0.0, 0.1, 0.5, 1.7, 3.14, 10.0 } };
    simplemc::custom_grid b;
    roundtrip_hdf5(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a symmetric_power_grid.
TEST(SerializerHdf5, SymmetricPowerGrid) {
    simplemc::symmetric_power_grid a { 0.0, 1.0, 9, 2.0 };
    simplemc::symmetric_power_grid b;
    roundtrip_hdf5(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing an nd_grid.
TEST(SerializerHdf5, NdGrid) {
    simplemc::nd_grid<simplemc::linear_grid, simplemc::power_grid> a {
        simplemc::linear_grid { 0.0, 1.0, 5 },
        simplemc::power_grid { 0.0, 10.0, 11, 2.0 },
    };
    simplemc::nd_grid<simplemc::linear_grid, simplemc::power_grid> b;
    roundtrip_hdf5(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a bravais_lattice.
TEST(SerializerHdf5, BravaisLattice2D) {
    Eigen::Matrix<double, 2, 2> A;
    A << 1.0, 0.0, 0.0, 2.0;
    simplemc::bravais_lattice<2> a { A };
    simplemc::bravais_lattice<2> b;
    roundtrip_hdf5(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a xoshiro256p.
TEST(SerializerHdf5, Xoshiro256Plus) {
    simplemc::xoshiro256p a { 42ULL };
    for (int i = 0; i < 100; ++i) {
        (void)a();
    }
    simplemc::xoshiro256p b { 0ULL };
    roundtrip_hdf5(a, b);
    EXPECT_EQ(a, b);
    // sanity: equal state ⇒ equal continued sequence
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(a(), b());
    }
}

// Test serializing/deserializing a splitmix64.
TEST(SerializerHdf5, Splitmix64) {
    simplemc::splitmix64 a { 12345ULL };
    for (int i = 0; i < 100; ++i) {
        (void)a();
    }
    simplemc::splitmix64 b { 0ULL };
    roundtrip_hdf5(a, b);
    EXPECT_EQ(a, b);
}

// Test append mode: write part of a tree, then reopen with append and add to it.
TEST(SerializerHdf5, WriteAppendPreservesExistingContent) {
    const auto path = temp_h5_path();
    {
        simplemc::hdf5_serializer s { path, simplemc::hdf5_file_mode::truncate };
        s.save_at("first", 42);
    }
    {
        simplemc::hdf5_serializer s { path, simplemc::hdf5_file_mode::append };
        s.save_at("second", 7.5);
    }

    const simplemc::hdf5_serializer d { path, simplemc::hdf5_file_mode::read };
    int first = 0;
    double second = 0.0;
    d.load_at("first", first);
    d.load_at("second", second);
    EXPECT_EQ(first, 42);
    EXPECT_DOUBLE_EQ(second, 7.5);
    std::filesystem::remove(path);
}

// Test that opening a missing file in read mode throws.
TEST(SerializerHdf5, OpeningMissingFileThrows) {
    const auto missing = std::filesystem::temp_directory_path() / "simplemc_h5_nonexistent_file.h5";
    std::filesystem::remove(missing);
    EXPECT_THROW((simplemc::hdf5_serializer { missing, simplemc::hdf5_file_mode::read }), simplemc::simplemc_exception);
}
