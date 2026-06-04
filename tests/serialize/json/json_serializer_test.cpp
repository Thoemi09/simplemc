#include "../serialize_test_utils.hpp"

#include <simplemc/random/splitmix64.hpp>
#include <simplemc/random/xoshiro256.hpp>
#include <simplemc/serialize.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <Eigen/Dense>

#include <complex>
#include <utility>
#include <vector>

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

} // namespace test_types

// In-memory save → load round-trip through json_serializer. File IO has its own smoke tests below.
template <class T>
void roundtrip_json(const T& src, T& dst) {
    simplemc::json_serializer s;
    simplemc_save(s, src);
    const simplemc::json_serializer d { std::move(s.root()) };
    simplemc_load(d, dst);
}

// Test serializer/deserializer concept conformance.
TEST(SerializerJson, ConceptConformance) {
    using simplemc::deserializer;
    using simplemc::json_serializer;
    using simplemc::serializer;

    static_assert(serializer<json_serializer>);
    static_assert(deserializer<json_serializer>);
    SUCCEED();
}

// Test ADL dispatch to friend simplemc_save / simplemc_load functions.
TEST(SerializerJson, AdlDispatchFriendPath) {
    using test_types::intrusive_point;
    const intrusive_point a { 1.25, -3.5 };

    simplemc::json_serializer s {};
    s.save_at("p", a);

    ASSERT_EQ(s.root().dump(), R"({"p":{"x":1.25,"y":-3.5}})");

    const simplemc::json_serializer d { std::move(s.root()) };
    intrusive_point b;
    d.load_at("p", b);
    ASSERT_EQ(b, a);
}

// Test ADL dispatch to namespace-scope simplemc_save / simplemc_load functions.
TEST(SerializerJson, AdlDispatchFreeFunctionPath) {
    using test_types::nonintrusive_box;
    const nonintrusive_box a { .width = 4, .height = 5 };
    nonintrusive_box b;
    roundtrip_json(a, b);
    ASSERT_EQ(b, a);
}

// Test ADL dispatch to a composite type.
TEST(SerializerJson, AdlDispatchComposite) {
    using namespace test_types;
    const composite a { .pt = intrusive_point { 1.0, 2.0 },
        .bx = nonintrusive_box { .width = 10, .height = 20 },
        .data = { 0.5, 1.5, 2.5 } };
    composite b;
    roundtrip_json(a, b);
    ASSERT_EQ(b, a);
}

// Test operator[] navigation and sub-tree building.
TEST(SerializerJson, OperatorBracketBuildsSubTree) {
    simplemc::json_serializer s {};
    s.save_at("step", 1);
    s["results"].save_at("mean", 0.5);
    s["results"].save_at("var", 0.1);

    const auto& j = s.root();
    ASSERT_EQ(j["step"].get<int>(), 1);
    ASSERT_DOUBLE_EQ(j["results"]["mean"].get<double>(), 0.5);
    ASSERT_DOUBLE_EQ(j["results"]["var"].get<double>(), 0.1);
}

// Test chaining of operator[] calls.
TEST(SerializerJson, OperatorBracketChainedNavigation) {
    simplemc::json_serializer s {};
    s["a"]["b"]["c"].save_at("x", 42);

    const auto& j = s.root();
    ASSERT_EQ(j["a"]["b"]["c"]["x"].get<int>(), 42);
}

// Test that sub-handler returned by operator[] shares the same document root as the parent.
TEST(SerializerJson, OperatorBracketSharesTree) {
    simplemc::json_serializer s {};
    auto results = s["results"];
    results.save_at("mean", 0.5);
    results.save_at("var", 0.1);

    // s sees the changes because the tree is shared
    const auto& j = s.root();
    ASSERT_DOUBLE_EQ(j["results"]["mean"].get<double>(), 0.5);
    ASSERT_DOUBLE_EQ(j["results"]["var"].get<double>(), 0.1);
}

// Test that operator[] on the read side throws if the key is missing.
TEST(SerializerJson, OperatorBracketReadSideThrowsOnMissing) {
    nlohmann::json j;
    j["existing"] = 1;
    const simplemc::json_serializer d { std::move(j) };
    EXPECT_THROW({ auto sub = d["missing"]; }, simplemc::simplemc_exception);
}

// Test the JSON pointer tracking of operator[] calls.
TEST(SerializerJson, JsonPtrTracksPosition) {
    using ptr = nlohmann::json::json_pointer;
    simplemc::json_serializer s {};
    EXPECT_EQ(s.json_ptr(), ptr {});
    auto sub = s["a"]["b"];
    EXPECT_EQ(sub.json_ptr(), ptr { "/a/b" });

    nlohmann::json j;
    j["a"]["b"] = 42;
    const simplemc::json_serializer d { std::move(j) };
    EXPECT_EQ(d.json_ptr(), ptr {});
    EXPECT_EQ(d["a"]["b"].json_ptr(), ptr { "/a/b" });
}

// Test presence of keys via has().
TEST(SerializerJson, HasKey) {
    nlohmann::json j;
    j["present"] = 1;
    j["list"] = { 10, 20, 30 };
    const simplemc::json_serializer d { std::move(j) };

    EXPECT_TRUE(d.has("present"));
    EXPECT_FALSE(d.has("missing"));
    EXPECT_EQ(d["list"].get().size(), 3u);
}

// Test serializing/deserializing a mean_acc with real scalar sample type.
TEST(SerializerJson, MeanAccScalar) {
    simplemc::mean_acc<double> a;
    a << 1.0 << 2.0 << 3.0 << 4.0;
    simplemc::mean_acc<double> b;
    roundtrip_json(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a mean_acc with real vector sample type.
TEST(SerializerJson, MeanAccVector) {
    using vec_t = Eigen::Vector3d;
    simplemc::mean_acc<vec_t> a;
    a << vec_t { 1.0, 2.0, 3.0 } << vec_t { 4.0, 5.0, 6.0 };
    simplemc::mean_acc<vec_t> b;
    roundtrip_json(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a var_acc with real scalar sample type.
TEST(SerializerJson, VarAccScalar) {
    simplemc::var_acc<double> a;
    a << 1.0 << 2.0 << 3.0 << 4.0 << 5.0;
    simplemc::var_acc<double> b;
    roundtrip_json(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a var_acc with real vector sample type.
TEST(SerializerJson, VarAccVector) {
    using vec_t = Eigen::Vector3d;
    simplemc::var_acc<vec_t> a;
    a << vec_t { 1.0, 2.0, 3.0 } << vec_t { 4.0, 5.0, 6.0 } << vec_t { -1.0, 0.5, 2.0 };
    simplemc::var_acc<vec_t> b;
    roundtrip_json(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a var_acc with complex scalar sample type.
TEST(SerializerJson, VarAccComplexScalar) {
    using cplx = std::complex<double>;
    simplemc::var_acc<cplx> a;
    a << cplx { 1.0, -1.0 } << cplx { 2.0, 0.5 } << cplx { 3.0, 1.5 } << cplx { -1.0, 2.0 };
    simplemc::var_acc<cplx> b;
    roundtrip_json(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a var_acc with complex vector sample type.
TEST(SerializerJson, VarAccComplexVector) {
    using vec_t = Eigen::Vector2cd;
    using cplx = std::complex<double>;
    simplemc::var_acc<vec_t> a;
    a << vec_t { cplx { 1.0, 2.0 }, cplx { -1.0, 0.5 } };
    a << vec_t { cplx { 3.0, -2.0 }, cplx { 2.0, 1.0 } };
    a << vec_t { cplx { 0.5, 0.5 }, cplx { 1.5, -0.5 } };
    simplemc::var_acc<vec_t> b;
    roundtrip_json(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a covar_acc with real scalar sample type.
TEST(SerializerJson, CovarAccScalar) {
    simplemc::covar_acc<double> a;
    a << 1.0 << 2.0 << 3.0 << 4.0;
    simplemc::covar_acc<double> b;
    roundtrip_json(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a covar_acc with real vector sample type.
TEST(SerializerJson, CovarAccVector) {
    using vec_t = Eigen::Vector2d;
    simplemc::covar_acc<vec_t> a;
    a << vec_t { 1.0, 2.0 } << vec_t { 4.0, 5.0 } << vec_t { -1.0, 0.5 };
    simplemc::covar_acc<vec_t> b;
    roundtrip_json(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a covar_acc with complex scalar sample type.
TEST(SerializerJson, CovarAccComplexScalar) {
    using cplx = std::complex<double>;
    simplemc::covar_acc<cplx> a;
    a << cplx { 1.0, -1.0 } << cplx { 2.0, 0.5 } << cplx { 3.0, 1.5 } << cplx { -1.0, 2.0 };
    simplemc::covar_acc<cplx> b;
    roundtrip_json(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a covar_acc with complex vector sample type.
TEST(SerializerJson, CovarAccComplexVector) {
    using vec_t = Eigen::Vector2cd;
    using cplx = std::complex<double>;
    simplemc::covar_acc<vec_t> a;
    a << vec_t { cplx { 1.0, 2.0 }, cplx { -1.0, 0.5 } };
    a << vec_t { cplx { 3.0, -2.0 }, cplx { 2.0, 1.0 } };
    a << vec_t { cplx { 0.5, 0.5 }, cplx { 1.5, -0.5 } };
    simplemc::covar_acc<vec_t> b;
    roundtrip_json(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a batch_acc with real scalar sample type.
TEST(SerializerJson, BatchAccScalar) {
    simplemc::batch_acc<double> a { 1, 8 };
    for (int i = 0; i < 50; ++i) {
        a << static_cast<double>(i);
    }
    simplemc::batch_acc<double> b { 1, 8 };
    roundtrip_json(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a block_acc wrapping a var_acc with real scalar sample type.
TEST(SerializerJson, BlockAccVar) {
    simplemc::block_acc<simplemc::var_acc<double>> a { 4 };
    for (int i = 0; i < 17; ++i) {
        a << static_cast<double>(i);
    }
    simplemc::block_acc<simplemc::var_acc<double>> b { 4 };
    roundtrip_json(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a block_acc wrapping a covar_acc with real vector sample type.
TEST(SerializerJson, BlockAccCovar) {
    using vec_t = Eigen::Vector2d;
    simplemc::block_acc<simplemc::covar_acc<vec_t>> a { 3, 2 };
    for (int i = 0; i < 13; ++i) {
        a << vec_t { static_cast<double>(i), static_cast<double>(2 * i + 1) };
    }
    simplemc::block_acc<simplemc::covar_acc<vec_t>> b { 3, 2 };
    roundtrip_json(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a block_acc wrapping a var_acc with complex scalar sample type.
TEST(SerializerJson, BlockAccVarComplex) {
    using cplx = std::complex<double>;
    simplemc::block_acc<simplemc::var_acc<cplx>> a { 4 };
    for (int i = 0; i < 17; ++i) {
        a << cplx { static_cast<double>(i), static_cast<double>(-i) };
    }
    simplemc::block_acc<simplemc::var_acc<cplx>> b { 4 };
    roundtrip_json(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a autocorr_acc wrapping a var_acc with real scalar sample type.
TEST(SerializerJson, AutocorrAccVar) {
    simplemc::autocorr_acc<simplemc::var_acc<double>> a;
    for (int i = 0; i < 1000; ++i) {
        a << static_cast<double>(i);
    }
    simplemc::autocorr_acc<simplemc::var_acc<double>> b;
    roundtrip_json(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a autocorr_acc wrapping a covar_acc with real vector sample type.
TEST(SerializerJson, AutocorrAccCovar) {
    using vec_t = Eigen::Vector2d;
    simplemc::autocorr_acc<simplemc::covar_acc<vec_t>> a { 2 };
    for (int i = 0; i < 512; ++i) {
        a << vec_t { static_cast<double>(i), static_cast<double>(0.5 * i) };
    }
    simplemc::autocorr_acc<simplemc::covar_acc<vec_t>> b { 2 };
    roundtrip_json(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a autocorr_acc wrapping a var_acc with complex scalar sample type.
TEST(SerializerJson, AutocorrAccVarComplex) {
    using cplx = std::complex<double>;
    simplemc::autocorr_acc<simplemc::var_acc<cplx>> a;
    for (int i = 0; i < 512; ++i) {
        a << cplx { static_cast<double>(i), static_cast<double>(-i) };
    }
    simplemc::autocorr_acc<simplemc::var_acc<cplx>> b;
    roundtrip_json(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a linear_grid.
TEST(SerializerJson, LinearGrid) {
    simplemc::linear_grid a { 0.0, 10.0, 11 };
    simplemc::linear_grid b;
    roundtrip_json(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a power_grid.
TEST(SerializerJson, PowerGrid) {
    simplemc::power_grid a { 0.0, 1.0, 11, 2.0 };
    simplemc::power_grid b;
    roundtrip_json(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a custom_grid.
TEST(SerializerJson, CustomGrid) {
    simplemc::custom_grid a { std::vector<double> { 0.0, 0.1, 0.5, 1.7, 3.14, 10.0 } };
    simplemc::custom_grid b;
    roundtrip_json(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a symmetric_power_grid.
TEST(SerializerJson, SymmetricPowerGrid) {
    simplemc::symmetric_power_grid a { 0.0, 1.0, 9, 2.0 };
    simplemc::symmetric_power_grid b;
    roundtrip_json(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing an nd_grid.
TEST(SerializerJson, NdGrid) {
    simplemc::nd_grid<simplemc::linear_grid, simplemc::power_grid> a {
        simplemc::linear_grid { 0.0, 1.0, 5 },
        simplemc::power_grid { 0.0, 10.0, 11, 2.0 },
    };
    simplemc::nd_grid<simplemc::linear_grid, simplemc::power_grid> b;
    roundtrip_json(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a bravais_lattice.
TEST(SerializerJson, BravaisLattice2D) {
    Eigen::Matrix<double, 2, 2> A;
    A << 1.0, 0.0, 0.0, 2.0;
    simplemc::bravais_lattice<2> a { A };
    simplemc::bravais_lattice<2> b;
    roundtrip_json(a, b);
    check_state_equal(a, b);
}

// Test serializing/deserializing a xoshiro256p.
TEST(SerializerJson, Xoshiro256Plus) {
    simplemc::xoshiro256p a { 42ULL };
    for (int i = 0; i < 100; ++i) {
        (void)a();
    }
    simplemc::xoshiro256p b { 0ULL };
    roundtrip_json(a, b);
    EXPECT_EQ(a, b);
    // sanity: equal state ⇒ equal continued sequence
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(a(), b());
    }
}

// Test serializing/deserializing a splitmix64.
TEST(SerializerJson, Splitmix64) {
    simplemc::splitmix64 a { 12345ULL };
    for (int i = 0; i < 100; ++i) {
        (void)a();
    }
    simplemc::splitmix64 b { 0ULL };
    roundtrip_json(a, b);
    EXPECT_EQ(a, b);
}

// Test JSON file IO together with JSON serializers.
TEST(SerializerJson, FileIORootWithSimplemcSave) {
    const simplemc::linear_grid g { 0.0, 5.0, 6 };
    simplemc::json_serializer s;
    simplemc_save(s, g);
    simplemc::write_json_file(s.root(), "grid.json");

    nlohmann::json doc;
    simplemc::read_json_file(doc, "grid.json");
    const simplemc::json_serializer d { std::move(doc) };
    simplemc::linear_grid back;
    simplemc_load(d, back);
    check_state_equal(g, back);
}
