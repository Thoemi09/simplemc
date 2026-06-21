#include <simplemc/serialize.hpp>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

using namespace simplemc;

namespace {

// A second, distinct serializer backend for the variant tests. It is a transparent pass-through over
// json_serializer: a different type (so variant_serializer<json_serializer, mock_serializer> has two
// genuinely distinct alternatives) that forwards every operation to the wrapped handle.
class mock_serializer {
public:
    explicit mock_serializer(nlohmann::json doc = {}) : inner_ { std::move(doc) } {}
    explicit mock_serializer(json_serializer s) : inner_ { std::move(s) } {}

    template <typename T>
    mock_serializer save_at(std::string_view key, const T& value) {
        inner_.save_at(key, value);
        return *this;
    }

    template <typename T>
    mock_serializer load_at(std::string_view key, T& value) const {
        inner_.load_at(key, value);
        return *this;
    }

    template <typename T>
    bool try_load_at(std::string_view key, T& value) const {
        return inner_.try_load_at(key, value);
    }

    mock_serializer operator[](std::string_view key) { return mock_serializer { inner_[key] }; }
    mock_serializer operator[](std::string_view key) const { return mock_serializer { inner_[key] }; }

    [[nodiscard]] bool has(std::string_view key) const { return inner_.has(key); }

    nlohmann::json& root() noexcept { return inner_.root(); }
    [[nodiscard]] const nlohmann::json& root() const noexcept { return inner_.root(); }

private:
    json_serializer inner_;
};

static_assert(serializer<mock_serializer>);

using var_serializer = variant_serializer<json_serializer, mock_serializer>;

// A type whose serialization is written against the variant handle itself (the pattern the
// simplemc-mc value types use). It exercises the variant-level ADL dispatch branch of save_at /
// load_at / try_load_at, rather than per-backend delegation.
struct nested_thing {
    int a = 0;
    double b = 0.0;

    friend void simplemc_save(var_serializer& s, const nested_thing& v) { // NOLINT
        s.save_at("a", v.a);
        s.save_at("b", v.b);
    }
    friend void simplemc_load(const var_serializer& s, nested_thing& v) { // NOLINT
        s.load_at("a", v.a);
        s.load_at("b", v.b);
    }
};

} // namespace

// The variant serializer models the serializer concept.
TEST(VariantSerializer, ConceptConformance) {
    static_assert(serializer<var_serializer>);
    static_assert(!std::is_same_v<json_serializer, mock_serializer>);
    SUCCEED();
}

// Default-construct holds the first backend (json_serializer).
TEST(VariantSerializer, DefaultHoldsFirstBackend) {
    var_serializer s;
    EXPECT_EQ(s.backend().index(), 0u);
    EXPECT_NE(std::get_if<json_serializer>(&s.backend()), nullptr);
    EXPECT_EQ(std::get_if<mock_serializer>(&s.backend()), nullptr);
}

// Round-trip a scalar through the JSON backend and inspect the underlying document.
TEST(VariantSerializer, JsonBackendRoundTrip) {
    var_serializer s { json_serializer {} };
    s.save_at("x", 42);
    s.save_at("y", 1.5);

    auto* js = std::get_if<json_serializer>(&s.backend());
    ASSERT_NE(js, nullptr);
    EXPECT_EQ(js->root().dump(), R"({"x":42,"y":1.5})");

    const var_serializer d { json_serializer { std::move(js->root()) } };
    int x = 0;
    double y = 0.0;
    d.load_at("x", x);
    d.load_at("y", y);
    EXPECT_EQ(x, 42);
    EXPECT_DOUBLE_EQ(y, 1.5);
}

// The active backend can be chosen at runtime; the mock backend round-trips identically.
TEST(VariantSerializer, MockBackendRoundTrip) {
    var_serializer s { mock_serializer {} };
    EXPECT_EQ(s.backend().index(), 1u);
    s.save_at("n", 7);

    auto* ms = std::get_if<mock_serializer>(&s.backend());
    ASSERT_NE(ms, nullptr);
    EXPECT_EQ(ms->root().dump(), R"({"n":7})");

    const var_serializer d { mock_serializer { std::move(ms->root()) } };
    int n = 0;
    d.load_at("n", n);
    EXPECT_EQ(n, 7);
}

// Nested navigation through operator[] writes and reads at the deeper location.
TEST(VariantSerializer, NestedOperatorIndex) {
    var_serializer s { json_serializer {} };
    s["group"].save_at("value", 3.25);

    auto* js = std::get_if<json_serializer>(&s.backend());
    ASSERT_NE(js, nullptr);
    EXPECT_EQ(js->root().dump(), R"({"group":{"value":3.25}})");

    const var_serializer d { json_serializer { std::move(js->root()) } };
    double v = 0.0;
    d["group"].load_at("value", v);
    EXPECT_DOUBLE_EQ(v, 3.25);
}

// try_load_at on a missing key returns false and leaves the target untouched.
TEST(VariantSerializer, TryLoadMissingKey) {
    var_serializer s { json_serializer {} };
    s.save_at("present", 10);

    auto* js = std::get_if<json_serializer>(&s.backend());
    const var_serializer d { json_serializer { std::move(js->root()) } };
    int loaded = -1;
    EXPECT_TRUE(d.try_load_at("present", loaded));
    EXPECT_EQ(loaded, 10);

    int untouched = 99;
    EXPECT_FALSE(d.try_load_at("absent", untouched));
    EXPECT_EQ(untouched, 99);
}

// A type with a variant-level simplemc_save/load dispatches through the variant's own ADL branch:
// save_at nests the fields under the key, and load_at / try_load_at round-trip it.
TEST(VariantSerializer, VariantLevelAdlDispatch) {
    var_serializer s { json_serializer {} };
    s.save_at("thing", nested_thing { 7, 2.5 });

    auto* js = std::get_if<json_serializer>(&s.backend());
    ASSERT_NE(js, nullptr);
    EXPECT_EQ(js->root().dump(), R"({"thing":{"a":7,"b":2.5}})");

    const var_serializer d { json_serializer { std::move(js->root()) } };
    nested_thing loaded;
    d.load_at("thing", loaded);
    EXPECT_EQ(loaded.a, 7);
    EXPECT_DOUBLE_EQ(loaded.b, 2.5);

    nested_thing via_try;
    EXPECT_TRUE(d.try_load_at("thing", via_try));
    EXPECT_EQ(via_try.a, 7);
    EXPECT_DOUBLE_EQ(via_try.b, 2.5);

    nested_thing untouched { -1, -1.0 };
    EXPECT_FALSE(d.try_load_at("absent", untouched));
    EXPECT_EQ(untouched.a, -1);
    EXPECT_DOUBLE_EQ(untouched.b, -1.0);
}

// has() reflects presence through the active backend.
TEST(VariantSerializer, HasReportsPresence) {
    var_serializer s { json_serializer {} };
    s.save_at("a", 1);

    auto* js = std::get_if<json_serializer>(&s.backend());
    const var_serializer d { json_serializer { std::move(js->root()) } };
    EXPECT_TRUE(d.has("a"));
    EXPECT_FALSE(d.has("b"));
}

// std::visit over backend() applies a visitor to whichever backend is active.
TEST(VariantSerializer, VisitDispatchesToActiveBackend) {
    var_serializer s { mock_serializer {} };
    s.save_at("k", 5);
    const std::string dump = std::visit([](auto& backend) { return backend.root().dump(); }, s.backend());
    EXPECT_EQ(dump, R"({"k":5})");
}
