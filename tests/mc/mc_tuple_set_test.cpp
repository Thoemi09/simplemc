#include <simplemc/mc/tuple_set.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

using namespace simplemc;

namespace {

struct int_entry {
    using value_type = int;
    int value_ = 0;
    std::string name_;
    [[nodiscard]] int& value() noexcept { return value_; }
    [[nodiscard]] const int& value() const noexcept { return value_; }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }
};

struct string_entry {
    using value_type = std::string;
    std::string value_;
    std::string name_;
    [[nodiscard]] std::string& value() noexcept { return value_; }
    [[nodiscard]] const std::string& value() const noexcept { return value_; }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }
};

} // namespace

// Test size of an empty and non-empty set.
TEST(MCTupleSet, SizeAndEmpty) {
    // empty set of size 0
    tuple_set<> empty_set {};
    static_assert(empty_set.size() == 0u);
    static_assert(empty_set.empty());

    // set of size 2
    tuple_set set { int_entry { .value_ = 1, .name_ = "a" }, string_entry { .value_ = "x", .name_ = "b" } };
    EXPECT_EQ(set.size(), 2u);
    EXPECT_FALSE(set.empty());
}

// Test that the constructor rejects duplicate entry names.
TEST(MCTupleSet, ConstructorThrowsOnDuplicateNames) {
    EXPECT_THROW((tuple_set { int_entry { .value_ = 1, .name_ = "a" }, string_entry { .value_ = "x", .name_ = "a" } }),
        simplemc_exception);
    EXPECT_THROW((tuple_set { int_entry { .value_ = 1, .name_ = "a" }, string_entry { .value_ = "x", .name_ = "b" },
                     int_entry { .value_ = 2, .name_ = "b" } }),
        simplemc_exception);
}

// Test access to entries by compile-time index.
TEST(MCTupleSet, GetCompileTimeIndex) {
    tuple_set set { int_entry { .value_ = 7, .name_ = "a" }, string_entry { .value_ = "hi", .name_ = "b" } };
    EXPECT_EQ(set.get<0>().value(), 7);
    EXPECT_EQ(set.get<1>().value(), "hi");

    set.get<0>().value() = 9;
    const auto& cset = set;
    EXPECT_EQ(cset.get<0>().value(), 9);
}

// Test for_each() functionalities.
TEST(MCTupleSet, ForEachVisitsAllInOrder) {
    tuple_set set { int_entry { .value_ = 1, .name_ = "a" }, string_entry { .value_ = "x", .name_ = "b" } };
    std::vector<std::string> visited;
    set.for_each([&](const auto& e) { visited.push_back(e.name()); });
    EXPECT_EQ(visited, (std::vector<std::string> { "a", "b" }));
}

TEST(MCTupleSet, ForEachMutates) {
    tuple_set set { int_entry { .value_ = 1, .name_ = "a" }, int_entry { .value_ = 2, .name_ = "b" } };
    set.for_each([](auto& e) { e.value() += 10; });
    EXPECT_EQ(set.get<0>().value(), 11);
    EXPECT_EQ(set.get<1>().value(), 12);
}

// Test visit_at() functionalities.
TEST(MCTupleSet, VisitAtDispatchesToSingleEntry) {
    tuple_set set { int_entry { .value_ = 1, .name_ = "a" }, string_entry { .value_ = "x", .name_ = "b" } };

    int visits = 0;
    set.visit_at(1, [&](auto& e) {
        ++visits;
        EXPECT_EQ(e.name(), "b");
    });
    EXPECT_EQ(visits, 1);

    const auto& cset = set;
    std::string seen;
    cset.visit_at(0, [&](const auto& e) { seen = e.name(); });
    EXPECT_EQ(seen, "a");
}

TEST(MCTupleSet, VisitAtMutatesThroughReference) {
    tuple_set set { int_entry { .value_ = 1, .name_ = "a" }, int_entry { .value_ = 2, .name_ = "b" } };
    set.visit_at(1, [](auto& e) { e.value() = 42; });
    EXPECT_EQ(set.get<0>().value(), 1);
    EXPECT_EQ(set.get<1>().value(), 42);
}

TEST(MCTupleSet, VisitAtByNameDispatchesToSingleEntry) {
    tuple_set set { int_entry { .value_ = 1, .name_ = "a" }, string_entry { .value_ = "x", .name_ = "b" } };

    int visits = 0;
    set.visit_at("b", [&](auto& e) {
        ++visits;
        EXPECT_EQ(e.name(), "b");
    });
    EXPECT_EQ(visits, 1);

    const auto& cset = set;
    std::string seen;
    cset.visit_at("a", [&](const auto& e) { seen = e.name(); });
    EXPECT_EQ(seen, "a");
}

TEST(MCTupleSet, VisitAtByNameMutatesThroughReference) {
    tuple_set set { int_entry { .value_ = 1, .name_ = "a" }, int_entry { .value_ = 2, .name_ = "b" } };
    set.visit_at("b", [](auto& e) { e.value() = 42; });
    EXPECT_EQ(set.get<0>().value(), 1);
    EXPECT_EQ(set.get<1>().value(), 42);
}

TEST(MCTupleSet, VisitAtByNameThrowsOnUnknownName) {
    tuple_set set { int_entry { .value_ = 1, .name_ = "a" }, string_entry { .value_ = "x", .name_ = "b" } };
    EXPECT_THROW(set.visit_at("missing", [](auto&) {}), simplemc_exception);

    const auto& cset = set;
    EXPECT_THROW(cset.visit_at("missing", [](const auto&) {}), simplemc_exception);

    // on an empty set every name is unknown
    tuple_set<> empty_set {};
    EXPECT_THROW(empty_set.visit_at("a", [](auto&) {}), simplemc_exception);
}

// Test finding entries by name.
TEST(MCTupleSet, FindReturnsMatchOrNullopt) {
    tuple_set set { int_entry { .value_ = 1, .name_ = "a" }, string_entry { .value_ = "x", .name_ = "b" } };
    EXPECT_EQ(set.find("a"), std::optional<std::size_t> { 0 });
    EXPECT_EQ(set.find("b"), std::optional<std::size_t> { 1 });
    EXPECT_FALSE(set.find("missing").has_value());
}

TEST(MCTupleSet, GetByNameChecksNameAndType) {
    tuple_set set { int_entry { .value_ = 5, .name_ = "a" }, string_entry { .value_ = "x", .name_ = "b" } };

    EXPECT_NE(set.get<int>("a"), nullptr);
    EXPECT_EQ(set.get<std::string>("a"), nullptr); // type mismatch
    EXPECT_EQ(set.get<int>("missing"), nullptr); // unknown name

    const auto& cset = set;
    ASSERT_NE(cset.get<std::string>("b"), nullptr);
    EXPECT_EQ(*cset.get<std::string>("b"), "x");
}
