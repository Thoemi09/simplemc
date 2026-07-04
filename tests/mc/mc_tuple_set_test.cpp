#include <simplemc/mc/tuple_set.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

using namespace simplemc;

namespace {

struct int_entry {
    using value_type = int;
    int value = 0;
    std::string name;
};

struct string_entry {
    using value_type = std::string;
    std::string value;
    std::string name;
};

} // namespace

// Test size of an empty and non-empty set.
TEST(MCTupleSet, SizeAndEmpty) {
    // empty set of size 0
    tuple_set<> empty_set {};
    static_assert(empty_set.size() == 0u);
    static_assert(empty_set.empty());

    // set of size 2
    tuple_set set { int_entry { .value = 1, .name = "a" }, string_entry { .value = "x", .name = "b" } };
    EXPECT_EQ(set.size(), 2u);
    EXPECT_FALSE(set.empty());
}

// Test access to entries by compile-time index.
TEST(MCTupleSet, GetCompileTimeIndex) {
    tuple_set set { int_entry { .value = 7, .name = "a" }, string_entry { .value = "hi", .name = "b" } };
    EXPECT_EQ(set.get<0>().value, 7);
    EXPECT_EQ(set.get<1>().value, "hi");

    set.get<0>().value = 9;
    const auto& cset = set;
    EXPECT_EQ(cset.get<0>().value, 9);
}

// Test for_each() functionalities.
TEST(MCTupleSet, ForEachVisitsAllInOrder) {
    tuple_set set { int_entry { .value = 1, .name = "a" }, string_entry { .value = "x", .name = "b" } };
    std::vector<std::string> visited;
    set.for_each([&](const auto& e) { visited.push_back(e.name); });
    EXPECT_EQ(visited, (std::vector<std::string> { "a", "b" }));
}

TEST(MCTupleSet, ForEachMutates) {
    tuple_set set { int_entry { .value = 1, .name = "a" }, int_entry { .value = 2, .name = "b" } };
    set.for_each([](auto& e) { e.value += 10; });
    EXPECT_EQ(set.get<0>().value, 11);
    EXPECT_EQ(set.get<1>().value, 12);
}

// Test visit_at() functionalities.
TEST(MCTupleSet, VisitAtDispatchesToSingleEntry) {
    tuple_set set { int_entry { .value = 1, .name = "a" }, string_entry { .value = "x", .name = "b" } };

    int visits = 0;
    set.visit_at(1, [&](auto& e) {
        ++visits;
        EXPECT_EQ(e.name, "b");
    });
    EXPECT_EQ(visits, 1);

    const auto& cset = set;
    std::string seen;
    cset.visit_at(0, [&](const auto& e) { seen = e.name; });
    EXPECT_EQ(seen, "a");
}

TEST(MCTupleSet, VisitAtMutatesThroughReference) {
    tuple_set set { int_entry { .value = 1, .name = "a" }, int_entry { .value = 2, .name = "b" } };
    set.visit_at(1, [](auto& e) { e.value = 42; });
    EXPECT_EQ(set.get<0>().value, 1);
    EXPECT_EQ(set.get<1>().value, 42);
}

TEST(MCTupleSet, VisitAtOutOfRangeInvokesNothing) {
    tuple_set set { int_entry { .value = 1, .name = "a" }, string_entry { .value = "x", .name = "b" } };

    int visits = 0;
    set.visit_at(2, [&](auto&) { ++visits; });
    set.visit_at(1'000'000, [&](auto&) { ++visits; });
    EXPECT_EQ(visits, 0);

    const auto& cset = set;
    cset.visit_at(2, [&](const auto&) { ++visits; });
    EXPECT_EQ(visits, 0);

    // on an empty set every index is out of range
    tuple_set<> empty_set {};
    empty_set.visit_at(0, [](auto&) { ADD_FAILURE() << "callable must not be invoked"; });
}

// Test finding entries by name.
TEST(MCTupleSet, FindReturnsFirstMatchOrNullopt) {
    tuple_set set { int_entry { .value = 1, .name = "a" }, string_entry { .value = "x", .name = "b" } };
    EXPECT_EQ(set.find("a"), std::optional<std::size_t> { 0 });
    EXPECT_EQ(set.find("b"), std::optional<std::size_t> { 1 });
    EXPECT_FALSE(set.find("missing").has_value());
}

// Test accessing entries by type via get().
TEST(MCTupleSet, GetByTypeReturnsPayloadPointer) {
    tuple_set set { int_entry { .value = 5, .name = "a" }, string_entry { .value = "x", .name = "b" } };

    int* p = set.get<int>();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(*p, 5);
    *p = 6;
    EXPECT_EQ(set.get<0>().value, 6);

    const auto& cset = set;
    const auto* q = cset.get<std::string>();
    ASSERT_NE(q, nullptr);
    EXPECT_EQ(*q, "x");
}

TEST(MCTupleSet, GetByNameChecksNameAndType) {
    tuple_set set { int_entry { .value = 5, .name = "a" }, string_entry { .value = "x", .name = "b" } };

    EXPECT_NE(set.get<int>("a"), nullptr);
    EXPECT_EQ(set.get<std::string>("a"), nullptr); // type mismatch
    EXPECT_EQ(set.get<int>("missing"), nullptr); // unknown name

    const auto& cset = set;
    ASSERT_NE(cset.get<std::string>("b"), nullptr);
    EXPECT_EQ(*cset.get<std::string>("b"), "x");
}
