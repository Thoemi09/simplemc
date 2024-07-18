#include <gtest/gtest.h>

#include <simplemc/utils/indexing.hpp>

#include <array>
#include <vector>

// Test calculating the size from a shape.
TEST(SimplemcUtils, SizeFromShape) {
    using namespace simplemc;
    auto shape = std::vector<int> { 2, 3, 4 };
    ASSERT_EQ(size_from_shape(shape), 24);
    constexpr std::array<int, 2> shape_arr_cepr { 2, 3 };
    constexpr auto size_cepr = size_from_shape(shape_arr_cepr);
    ASSERT_EQ(size_cepr, 6);
    std::array<long, 2> shape_arr { 2, 3 };
    auto size = size_from_shape(shape_arr);
    ASSERT_EQ(size, 6);
}

// Test constexpr indexing with arrays.
TEST(SimplemcUtils, ConstexprIndexing) {
    using namespace simplemc;
    constexpr std::array<int, 3> shape { 2, 3, 4 };
    constexpr auto size = size_from_shape(shape);
    ASSERT_EQ(size, 24);
    constexpr std::array<int, 3> exp_idxs1 { 0, 0, 0 };
    constexpr int exp_fidx1 = 0;
    constexpr auto cfidx1 = flat_index(exp_idxs1, shape);
    constexpr auto cidxs1 = multi_index(exp_fidx1, shape);
    constexpr auto rfidx1 = flat_index(exp_idxs1, shape, row_major {});
    constexpr auto ridxs1 = multi_index(exp_fidx1, shape, row_major {});
    ASSERT_EQ(cfidx1, exp_fidx1);
    ASSERT_EQ(cidxs1, exp_idxs1);
    ASSERT_EQ(rfidx1, exp_fidx1);
    ASSERT_EQ(ridxs1, exp_idxs1);
    constexpr std::array<int, 3> exp_idxs2 { 1, 2, 3 };
    constexpr int exp_fidx2 = 23;
    constexpr auto cfidx2 = flat_index(exp_idxs2, shape);
    constexpr auto cidxs2 = multi_index(exp_fidx2, shape);
    constexpr auto rfidx2 = flat_index(exp_idxs2, shape, row_major {});
    constexpr auto ridxs2 = multi_index(exp_fidx2, shape, row_major {});
    ASSERT_EQ(cfidx2, exp_fidx2);
    ASSERT_EQ(cidxs2, exp_idxs2);
    ASSERT_EQ(rfidx2, exp_fidx2);
    ASSERT_EQ(ridxs2, exp_idxs2);
    constexpr std::array<int, 3> exp_idxs3 { 1, 1, 1 };
    constexpr int exp_cfidx3 = 9;
    constexpr int exp_rfidx3 = 17;
    constexpr auto cfidx3 = flat_index(exp_idxs3, shape);
    constexpr auto cidxs3 = multi_index(exp_cfidx3, shape);
    constexpr auto rfidx3 = flat_index(exp_idxs3, shape, row_major {});
    constexpr auto ridxs3 = multi_index(exp_rfidx3, shape, row_major {});
    ASSERT_EQ(cfidx3, exp_cfidx3);
    ASSERT_EQ(cidxs3, exp_idxs3);
    ASSERT_EQ(rfidx3, exp_rfidx3);
    ASSERT_EQ(ridxs3, exp_idxs3);
}

// Test indexing with vectors.
TEST(SimplemcUtils, Indexing) {
    using namespace simplemc;
    std::vector<long> shape { 2, 3, 4 };
    auto size = size_from_shape(shape);
    ASSERT_EQ(size, 24);
    std::vector<long> exp_idxs { 0, 0, 0 };
    int exp_fidx = 0;
    ASSERT_EQ(flat_index(exp_idxs, shape), exp_fidx);
    ASSERT_EQ(multi_index(exp_fidx, shape), exp_idxs);
    ASSERT_EQ(flat_index(exp_idxs, shape, row_major {}), exp_fidx);
    ASSERT_EQ(multi_index(exp_fidx, shape, row_major {}), exp_idxs);
    exp_idxs = { 1, 2, 3 };
    exp_fidx = 23;
    ASSERT_EQ(flat_index(exp_idxs, shape), exp_fidx);
    ASSERT_EQ(multi_index(exp_fidx, shape), exp_idxs);
    ASSERT_EQ(flat_index(exp_idxs, shape, row_major {}), exp_fidx);
    ASSERT_EQ(multi_index(exp_fidx, shape, row_major {}), exp_idxs);
    exp_idxs = { 1, 1, 1 };
    exp_fidx = 9;
    ASSERT_EQ(flat_index(exp_idxs, shape), exp_fidx);
    ASSERT_EQ(multi_index(exp_fidx, shape), exp_idxs);
    exp_fidx = 17;
    ASSERT_EQ(flat_index(exp_idxs, shape, row_major {}), exp_fidx);
    ASSERT_EQ(multi_index(exp_fidx, shape, row_major {}), exp_idxs);
}

// Test integer subrange.
TEST(SimplemcUtils, IntegerSubrange) {
    using namespace simplemc;
    ASSERT_EQ(integer_subrange(4, 9, 1), 4);
    ASSERT_EQ(integer_subrange(4, 9, 2), 4);
    ASSERT_EQ(integer_subrange(4, 9, 3), 3);
    ASSERT_EQ(integer_subrange(4, 9, 6), 2);
    ASSERT_EQ(integer_subrange(4, 9, 8), 1);
    ASSERT_EQ(integer_subrange(4, 9, 9), 0);
    ASSERT_EQ(integer_subrange(7, 9, 1), 7);
    ASSERT_EQ(integer_subrange(7, 9, 2), 7);
    ASSERT_EQ(integer_subrange(7, 9, 3), 6);
    ASSERT_EQ(integer_subrange(7, 9, 4), 5);
    ASSERT_EQ(integer_subrange(7, 9, 7), 2);
    ASSERT_EQ(integer_subrange(1, 9, 1), 1);
    ASSERT_EQ(integer_subrange(1, 9, 2), 1);
    ASSERT_EQ(integer_subrange(1, 9, 3), 0);
    ASSERT_EQ(integer_subrange(1, 9, 4), 0);
    ASSERT_EQ(integer_subrange(1, 9, 9), 0);
}
