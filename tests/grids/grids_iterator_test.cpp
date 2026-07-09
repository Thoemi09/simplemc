#include "../test_utils.hpp"

#include <simplemc/grids/concepts.hpp>
#include <simplemc/grids/grid_iterator.hpp>
#include <simplemc/utils/nd_indexing.hpp>

#include <array>
#include <cstddef>
#include <tuple>

namespace {

// Minimal dummy 1D grid for testing grid_iterator.
//
// This grid represents a simple linear grid with M points from 0.0 to (M-1) * 1.0.
// Grid points: [0.0, 1.0, 2.0, ..., (M-1) * 1.0]
// Bins: [0.0, 1.0), [1.0, 2.0), ..., [(M-2) * 1.0, (M-1) * 1.0]
class dummy_grid_1d {
public:
    using value_type = double;
    using size_type = long;

    constexpr explicit dummy_grid_1d(long size) noexcept : size_(size) {}

    [[nodiscard]] static constexpr std::size_t dim() noexcept { return 1; }

    [[nodiscard]] constexpr long size() const noexcept { return size_; }

    [[nodiscard]] constexpr double first() const noexcept { return 0.0; }

    [[nodiscard]] constexpr double last() const noexcept { return static_cast<double>(size_ - 1); }

    [[nodiscard]] constexpr double at(long i) const noexcept { return static_cast<double>(i); }

    [[nodiscard]] constexpr long index(double x) const noexcept { return static_cast<long>(x); }

    [[nodiscard]] constexpr double volume(long /* i */) const noexcept { return 1.0; }

    [[nodiscard]] constexpr double center(long i) const noexcept { return static_cast<double>(i) + 0.5; }

    [[nodiscard]] constexpr auto begin() const noexcept { return simplemc::grid_iterator<dummy_grid_1d> { *this }; }

    [[nodiscard]] constexpr auto cbegin() const noexcept { return begin(); }

    [[nodiscard]] constexpr auto end() const noexcept {
        return simplemc::grid_iterator<dummy_grid_1d> { *this, size_ };
    }

    [[nodiscard]] constexpr auto cend() const noexcept { return end(); }

private:
    long size_;
};

// Minimal dummy 2D grid for testing grid_iterator.
//
// This grid represents a 2D grid with shape [M0, M1] where each underlying 1D grid is a
// dummy_grid_1d. Grid points are multi-dimensional indices converted to double arrays.
class dummy_grid_2d {
public:
    using value_type = std::array<double, 2>;
    using size_type = std::array<long, 2>;
    using tuple_type = std::tuple<dummy_grid_1d, dummy_grid_1d>;

    constexpr explicit dummy_grid_2d(size_type shape) noexcept : shape_(shape), grid0_(shape[0]), grid1_(shape[1]) {}

    [[nodiscard]] static constexpr std::size_t dim() noexcept { return 2; }

    [[nodiscard]] constexpr long size() const noexcept { return shape_[0] * shape_[1]; }

    [[nodiscard]] constexpr size_type shape() const noexcept { return shape_; }

    [[nodiscard]] constexpr tuple_type grids() const noexcept { return std::make_tuple(grid0_, grid1_); }

    [[nodiscard]] constexpr value_type first() const noexcept { return { grid0_.first(), grid1_.first() }; }

    [[nodiscard]] constexpr value_type last() const noexcept { return { grid0_.last(), grid1_.last() }; }

    [[nodiscard]] constexpr value_type at(size_type idx) const noexcept {
        return { grid0_.at(idx[0]), grid1_.at(idx[1]) };
    }

    [[nodiscard]] constexpr size_type index(value_type x) const noexcept {
        return { grid0_.index(x[0]), grid1_.index(x[1]) };
    }

    [[nodiscard]] constexpr double volume(size_type idx) const noexcept {
        return grid0_.volume(idx[0]) * grid1_.volume(idx[1]);
    }

    [[nodiscard]] constexpr value_type center(size_type idx) const noexcept {
        return { grid0_.center(idx[0]), grid1_.center(idx[1]) };
    }

    [[nodiscard]] constexpr auto begin() const noexcept { return simplemc::grid_iterator<dummy_grid_2d> { *this }; }

    [[nodiscard]] constexpr auto cbegin() const noexcept { return begin(); }

    [[nodiscard]] constexpr auto end() const noexcept {
        return simplemc::grid_iterator<dummy_grid_2d> { *this, size() };
    }

    [[nodiscard]] constexpr auto cend() const noexcept { return end(); }

private:
    size_type shape_;
    dummy_grid_1d grid0_;
    dummy_grid_1d grid1_;
};

} // anonymous namespace

// Verify that dummy grids satisfy the grid_common concept.
static_assert(simplemc::grid_1d<dummy_grid_1d>);
static_assert(simplemc::grid_nd<dummy_grid_2d>);

// Verify that the iterators satisfy the random access iterator concept.
static_assert(std::random_access_iterator<simplemc::grid_iterator<dummy_grid_1d>>);
static_assert(std::random_access_iterator<simplemc::grid_iterator<dummy_grid_2d>>);

// Test default construction and basic properties.
TEST(SimplemcGridIterator, DefaultConstruction) {
    simplemc::grid_iterator<dummy_grid_1d> default_it_1d;
    ASSERT_EQ(default_it_1d.grid_ptr(), nullptr);
    ASSERT_EQ(default_it_1d.flat_index(), 0);

    simplemc::grid_iterator<dummy_grid_2d> default_it_2d;
    ASSERT_EQ(default_it_2d.grid_ptr(), nullptr);
    ASSERT_EQ(default_it_2d.flat_index(), 0);
}

// Test construction and dereference for 1D grids.
TEST(SimplemcGridIterator, Construction1D) {
    dummy_grid_1d grid(5);

    // construction at index 0
    simplemc::grid_iterator<dummy_grid_1d> it_begin(grid);
    ASSERT_EQ(it_begin.grid_ptr(), &grid);
    ASSERT_EQ(it_begin.flat_index(), 0);
    ASSERT_EQ(*it_begin, 0.0);
    ASSERT_EQ(it_begin.nd_index(), 0);

    // construction at specific index
    simplemc::grid_iterator<dummy_grid_1d> it_mid(grid, 2);
    ASSERT_EQ(it_mid.flat_index(), 2);
    ASSERT_EQ(*it_mid, 2.0);
    ASSERT_EQ(it_mid.nd_index(), 2);

    // construction at end
    simplemc::grid_iterator<dummy_grid_1d> it_end(grid, 5);
    ASSERT_EQ(it_end.flat_index(), 5);

    // dereference returns correct values for all indices
    for (long i = 0; i < grid.size(); ++i) {
        simplemc::grid_iterator<dummy_grid_1d> curr_it(grid, i);
        ASSERT_DOUBLE_EQ(*curr_it, grid.at(i));
        ASSERT_EQ(curr_it.nd_index(), i);
    }
}

// Test construction and dereference for 2D grids.
TEST(SimplemcGridIterator, Construction2D) {
    dummy_grid_2d grid({ 3, 4 });

    // construction at index 0
    simplemc::grid_iterator<dummy_grid_2d> it_begin(grid);
    ASSERT_EQ(it_begin.grid_ptr(), &grid);
    ASSERT_EQ(it_begin.flat_index(), 0);
    auto val_begin = *it_begin;
    ASSERT_DOUBLE_EQ(val_begin[0], 0.0);
    ASSERT_DOUBLE_EQ(val_begin[1], 0.0);

    auto nd_idx_begin = it_begin.nd_index();
    ASSERT_EQ(nd_idx_begin[0], 0);
    ASSERT_EQ(nd_idx_begin[1], 0);

    // construction at specific index
    simplemc::grid_iterator<dummy_grid_2d> it_mid(grid, 5);
    ASSERT_EQ(it_mid.flat_index(), 5);
    auto val_mid = *it_mid;
    ASSERT_DOUBLE_EQ(val_mid[0], 1.0);
    ASSERT_DOUBLE_EQ(val_mid[1], 1.0);

    auto nd_idx_mid = it_mid.nd_index();
    ASSERT_EQ(nd_idx_mid[0], 1);
    ASSERT_EQ(nd_idx_mid[1], 1);

    // iteration over all flat indices
    for (long flat_idx = 0; flat_idx < grid.size(); ++flat_idx) {
        simplemc::grid_iterator<dummy_grid_2d> curr_it(grid, flat_idx);
        auto idx_2d = simplemc::nd_index(flat_idx, grid.shape(), simplemc::row_major {});
        auto expected_val = grid.at(idx_2d);
        auto actual_val = *curr_it;

        ASSERT_DOUBLE_EQ(actual_val[0], expected_val[0]);
        ASSERT_DOUBLE_EQ(actual_val[1], expected_val[1]);

        auto nd_idx = curr_it.nd_index();
        ASSERT_EQ(nd_idx[0], idx_2d[0]);
        ASSERT_EQ(nd_idx[1], idx_2d[1]);
    }
}

// Test pre-increment and post-increment operators for 1D grids.
TEST(SimplemcGridIterator, Increment1D) {
    dummy_grid_1d grid(5);
    simplemc::grid_iterator<dummy_grid_1d> it(grid);

    // pre-increment
    ASSERT_EQ(it.flat_index(), 0);
    auto& ref = ++it;
    ASSERT_EQ(it.flat_index(), 1);
    ASSERT_EQ(*it, 1.0);
    ASSERT_EQ(&ref, &it);

    // post-increment
    auto old_it = it++;
    ASSERT_EQ(it.flat_index(), 2);
    ASSERT_EQ(*it, 2.0);
    ASSERT_EQ(old_it.flat_index(), 1);
    ASSERT_EQ(*old_it, 1.0);

    // multiple increments.
    ++it;
    ++it;
    ASSERT_EQ(it.flat_index(), 4);
    ASSERT_EQ(*it, 4.0);
}

// Test pre-increment and post-increment operators for 2D grids.
TEST(SimplemcGridIterator, Increment2D) {
    dummy_grid_2d grid({ 3, 4 });
    simplemc::grid_iterator<dummy_grid_2d> it(grid);

    // pre-increment
    ASSERT_EQ(it.flat_index(), 0);
    ++it;
    ASSERT_EQ(it.flat_index(), 1);
    auto val1 = *it;
    ASSERT_DOUBLE_EQ(val1[0], 0.0);
    ASSERT_DOUBLE_EQ(val1[1], 1.0);

    // post-increment
    auto old_it = it++;
    ASSERT_EQ(it.flat_index(), 2);
    ASSERT_EQ(old_it.flat_index(), 1);

    // increment to next row
    it = simplemc::grid_iterator<dummy_grid_2d>(grid, 3);
    ++it;
    ASSERT_EQ(it.flat_index(), 4);
    auto val4 = *it;
    ASSERT_DOUBLE_EQ(val4[0], 1.0);
    ASSERT_DOUBLE_EQ(val4[1], 0.0);
}

// Test pre-decrement and post-decrement operators for 1D grids.
TEST(SimplemcGridIterator, Decrement1D) {
    dummy_grid_1d grid(5);
    simplemc::grid_iterator<dummy_grid_1d> it(grid, 4);

    // pre-decrement
    ASSERT_EQ(it.flat_index(), 4);
    auto& ref = --it;
    ASSERT_EQ(it.flat_index(), 3);
    ASSERT_EQ(*it, 3.0);
    ASSERT_EQ(&ref, &it);

    // post-decrement
    auto old_it = it--;
    ASSERT_EQ(it.flat_index(), 2);
    ASSERT_EQ(*it, 2.0);
    ASSERT_EQ(old_it.flat_index(), 3);
    ASSERT_EQ(*old_it, 3.0);

    // multiple decrements
    --it;
    --it;
    ASSERT_EQ(it.flat_index(), 0);
    ASSERT_EQ(*it, 0.0);
}

// Test pre-decrement and post-decrement operators for 2D grids.
TEST(SimplemcGridIterator, Decrement2D) {
    dummy_grid_2d grid({ 3, 4 });
    simplemc::grid_iterator<dummy_grid_2d> it(grid, 11);

    // pre-decrement
    ASSERT_EQ(it.flat_index(), 11);
    --it;
    ASSERT_EQ(it.flat_index(), 10);
    auto val10 = *it;
    ASSERT_DOUBLE_EQ(val10[0], 2.0);
    ASSERT_DOUBLE_EQ(val10[1], 2.0);

    // post-decrement
    auto old_it = it--;
    ASSERT_EQ(it.flat_index(), 9);
    ASSERT_EQ(old_it.flat_index(), 10);

    // decrement across row boundary
    it = simplemc::grid_iterator<dummy_grid_2d>(grid, 4);
    --it;
    ASSERT_EQ(it.flat_index(), 3);
    auto val3 = *it;
    ASSERT_DOUBLE_EQ(val3[0], 0.0);
    ASSERT_DOUBLE_EQ(val3[1], 3.0);
}

// Test addition assignment operator for 1D grids.
TEST(SimplemcGridIterator, AdditionAssignment1D) {
    dummy_grid_1d grid(10);
    simplemc::grid_iterator<dummy_grid_1d> it(grid);

    it += 3;
    ASSERT_EQ(it.flat_index(), 3);
    ASSERT_EQ(*it, 3.0);

    it += 0;
    ASSERT_EQ(it.flat_index(), 3);

    it += -2;
    ASSERT_EQ(it.flat_index(), 1);
    ASSERT_EQ(*it, 1.0);
}

// Test addition assignment operator for 2D grids.
TEST(SimplemcGridIterator, AdditionAssignment2D) {
    dummy_grid_2d grid({ 3, 4 });
    simplemc::grid_iterator<dummy_grid_2d> it(grid);

    it += 5;
    ASSERT_EQ(it.flat_index(), 5);
    auto val5 = *it;
    ASSERT_DOUBLE_EQ(val5[0], 1.0);
    ASSERT_DOUBLE_EQ(val5[1], 1.0);

    it += -3;
    ASSERT_EQ(it.flat_index(), 2);
    auto val2 = *it;
    ASSERT_DOUBLE_EQ(val2[0], 0.0);
    ASSERT_DOUBLE_EQ(val2[1], 2.0);
}

// Test subtraction assignment operator for 1D grids.
TEST(SimplemcGridIterator, SubtractionAssignment1D) {
    dummy_grid_1d grid(10);
    simplemc::grid_iterator<dummy_grid_1d> it(grid, 7);

    it -= 3;
    ASSERT_EQ(it.flat_index(), 4);
    ASSERT_EQ(*it, 4.0);

    it -= 0;
    ASSERT_EQ(it.flat_index(), 4);

    it -= -2;
    ASSERT_EQ(it.flat_index(), 6);
    ASSERT_EQ(*it, 6.0);
}

// Test subtraction assignment operator for 2D grids.
TEST(SimplemcGridIterator, SubtractionAssignment2D) {
    dummy_grid_2d grid({ 3, 4 });
    simplemc::grid_iterator<dummy_grid_2d> it(grid, 10);

    it -= 5;
    ASSERT_EQ(it.flat_index(), 5);
    auto val5 = *it;
    ASSERT_DOUBLE_EQ(val5[0], 1.0);
    ASSERT_DOUBLE_EQ(val5[1], 1.0);

    it -= -3;
    ASSERT_EQ(it.flat_index(), 8);
    auto val8 = *it;
    ASSERT_DOUBLE_EQ(val8[0], 2.0);
    ASSERT_DOUBLE_EQ(val8[1], 0.0);
}

// Test iterator addition operator for 1D grids.
TEST(SimplemcGridIterator, Addition1D) {
    dummy_grid_1d grid(10);
    simplemc::grid_iterator<dummy_grid_1d> it(grid, 2);

    // it + n
    auto it_plus_3 = it + 3;
    ASSERT_EQ(it.flat_index(), 2);
    ASSERT_EQ(it_plus_3.flat_index(), 5);
    ASSERT_EQ(*it_plus_3, 5.0);

    // n + it
    auto three_plus_it = 4 + it;
    ASSERT_EQ(three_plus_it.flat_index(), 6);
    ASSERT_EQ(*three_plus_it, 6.0);

    // negative offset
    auto it_minus = it + (-1);
    ASSERT_EQ(it_minus.flat_index(), 1);
    ASSERT_EQ(*it_minus, 1.0);
}

// Test iterator addition operator for 2D grids.
TEST(SimplemcGridIterator, Addition2D) {
    dummy_grid_2d grid({ 3, 4 });
    simplemc::grid_iterator<dummy_grid_2d> it(grid, 2);

    auto it_plus_5 = it + 5;
    ASSERT_EQ(it.flat_index(), 2);
    ASSERT_EQ(it_plus_5.flat_index(), 7);
    auto val7 = *it_plus_5;
    ASSERT_DOUBLE_EQ(val7[0], 1.0);
    ASSERT_DOUBLE_EQ(val7[1], 3.0);

    auto five_plus_it = 3 + it;
    ASSERT_EQ(five_plus_it.flat_index(), 5);
}

// Test iterator subtraction operator for 1D grids.
TEST(SimplemcGridIterator, Subtraction1D) {
    dummy_grid_1d grid(10);
    simplemc::grid_iterator<dummy_grid_1d> it(grid, 7);

    // it - n
    auto it_minus_3 = it - 3;
    ASSERT_EQ(it.flat_index(), 7);
    ASSERT_EQ(it_minus_3.flat_index(), 4);
    ASSERT_EQ(*it_minus_3, 4.0);

    // it1 - it2
    simplemc::grid_iterator<dummy_grid_1d> it2(grid, 2);
    auto diff = it - it2;
    ASSERT_EQ(diff, 5);

    auto diff_reverse = it2 - it;
    ASSERT_EQ(diff_reverse, -5);

    // same iterator
    auto diff_same = it - it;
    ASSERT_EQ(diff_same, 0);
}

// Test iterator subtraction operator for 2D grids.
TEST(SimplemcGridIterator, Subtraction2D) {
    dummy_grid_2d grid({ 3, 4 });
    simplemc::grid_iterator<dummy_grid_2d> it(grid, 10);

    auto it_minus_5 = it - 5;
    ASSERT_EQ(it.flat_index(), 10);
    ASSERT_EQ(it_minus_5.flat_index(), 5);
    auto val5 = *it_minus_5;
    ASSERT_DOUBLE_EQ(val5[0], 1.0);
    ASSERT_DOUBLE_EQ(val5[1], 1.0);

    // iterator difference
    simplemc::grid_iterator<dummy_grid_2d> it2(grid, 3);
    auto diff = it - it2;
    ASSERT_EQ(diff, 7);
}

// Test subscript operator for 1D grids.
TEST(SimplemcGridIterator, Subscript1D) {
    dummy_grid_1d grid(10);
    simplemc::grid_iterator<dummy_grid_1d> it(grid, 3);

    // positive offsets
    ASSERT_EQ(it[0], 3.0);
    ASSERT_EQ(it[1], 4.0);
    ASSERT_EQ(it[2], 5.0);

    // negative offsets
    ASSERT_EQ(it[-1], 2.0);
    ASSERT_EQ(it[-2], 1.0);
    ASSERT_EQ(it[-3], 0.0);

    // verify original iterator unchanged
    ASSERT_EQ(it.flat_index(), 3);
}

// Test subscript operator for 2D grids.
TEST(SimplemcGridIterator, Subscript2D) {
    dummy_grid_2d grid({ 3, 4 });
    simplemc::grid_iterator<dummy_grid_2d> it(grid, 5);

    // positive offsets
    auto val0 = it[0];
    ASSERT_DOUBLE_EQ(val0[0], 1.0);
    ASSERT_DOUBLE_EQ(val0[1], 1.0);

    auto val1 = it[1];
    ASSERT_DOUBLE_EQ(val1[0], 1.0);
    ASSERT_DOUBLE_EQ(val1[1], 2.0);

    auto val3 = it[3];
    ASSERT_DOUBLE_EQ(val3[0], 2.0);
    ASSERT_DOUBLE_EQ(val3[1], 0.0);

    // negative offsets
    auto val_neg1 = it[-1];
    ASSERT_DOUBLE_EQ(val_neg1[0], 1.0);
    ASSERT_DOUBLE_EQ(val_neg1[1], 0.0);

    // verify original iterator unchanged.
    ASSERT_EQ(it.flat_index(), 5);
}

// Test equality comparison operator.
TEST(SimplemcGridIterator, EqualityComparison) {
    dummy_grid_1d grid1(5);
    dummy_grid_1d grid2(5);

    simplemc::grid_iterator<dummy_grid_1d> it1(grid1, 2);
    simplemc::grid_iterator<dummy_grid_1d> it2(grid1, 2);
    simplemc::grid_iterator<dummy_grid_1d> it3(grid1, 3);
    simplemc::grid_iterator<dummy_grid_1d> it4(grid2, 2);

    // equality
    ASSERT_TRUE(it1 == it2);
    ASSERT_FALSE(it1 != it2);

    // inequality (different index)
    ASSERT_FALSE(it1 == it3);
    ASSERT_TRUE(it1 != it3);

    // inequality (different grid)
    ASSERT_FALSE(it1 == it4);
    ASSERT_TRUE(it1 != it4);
}

// Test three-way comparison operator for 1D grids.
TEST(SimplemcGridIterator, ThreeWayComparison1D) {
    dummy_grid_1d grid(10);

    simplemc::grid_iterator<dummy_grid_1d> it1(grid, 2);
    simplemc::grid_iterator<dummy_grid_1d> it2(grid, 5);
    simplemc::grid_iterator<dummy_grid_1d> it3(grid, 2);

    // less than
    ASSERT_TRUE(it1 < it2);
    ASSERT_FALSE(it2 < it1);
    ASSERT_FALSE(it1 < it3);

    // less than or equal
    ASSERT_TRUE(it1 <= it2);
    ASSERT_TRUE(it1 <= it3);
    ASSERT_FALSE(it2 <= it1);

    // greater than
    ASSERT_TRUE(it2 > it1);
    ASSERT_FALSE(it1 > it2);
    ASSERT_FALSE(it1 > it3);

    // greater than or equal
    ASSERT_TRUE(it2 >= it1);
    ASSERT_TRUE(it1 >= it3);
    ASSERT_FALSE(it1 >= it2);
}

// Test three-way comparison operator for 2D grids.
TEST(SimplemcGridIterator, ThreeWayComparison2D) {
    dummy_grid_2d grid({ 3, 4 });

    simplemc::grid_iterator<dummy_grid_2d> it1(grid, 3);
    simplemc::grid_iterator<dummy_grid_2d> it2(grid, 8);
    simplemc::grid_iterator<dummy_grid_2d> it3(grid, 3);

    ASSERT_TRUE(it1 < it2);
    ASSERT_TRUE(it1 <= it2);
    ASSERT_TRUE(it1 <= it3);
    ASSERT_TRUE(it2 > it1);
    ASSERT_TRUE(it2 >= it1);
    ASSERT_TRUE(it1 >= it3);
}

// Test dimension queries.
TEST(SimplemcGridIterator, DimensionQueries) {
    dummy_grid_1d grid_1d(5);
    dummy_grid_2d grid_2d({ 3, 4 });

    simplemc::grid_iterator<dummy_grid_1d> it_1d(grid_1d);
    simplemc::grid_iterator<dummy_grid_2d> it_2d(grid_2d);

    ASSERT_EQ(it_1d.dim(), 1);
    ASSERT_EQ(it_2d.dim(), 2);
}
