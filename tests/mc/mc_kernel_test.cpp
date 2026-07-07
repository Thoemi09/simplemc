#include <simplemc/mc/concepts.hpp>
#include <simplemc/random/xoshiro256.hpp>

#include <gtest/gtest.h>

using namespace simplemc;

namespace {

struct good_kernel {
    void step(xoshiro256ss&) {}
};

struct bad_kernel_no_step {};

struct bad_kernel_wrong_arg {
    void step(int) {}
};

} // namespace

static_assert(mc_kernel<good_kernel, xoshiro256ss>);
static_assert(!mc_kernel<bad_kernel_no_step, xoshiro256ss>);
static_assert(!mc_kernel<bad_kernel_wrong_arg, xoshiro256ss>);
static_assert(!mc_kernel<int, xoshiro256ss>);

// The static_asserts above do the work; a dummy runtime test keeps GoogleTest happy.
TEST(MCKernel, ConceptCompiles) {
    good_kernel k;
    xoshiro256ss rng;
    k.step(rng);
    SUCCEED();
}
