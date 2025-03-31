/**
 * @file
 * @brief Implementation details for simplemc/numeric/laguerre_polynomial.cpp.
 */

#include <simplemc/numeric/laguerre_polynomial.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

namespace simplemc {

laguerre_polynomial::laguerre_polynomial(double x) : x_(x), llm1_(0), ll_(1), dllm1_(0), dll_(0) {
    if (x < 0) {
        throw simplemc_exception("Lagurre polynomials are only defined on the interval [0, inf)",
            "laguerre_polynomial::laguerre_polynomial");
    }
}

double laguerre_polynomial::next() {
    double llp1 = 1.0 - x_;
    double dllp1 = -1.0;
    if (l_ > 0) {
        const double a = -1.0 / (l_ + 1);
        const double b = 2.0 + a;
        const double c = 1.0 + a;
        llp1 = (a * x_ + b) * ll_ - c * llm1_;
        dllp1 = a * ll_ + (a * x_ + b) * dll_ - c * dllm1_;
    }
    llm1_ = ll_;
    ll_ = llp1;
    dllm1_ = dll_;
    dll_ = dllp1;
    ++l_;
    return llm1_;
}

} // namespace simplemc
