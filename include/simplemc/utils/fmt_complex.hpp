// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Specialized fmt formatter for `std::complex`.
 */

#ifndef SIMPLEMC_UTILS_FMT_COMPLEX_HPP
#define SIMPLEMC_UTILS_FMT_COMPLEX_HPP

#include <fmt/format.h>

#include <complex>

namespace fmt {

/**
 * @addtogroup simplemc-utils-other
 * @{
 */

/**
 * @brief Specialized **fmt** formatter for `std::complex`.
 *
 * @details The format specifier is simply applied to both the real and imaginary parts:
 *
 * @code{.cpp}
 * std::complex<double> z(1.0003, 2.00041291823);
 * fmt::print("z = {:^30.15}\n", z);
 * @endcode
 *
 * Output:
 *
 * ```
 * z = (            1.0003            ,        2.00041291823         )
 * ```
 *
 * See @ref tut_utils_3 and <a href="https://fmt.dev/latest/index.html">fmt documentation</a> for more
 * information.
 *
 * @warning fmtlib has recently added their own formatter for `std::complex` in the `fmt/std.h` header
 * file. Including the current header together with `fmt/std.h` will result in a compilation error.
 *
 * @tparam T Value type of `std::complex`.
 * @tparam Char Character type.
 */
template <typename T, typename Char>
struct formatter<std::complex<T>, Char> : public formatter<T, Char> {
    /**
     * @brief Base class formatter.
     */
    using base = formatter<T, Char>;

    /**
     * @brief Parse the format string.
     *
     * @param ctx Format parse context.
     * @return Iterator to the end of the parsed range.
     */
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) { return base::parse(ctx); }

    /**
     * @brief Format the `std::complex` type.
     *
     * @tparam FormatContext Format context type of **fmt**.
     * @param z Complex value to format.
     * @param ctx Format context.
     * @return Iterator to the end of the formatted range.
     */
    template <typename FormatContext>
    auto format(const std::complex<T>& z, FormatContext& ctx) const {
        fmt::format_to(ctx.out(), "(");
        base::format(z.real(), ctx);
        fmt::format_to(ctx.out(), ",");
        base::format(z.imag(), ctx);
        return fmt::format_to(ctx.out(), ")");
    }
};

/** @} */

} // namespace fmt

#endif // SIMPLEMC_UTILS_FMT_COMPLEX_HPP
