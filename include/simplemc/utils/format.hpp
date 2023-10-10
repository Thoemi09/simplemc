/**
 * @file format.hpp
 * @brief Some specialized formatter for fmtlib.
 */

#ifndef SIMPLEMC_UTILS_FORMAT_HPP
#define SIMPLEMC_UTILS_FORMAT_HPP

#include <fmt/format.h>

#include <complex>

namespace fmt {

/**
 * @brief Specalized formatter for std::complex.
 */
template <typename T, typename Char>
struct formatter<std::complex<T>, Char> : public formatter<T, Char> {
    using base = formatter<T, Char>;

    /**
     * @brief Parses the format string.

     * @param ctx The format parse context.
     * @return The iterator to the end of the parsed range.
     */
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) { return base::parse(ctx); }

    /**
     * @brief Formats the std::complex<T> type.

     * @param z The std::complex<T> value to format.
     * @param ctx The format context.
     * @return The iterator to the end of the formatted range.
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

} // namespace fmt

#endif // SIMPLEMC_UTILS_FORMAT_HPP
