/**
 * @file format.hpp
 * @brief Some specialized formatters for fmtlib.
 */

#ifndef SIMPLEMC_UTILS_FORMAT_HPP
#define SIMPLEMC_UTILS_FORMAT_HPP

#include <fmt/format.h>

#include <complex>

namespace fmt {

/**
 * @brief Specalized formatter for std::complex.
 *
 * @tparam T Value type of std::complex.
 * @tparam Char Character type.
 */
template <typename T, typename Char>
struct formatter<std::complex<T>, Char> : public formatter<T, Char> {
    /**
     * @brief Base class formatter.
     */
    using base = formatter<T, Char>;

    /**
     * @brief Parses the format string.
     *
     * @param ctx Format parse context.
     * @return Iterator to the end of the parsed range.
     */
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) { return base::parse(ctx); }

    /**
     * @brief Formats the std::complex<T> type.
     *
     * @tparam FormatContext Format context type of fmtlib.
     * @param z std::complex value to format.
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

} // namespace fmt

#endif // SIMPLEMC_UTILS_FORMAT_HPP
