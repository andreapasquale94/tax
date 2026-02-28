#pragma once

#include <tax/kernels.hpp>

namespace tax::detail {

/**
 * @brief Tag for `square(expr)`.
 * @details Delegates to `seriesSquare`.
 */
template <int N, int M>
struct OpSquare {
    template <typename T>
    static constexpr void apply(
        std::array<T, numMonomials(N, M)>&       out,
        const std::array<T, numMonomials(N, M)>& a) noexcept
    { seriesSquare<T, N, M>(out, a); }
};

/**
 * @brief Tag for `cube(expr)`.
 * @details Delegates to `seriesCube`.
 */
template <int N, int M>
struct OpCube {
    template <typename T>
    static constexpr void apply(
        std::array<T, numMonomials(N, M)>&       out,
        const std::array<T, numMonomials(N, M)>& a) noexcept
    { seriesCube<T, N, M>(out, a); }
};

/**
 * @brief Tag for `sqrt(expr)`.
 * @details Delegates to `seriesSqrt`.
 */
template <int N, int M>
struct OpSqrt {
    template <typename T>
    static constexpr void apply(
        std::array<T, numMonomials(N, M)>&       out,
        const std::array<T, numMonomials(N, M)>& a) noexcept
    { seriesSqrt<T, N, M>(out, a); }
};

/**
 * @brief Tag for reciprocal series `1/expr`.
 * @details Delegates to `seriesReciprocal`.
 */
template <int N, int M>
struct OpReciprocal {
    template <typename T>
    static constexpr void apply(
        std::array<T, numMonomials(N, M)>&       out,
        const std::array<T, numMonomials(N, M)>& a) noexcept
    { seriesReciprocal<T, N, M>(out, a); }
};

/**
 * @brief Tag for `sin(expr)`.
 * @details Delegates to `seriesSin`.
 */
template <int N, int M>
struct OpSin {
    template <typename T>
    static constexpr void apply(
        std::array<T, numMonomials(N, M)>&       out,
        const std::array<T, numMonomials(N, M)>& a) noexcept
    { seriesSin<T, N, M>(out, a); }
};

/**
 * @brief Tag for `cos(expr)`.
 * @details Delegates to `seriesCos`.
 */
template <int N, int M>
struct OpCos {
    template <typename T>
    static constexpr void apply(
        std::array<T, numMonomials(N, M)>&       out,
        const std::array<T, numMonomials(N, M)>& a) noexcept
    { seriesCos<T, N, M>(out, a); }
};

/**
 * @brief Tag for `tan(expr)`.
 * @details Delegates to `seriesTan`.
 */
template <int N, int M>
struct OpTan {
    template <typename T>
    static constexpr void apply(
        std::array<T, numMonomials(N, M)>&       out,
        const std::array<T, numMonomials(N, M)>& a) noexcept
    { seriesTan<T, N, M>(out, a); }
};

/**
 * @brief Tag for natural logarithm `log(expr)`.
 * @details Delegates to `seriesLog`.
 */
template <int N, int M>
struct OpLog {
    template <typename T>
    static constexpr void apply(
        std::array<T, numMonomials(N, M)>&       out,
        const std::array<T, numMonomials(N, M)>& a) noexcept
    { seriesLog<T, N, M>(out, a); }
};

/**
 * @brief Tag for base-10 logarithm `log10(expr)`.
 * @details Uses `seriesLog` then scales by `1/log(10)`.
 */
template <int N, int M>
struct OpLog10 {
    template <typename T>
    static constexpr void apply(
        std::array<T, numMonomials(N, M)>&       out,
        const std::array<T, numMonomials(N, M)>& a) noexcept
    {
        seriesLog<T, N, M>(out, a);
        const T scale = T{1} / std::log(T{10});
        for (auto& v : out) v *= scale;
    }
};

} // namespace tax::detail
