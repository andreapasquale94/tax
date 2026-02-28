#pragma once

#include <tax/kernels.hpp>

namespace da::detail {

// =============================================================================
// Math operation tags for FuncExpr
// =============================================================================

template <int N, int M>
struct OpSquare {
    template <typename T>
    static constexpr void apply(
        std::array<T, numMonomials(N, M)>&       out,
        const std::array<T, numMonomials(N, M)>& a) noexcept
    { seriesSquare<T, N, M>(out, a); }
};

template <int N, int M>
struct OpCube {
    template <typename T>
    static constexpr void apply(
        std::array<T, numMonomials(N, M)>&       out,
        const std::array<T, numMonomials(N, M)>& a) noexcept
    { seriesCube<T, N, M>(out, a); }
};

template <int N, int M>
struct OpSqrt {
    template <typename T>
    static constexpr void apply(
        std::array<T, numMonomials(N, M)>&       out,
        const std::array<T, numMonomials(N, M)>& a) noexcept
    { seriesSqrt<T, N, M>(out, a); }
};

template <int N, int M>
struct OpReciprocal {
    template <typename T>
    static constexpr void apply(
        std::array<T, numMonomials(N, M)>&       out,
        const std::array<T, numMonomials(N, M)>& a) noexcept
    { seriesReciprocal<T, N, M>(out, a); }
};

template <int N, int M>
struct OpSin {
    template <typename T>
    static constexpr void apply(
        std::array<T, numMonomials(N, M)>&       out,
        const std::array<T, numMonomials(N, M)>& a) noexcept
    { seriesSin<T, N, M>(out, a); }
};

template <int N, int M>
struct OpCos {
    template <typename T>
    static constexpr void apply(
        std::array<T, numMonomials(N, M)>&       out,
        const std::array<T, numMonomials(N, M)>& a) noexcept
    { seriesCos<T, N, M>(out, a); }
};

template <int N, int M>
struct OpTan {
    template <typename T>
    static constexpr void apply(
        std::array<T, numMonomials(N, M)>&       out,
        const std::array<T, numMonomials(N, M)>& a) noexcept
    { seriesTan<T, N, M>(out, a); }
};

} // namespace da::detail
