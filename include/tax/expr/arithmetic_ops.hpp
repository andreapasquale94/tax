#pragma once

#include <tax/kernels.hpp>

namespace da::detail {

// =============================================================================
// Arithmetic operation tags for BinExpr / ScalarExpr / UnaryExpr
// =============================================================================

// -- Additive binary ops ------------------------------------------------------

struct OpAdd {
    static constexpr bool is_additive  = true;
    static constexpr bool negate_right = false;
};

struct OpSub {
    static constexpr bool is_additive  = true;
    static constexpr bool negate_right = true;
};

// -- Non-additive binary ops (Cauchy / division) ------------------------------

template <int N, int M>
struct OpMul {
    static constexpr bool is_additive    = false;
    static constexpr bool is_convolution = true;
    template <typename T>
    static constexpr void apply(
        std::array<T, numMonomials(N, M)>&       o,
        const std::array<T, numMonomials(N, M)>& a,
        const std::array<T, numMonomials(N, M)>& b) noexcept
    { cauchyProduct<T, N, M>(o, a, b); }
};

template <int N, int M>
struct OpDiv {
    static constexpr bool is_additive    = false;
    static constexpr bool is_convolution = false;
    template <typename T>
    static constexpr void apply(
        std::array<T, numMonomials(N, M)>&       o,
        const std::array<T, numMonomials(N, M)>& a,
        const std::array<T, numMonomials(N, M)>& b) noexcept
    {
        std::array<T, numMonomials(N, M)> rec{};
        seriesReciprocal<T, N, M>(rec, b);
        cauchyProduct<T, N, M>(o, a, rec);
    }
};

// -- Scalar ops (0 temps, all in-place on out) --------------------------------

struct OpScalarAdd { template <typename T, std::size_t S>
    static constexpr void apply(std::array<T,S>& o, T s) noexcept { o[0] += s; } };
struct OpScalarSubR { template <typename T, std::size_t S>
    static constexpr void apply(std::array<T,S>& o, T s) noexcept { o[0] -= s; } };
struct OpScalarSubL { template <typename T, std::size_t S>
    static constexpr void apply(std::array<T,S>& o, T s) noexcept
    { negateInPlace<T,S>(o); o[0] += s; } };
struct OpScalarMul  { template <typename T, std::size_t S>
    static constexpr void apply(std::array<T,S>& o, T s) noexcept
    { scaleInPlace<T,S>(o, s); } };
struct OpScalarDivR { template <typename T, std::size_t S>
    static constexpr void apply(std::array<T,S>& o, T s) noexcept
    { scaleInPlace<T,S>(o, T{1} / s); } };

// -- Unary negation (0 temps) -------------------------------------------------

struct OpNeg { template <typename T, std::size_t S>
    static constexpr void apply(std::array<T,S>& o) noexcept
    { negateInPlace<T,S>(o); } };

} // namespace da::detail
