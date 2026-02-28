#pragma once

#include <tax/kernels.hpp>

namespace tax::detail {

// -- Additive binary ops ------------------------------------------------------

/// @brief Tag for coefficient-wise addition.
struct OpAdd {
    static constexpr bool is_additive  = true;
    static constexpr bool negate_right = false;
};

/// @brief Tag for coefficient-wise subtraction.
struct OpSub {
    static constexpr bool is_additive  = true;
    static constexpr bool negate_right = true;
};

// -- Non-additive binary ops (Cauchy / division) ------------------------------

/**
 * @brief Tag for DA multiplication via Cauchy product.
 * @details Produces coefficients truncated to total order `N`.
 */
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

/**
 * @brief Tag for DA division via reciprocal series then Cauchy product.
 * @details Requires a non-zero constant term in the denominator series.
 */
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

/// @brief Tag for `expr + scalar`.
struct OpScalarAdd { template <typename T, std::size_t S>
    static constexpr void apply(std::array<T,S>& o, T s) noexcept { o[0] += s; } };
/// @brief Tag for `expr - scalar`.
struct OpScalarSubR { template <typename T, std::size_t S>
    static constexpr void apply(std::array<T,S>& o, T s) noexcept { o[0] -= s; } };
/// @brief Tag for `scalar - expr`.
struct OpScalarSubL { template <typename T, std::size_t S>
    static constexpr void apply(std::array<T,S>& o, T s) noexcept
    { negateInPlace<T,S>(o); o[0] += s; } };
/// @brief Tag for `expr * scalar` or `scalar * expr`.
struct OpScalarMul  { template <typename T, std::size_t S>
    static constexpr void apply(std::array<T,S>& o, T s) noexcept
    { scaleInPlace<T,S>(o, s); } };
/// @brief Tag for `expr / scalar`.
struct OpScalarDivR { template <typename T, std::size_t S>
    static constexpr void apply(std::array<T,S>& o, T s) noexcept
    { scaleInPlace<T,S>(o, T{1} / s); } };

// -- Unary negation (0 temps) -------------------------------------------------

/// @brief Tag for unary negation.
struct OpNeg { template <typename T, std::size_t S>
    static constexpr void apply(std::array<T,S>& o) noexcept
    { negateInPlace<T,S>(o); } };

} // namespace tax::detail
