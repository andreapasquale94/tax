#pragma once

#include <array>
#include <cstddef>
#include <tax/core/concepts.hpp>
#include <tax/series/basis.hpp>
#include <tax/series/chebyshev_basis.hpp>
#include <tax/series/taylor_basis.hpp>

namespace tax
{

// ===========================================================================
// Series< Basis, N, T > — a truncated univariate expansion in a chosen basis
// ===========================================================================
//
// The generalisation of `TaylorExpansion` to an arbitrary polynomial family.
// The coefficient array and all linear-space behaviour live here; everything
// that depends on *which* polynomials are being summed is delegated to the
// `Basis` policy (see basis.hpp).
//
//   f(x) = sum_{k=0}^{N} c_k * P_k(x)
//
// `constant(v)` and `variable()` are basis-independent because every basis has
// P_0 == 1 and P_1 == x.
// ===========================================================================

template < typename Basis, int N, typename T = double >
    requires tax::Basis< Basis > && Scalar< T >
class Series
{
   public:
    static_assert( N >= 0, "Series order must be non-negative" );

    using basis = Basis;
    using scalar_type = T;
    using Data = std::array< T, std::size_t( N ) + 1 >;

    static constexpr int order_v = N;
    static constexpr std::size_t nCoefficients = std::size_t( N ) + 1;

    // ------------------------------------------------------------------
    // Constructors / factories
    // ------------------------------------------------------------------

    constexpr Series() noexcept = default;

    /// Constant expansion  f(x) = v  (v * P_0).
    /*implicit*/ constexpr Series( T v ) noexcept { c_[0] = v; }

    /// Construct directly from a raw coefficient array.
    explicit constexpr Series( Data c ) noexcept : c_{ c } {}

    [[nodiscard]] static constexpr Series zero() noexcept { return {}; }
    [[nodiscard]] static constexpr Series constant( T v ) noexcept { return Series{ v }; }

    /// The identity map  f(x) = x  (1 * P_1). Valid in any basis with P_1 == x.
    [[nodiscard]] static constexpr Series variable() noexcept
    {
        Series r{};
        if constexpr ( N >= 1 ) r.c_[1] = T{ 1 };
        return r;
    }

    // ------------------------------------------------------------------
    // Element access
    // ------------------------------------------------------------------

    /// Constant (zeroth) coefficient.
    [[nodiscard]] constexpr T value() const noexcept { return c_[0]; }

    [[nodiscard]] constexpr T operator[]( std::size_t k ) const noexcept { return c_[k]; }
    [[nodiscard]] constexpr T& operator[]( std::size_t k ) noexcept { return c_[k]; }

    /// Coefficient of P_k (0 outside the kept range).
    [[nodiscard]] constexpr T coeff( int k ) const noexcept
    {
        return ( k < 0 || k > N ) ? T{} : c_[std::size_t( k )];
    }

    [[nodiscard]] constexpr const Data& coefficients() const noexcept { return c_; }
    [[nodiscard]] constexpr Data& coefficients() noexcept { return c_; }

    // ------------------------------------------------------------------
    // Evaluation and calculus (delegated to the basis policy)
    // ------------------------------------------------------------------

    /// Evaluate f(x) at the absolute point x.
    [[nodiscard]] constexpr T eval( T x ) const noexcept
    {
        return Basis::template eval< T, N >( c_, x );
    }

    /// The derivative polynomial f'(x), in the same basis and order.
    [[nodiscard]] constexpr Series deriv() const noexcept
    {
        Series r{};
        Basis::template derivative< T, N >( r.c_, c_ );
        return r;
    }

    /// The indefinite-integral polynomial (constant of integration 0).
    [[nodiscard]] constexpr Series integ() const noexcept
    {
        Series r{};
        Basis::template integral< T, N >( r.c_, c_ );
        return r;
    }

   private:
    Data c_{};
};

// ---------------------------------------------------------------------------
// Convenience aliases
// ---------------------------------------------------------------------------

/// Order-N univariate Taylor (monomial-basis) series.
template < int N, typename T = double >
using TaylorSeries = Series< TaylorBasis, N, T >;

/// Order-N univariate Chebyshev (first-kind) series.
template < int N, typename T = double >
using ChebyshevSeries = Series< ChebyshevBasis, N, T >;

}  // namespace tax
