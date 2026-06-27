#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <tax/core/concepts.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/scheme/isotropic.hpp>
#include <tax/core/storage/dense.hpp>
#include <tax/series/basis.hpp>
#include <tax/series/chebyshev_basis.hpp>
#include <tax/series/taylor_basis.hpp>

namespace tax
{

// ===========================================================================
// Expansion< T, Basis, Scheme, Storage > — a truncated expansion in a chosen
// polynomial basis over a chosen monomial index scheme.
// ===========================================================================
//
// This is the basis-generic generalisation of `TaylorExpansion`. It composes
// three orthogonal policies:
//
//   * Basis   — the polynomial family (algebra): product, evaluation, and the
//               coefficient-space derivative / integral. See basis.hpp.
//   * Scheme  — the monomial layout (which multi-indices are kept and how they
//               flatten). IsotropicScheme<N,M> or MixedScheme<...>.
//   * Storage — coefficient storage. Dense (std::array) for now.
//
//   f(x) = Σ_k c_k · P_{α(k)}(x),   α(k) = Scheme::multiOf(k)
//
// where P over a multi-index α is the tensor product Π_i P_{α_i}(x_i). The
// carrier owns the coefficient array and the basis-independent linear-space
// surface; everything family-specific is one `Basis::…` call away.
//
// `TaylorExpansion<T, Scheme, Storage>` is the same mathematics as
// `Expansion<T, TaylorBasis, Scheme, Storage>` (shared kernels); it is retained
// as the feature-rich Taylor-basis instance (named axes, sparse, batch, Eigen).
// ===========================================================================

template < typename T, typename Basis, typename Scheme, typename Storage = storage::Dense >
    requires Scalar< T > && tax::Basis< Basis > && IndexScheme< Scheme >
class Expansion
{
   public:
    static_assert( std::is_same_v< Storage, storage::Dense >,
                   "Expansion currently supports only Dense storage" );
    static_assert( Scheme::order >= 0, "Expansion order must be non-negative" );
    static_assert( Scheme::vars >= 1, "Expansion variable count must be at least 1" );

    using scalar_type = T;
    using basis = Basis;
    using scheme = Scheme;
    using Data = std::array< T, Scheme::nCoeff >;
    using Input = std::array< T, std::size_t( Scheme::vars ) >;

    static constexpr int order_v = Scheme::order;
    static constexpr int vars_v = Scheme::vars;
    static constexpr std::size_t nCoefficients = Scheme::nCoeff;

    // ------------------------------------------------------------------
    // Constructors / factories
    // ------------------------------------------------------------------

    constexpr Expansion() noexcept = default;

    /// Constant expansion  f ≡ v   (v · P_0, and P_0 ≡ 1 in every basis).
    /*implicit*/ constexpr Expansion( T v ) noexcept { c_[0] = v; }

    /// Construct directly from a raw coefficient array.
    explicit constexpr Expansion( Data c ) noexcept : c_{ c } {}

    [[nodiscard]] static constexpr Expansion zero() noexcept { return {}; }
    [[nodiscard]] static constexpr Expansion constant( T v ) noexcept { return Expansion{ v }; }

    /// Univariate identity map  f(x) = x   (1 · P_1, and P_1 ≡ x in every basis).
    [[nodiscard]] static constexpr Expansion variable() noexcept
        requires( Scheme::isUnivariate )
    {
        Expansion r{};
        if constexpr ( Scheme::order >= 1 ) r.c_[1] = T{ 1 };
        return r;
    }

    /// The I-th coordinate variable  f(x) = x_I.
    template < int I >
    [[nodiscard]] static constexpr Expansion variable() noexcept
        requires( I >= 0 && I < Scheme::vars )
    {
        Expansion r{};
        if constexpr ( Scheme::order >= 1 )
        {
            MultiIndex< Scheme::vars > e{};
            e[std::size_t( I )] = 1;
            const std::size_t k = Scheme::flatOf( e );
            if ( k != Scheme::kNotInBox ) r.c_[k] = T{ 1 };
        }
        return r;
    }

    // ------------------------------------------------------------------
    // Element access
    // ------------------------------------------------------------------

    [[nodiscard]] constexpr T value() const noexcept { return c_[0]; }
    [[nodiscard]] constexpr T operator[]( std::size_t k ) const noexcept { return c_[k]; }
    [[nodiscard]] constexpr T& operator[]( std::size_t k ) noexcept { return c_[k]; }

    /// Coefficient of the basis monomial with flat index k (0 outside the kept set).
    [[nodiscard]] constexpr T coeff( int k ) const noexcept
    {
        return ( k < 0 || std::size_t( k ) >= nCoefficients ) ? T{} : c_[std::size_t( k )];
    }

    /// Coefficient at multi-index α (0 if α is outside the kept set).
    [[nodiscard]] constexpr T coeff( const MultiIndex< Scheme::vars >& alpha ) const noexcept
    {
        const std::size_t k = Scheme::flatOf( alpha );
        return k == Scheme::kNotInBox ? T{} : c_[k];
    }

    [[nodiscard]] constexpr const Data& coefficients() const noexcept { return c_; }
    [[nodiscard]] constexpr Data& coefficients() noexcept { return c_; }

    // ------------------------------------------------------------------
    // Evaluation (delegated to the basis policy)
    // ------------------------------------------------------------------

    /// Evaluate at the point vector x.
    [[nodiscard]] constexpr T eval( const Input& x ) const noexcept
    {
        return Basis::template eval< T, Scheme >( c_, x );
    }

    /// Univariate convenience: evaluate at the scalar point x.
    [[nodiscard]] constexpr T eval( T x ) const noexcept
        requires( Scheme::isUnivariate )
    {
        return Basis::template eval< T, Scheme >( c_, Input{ x } );
    }

    // ------------------------------------------------------------------
    // Calculus (delegated to the basis policy)
    // ------------------------------------------------------------------

    /// Partial derivative ∂f/∂x_I.
    template < int I = 0 >
    [[nodiscard]] constexpr Expansion deriv() const noexcept
        requires( I >= 0 && I < Scheme::vars )
    {
        Expansion r{};
        Basis::template derivative< T, Scheme >( r.c_, c_, I );
        return r;
    }

    /// Runtime-axis partial derivative ∂f/∂x_var.
    [[nodiscard]] Expansion deriv( int var ) const
    {
        if ( var < 0 || var >= Scheme::vars )
            throw std::out_of_range( "Expansion::deriv(var): var out of range" );
        Expansion r{};
        Basis::template derivative< T, Scheme >( r.c_, c_, var );
        return r;
    }

    /// Indefinite integral ∫ f dx_I (constant of integration 0).
    template < int I = 0 >
    [[nodiscard]] constexpr Expansion integ() const noexcept
        requires( I >= 0 && I < Scheme::vars )
    {
        Expansion r{};
        Basis::template integral< T, Scheme >( r.c_, c_, I );
        return r;
    }

    /// Runtime-axis indefinite integral ∫ f dx_var.
    [[nodiscard]] Expansion integ( int var ) const
    {
        if ( var < 0 || var >= Scheme::vars )
            throw std::out_of_range( "Expansion::integ(var): var out of range" );
        Expansion r{};
        Basis::template integral< T, Scheme >( r.c_, c_, var );
        return r;
    }

   private:
    Data c_{};
};

// ---------------------------------------------------------------------------
// Convenience aliases
// ---------------------------------------------------------------------------

/// Univariate basis-generic series (back-compat spelling).
template < typename Basis, int N, typename T = double >
using Series = Expansion< T, Basis, IsotropicScheme< N, 1 > >;

/// Order-N, M-variate Taylor (monomial-basis) expansion.
template < int N, int M = 1, typename T = double >
using TaylorSeries = Expansion< T, TaylorBasis, IsotropicScheme< N, M > >;

/// Order-N, M-variate Chebyshev (first-kind) expansion.
template < int N, int M = 1, typename T = double >
using ChebyshevSeries = Expansion< T, ChebyshevBasis, IsotropicScheme< N, M > >;

}  // namespace tax
