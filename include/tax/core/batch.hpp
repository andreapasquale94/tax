#pragma once

// ---------------------------------------------------------------------------
// Vectorised ("batched" / SIMD) Taylor-expansion coefficients.
// ---------------------------------------------------------------------------
//
// `Batch< T, K >` packs K independent floating-point problem instances into one
// coefficient slot, backed by an Eigen fixed-size array. Every operation a
// TaylorExpansion needs -- the four arithmetic operators, unary minus, and the
// transcendental seeds the recurrence kernels evaluate on the constant term --
// is element-wise across the K lanes. Substituting `Batch< T, K >` for the
// scalar coefficient type therefore makes
//
//     TaylorExpansion< Batch< double, K >, N, M >   (== tax::TE< N, M, K >)
//
// evaluate K independent expansions in lock-step: one pass through the kernels,
// K results, with the inner element-wise work vectorised. Restricted to dense
// storage: sparse storage keys off exact-zero coefficients, which is not well
// defined per-lane.

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <tax/core/concepts.hpp>
#include <tax/core/expansion.hpp>
#include <type_traits>

namespace tax
{

/**
 * @brief K-lane SIMD-friendly scalar: K independent `T` instances.
 *
 * @tparam T  Lane scalar (a floating-point type).
 * @tparam K  Number of lanes (>= 1).
 */
template < typename T, int K >
struct Batch
{
    static_assert( K >= 1, "Batch lane count must be >= 1" );
    static_assert( std::floating_point< T >, "Batch lane type must be floating-point" );

    using lane_type = T;
    static constexpr int lanes = K;
    using array_t = Eigen::Array< T, K, 1, Eigen::DontAlign >;

    array_t v;

    Batch() noexcept : v( array_t::Zero() ) {}
    /*implicit*/ Batch( T s ) noexcept : v( array_t::Constant( s ) ) {}
    /*implicit*/ Batch( const array_t& a ) noexcept : v( a ) {}

    /// @brief Build from K explicit lane values.
    [[nodiscard]] static Batch fromLanes( const std::array< T, std::size_t( K ) >& a ) noexcept
    {
        Batch r;
        for ( int i = 0; i < K; ++i ) r.v[i] = a[std::size_t( i )];
        return r;
    }

    [[nodiscard]] T operator[]( int i ) const noexcept { return v[i]; }
    [[nodiscard]] T& operator[]( int i ) noexcept { return v[i]; }
    [[nodiscard]] T lane( int i ) const noexcept { return v[i]; }

    Batch& operator+=( const Batch& o ) noexcept
    {
        v += o.v;
        return *this;
    }
    Batch& operator-=( const Batch& o ) noexcept
    {
        v -= o.v;
        return *this;
    }
    Batch& operator*=( const Batch& o ) noexcept
    {
        v *= o.v;
        return *this;
    }
    Batch& operator/=( const Batch& o ) noexcept
    {
        v /= o.v;
        return *this;
    }
};

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------
template < typename T, int K >
[[nodiscard]] inline Batch< T, K > operator+( Batch< T, K > a, const Batch< T, K >& b ) noexcept
{
    a += b;
    return a;
}
template < typename T, int K >
[[nodiscard]] inline Batch< T, K > operator-( Batch< T, K > a, const Batch< T, K >& b ) noexcept
{
    a -= b;
    return a;
}
template < typename T, int K >
[[nodiscard]] inline Batch< T, K > operator*( Batch< T, K > a, const Batch< T, K >& b ) noexcept
{
    a *= b;
    return a;
}
template < typename T, int K >
[[nodiscard]] inline Batch< T, K > operator/( Batch< T, K > a, const Batch< T, K >& b ) noexcept
{
    a /= b;
    return a;
}
template < typename T, int K >
[[nodiscard]] inline Batch< T, K > operator-( const Batch< T, K >& a ) noexcept
{
    return Batch< T, K >( ( -a.v ).eval() );
}

// Comparisons collapse all lanes (used only by the dense deriv/integ zero-skip
// paths: a coefficient is skipped only when every lane is exactly zero).
template < typename T, int K >
[[nodiscard]] inline bool operator==( const Batch< T, K >& a, const Batch< T, K >& b ) noexcept
{
    return ( a.v == b.v ).all();
}
template < typename T, int K >
[[nodiscard]] inline bool operator!=( const Batch< T, K >& a, const Batch< T, K >& b ) noexcept
{
    return !( a == b );
}

// ---------------------------------------------------------------------------
// Element-wise math (found by ADL from the kernels). Common functions use
// Eigen's vectorised array methods; the few Eigen-core lacks (cbrt, erf,
// atan2) fall back to a per-lane apply.
// ---------------------------------------------------------------------------
#define TAX_BATCH_UNARY( fn )                                                \
    template < typename T, int K >                                           \
    [[nodiscard]] inline Batch< T, K > fn( const Batch< T, K >& a ) noexcept \
    {                                                                        \
        return Batch< T, K >( a.v.fn().eval() );                             \
    }

TAX_BATCH_UNARY( sqrt )
TAX_BATCH_UNARY( exp )
TAX_BATCH_UNARY( log )
TAX_BATCH_UNARY( sin )
TAX_BATCH_UNARY( cos )
TAX_BATCH_UNARY( tan )
TAX_BATCH_UNARY( asin )
TAX_BATCH_UNARY( acos )
TAX_BATCH_UNARY( atan )
TAX_BATCH_UNARY( sinh )
TAX_BATCH_UNARY( cosh )
TAX_BATCH_UNARY( tanh )
TAX_BATCH_UNARY( asinh )
TAX_BATCH_UNARY( acosh )
TAX_BATCH_UNARY( atanh )
TAX_BATCH_UNARY( abs )
#undef TAX_BATCH_UNARY

template < typename T, int K >
[[nodiscard]] inline Batch< T, K > pow( const Batch< T, K >& a, const Batch< T, K >& b ) noexcept
{
    return Batch< T, K >( a.v.pow( b.v ).eval() );
}

template < typename T, int K >
[[nodiscard]] inline Batch< T, K > cbrt( const Batch< T, K >& a ) noexcept
{
    return Batch< T, K >( a.v.unaryExpr( []( T x ) {
                                 using std::cbrt;
                                 return cbrt( x );
                             } )
                              .eval() );
}
template < typename T, int K >
[[nodiscard]] inline Batch< T, K > erf( const Batch< T, K >& a ) noexcept
{
    return Batch< T, K >( a.v.unaryExpr( []( T x ) {
                                 using std::erf;
                                 return erf( x );
                             } )
                              .eval() );
}
template < typename T, int K >
[[nodiscard]] inline Batch< T, K > atan2( const Batch< T, K >& y, const Batch< T, K >& x ) noexcept
{
    return Batch< T, K >( y.v.binaryExpr( x.v,
                                          []( T yy, T xx ) {
                                              using std::atan2;
                                              return atan2( yy, xx );
                                          } )
                              .eval() );
}

// ---------------------------------------------------------------------------
// Coefficient-trait opt-ins
// ---------------------------------------------------------------------------
template < typename T, int K >
struct is_tax_scalar< Batch< T, K > > : std::true_type
{
};

template < typename T, int K >
struct real_scalar< Batch< T, K > >
{
    using type = T;
};

// ---------------------------------------------------------------------------
// Convenience aliases
// ---------------------------------------------------------------------------

/// @brief `Batchd<K>` — K-lane double batch.
template < int K >
using Batchd = Batch< double, K >;

/// @brief `Batchf<K>` — K-lane float batch.
template < int K >
using Batchf = Batch< float, K >;

}  // namespace tax

// ---------------------------------------------------------------------------
// Eigen scalar traits — lets Eigen matrices/vectors hold Batch (and, by
// extension, TaylorExpansion<Batch,...>) as their scalar type.
// ---------------------------------------------------------------------------
namespace Eigen
{

template < typename T, int K >
struct NumTraits< tax::Batch< T, K > > : NumTraits< T >
{
    using Self = tax::Batch< T, K >;
    using Real = Self;
    using NonInteger = Self;
    using Nested = Self;
    using Literal = Self;
    enum
    {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = K,
        AddCost = K,
        MulCost = K
    };
    static Self epsilon() { return Self( NumTraits< T >::epsilon() ); }
    static Self dummy_precision() { return Self( NumTraits< T >::dummy_precision() ); }
    static Self highest() { return Self( NumTraits< T >::highest() ); }
    static Self lowest() { return Self( NumTraits< T >::lowest() ); }
    static Self infinity() { return Self( NumTraits< T >::infinity() ); }
    static Self quiet_NaN() { return Self( NumTraits< T >::quiet_NaN() ); }
    static int digits10() { return NumTraits< T >::digits10(); }
};

}  // namespace Eigen
