#pragma once

#include <tax/core/scheme/isotropic.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/kernels/algebra.hpp>
#include <tax/kernels/cauchy.hpp>
#include <tax/kernels/sparse_cauchy.hpp>
#include <tax/kernels/sparse_subs.hpp>
#include <type_traits>

namespace tax
{

// ===========================================================================
// Dense arithmetic — basis-generic over Expansion< T, Basis, Scheme >.
//
// Every linear-space operation, and the bilinear product (which delegates to
// the basis' own Basis::product), is identical for every basis, so each is
// written once here over `Basis B`. Division is the exception: the
// expansion/expansion quotient and the scalar/expansion reciprocal use the
// Taylor recurrence kernels and stay TaylorBasis-specific — other families
// supply their own division where it is defined (e.g. series/chebyshev_math.hpp).
//
// `TaylorExpansion< T, Scheme >` is the B = TaylorBasis instance of Expansion,
// so these templates cover the Taylor hot path as well as Chebyshev/Legendre/…
// ===========================================================================

// ---------------------------------------------------------------------------
// Addition
// ---------------------------------------------------------------------------

template < typename T, Basis B, IndexScheme Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator+(
    const Expansion< T, B, Scheme >& a, const Expansion< T, B, Scheme >& b ) noexcept
{
    Expansion< T, B, Scheme > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) r[k] = a[k] + b[k];
    return r;
}

template < typename T, Basis B, IndexScheme Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator+( const Expansion< T, B, Scheme >& a,
                                                             std::type_identity_t< T > s ) noexcept
{
    Expansion< T, B, Scheme > r = a;
    r[0] += s;
    return r;
}

template < typename T, Basis B, IndexScheme Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator+(
    std::type_identity_t< T > s, const Expansion< T, B, Scheme >& a ) noexcept
{
    return a + s;
}

// ---------------------------------------------------------------------------
// Subtraction
// ---------------------------------------------------------------------------

template < typename T, Basis B, IndexScheme Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator-(
    const Expansion< T, B, Scheme >& a, const Expansion< T, B, Scheme >& b ) noexcept
{
    Expansion< T, B, Scheme > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) r[k] = a[k] - b[k];
    return r;
}

template < typename T, Basis B, IndexScheme Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator-( const Expansion< T, B, Scheme >& a,
                                                             std::type_identity_t< T > s ) noexcept
{
    Expansion< T, B, Scheme > r = a;
    r[0] -= s;
    return r;
}

template < typename T, Basis B, IndexScheme Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator-(
    std::type_identity_t< T > s, const Expansion< T, B, Scheme >& a ) noexcept
{
    Expansion< T, B, Scheme > r;
    r[0] = s - a[0];
    for ( std::size_t k = 1; k < a.nCoefficients; ++k ) r[k] = -a[k];
    return r;
}

// ---------------------------------------------------------------------------
// Unary negation
// ---------------------------------------------------------------------------

template < typename T, Basis B, IndexScheme Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator-(
    const Expansion< T, B, Scheme >& a ) noexcept
{
    Expansion< T, B, Scheme > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) r[k] = -a[k];
    return r;
}

// ---------------------------------------------------------------------------
// Scalar multiplication / division
// ---------------------------------------------------------------------------

template < typename T, Basis B, IndexScheme Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator*( const Expansion< T, B, Scheme >& a,
                                                             std::type_identity_t< T > s ) noexcept
{
    Expansion< T, B, Scheme > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) r[k] = a[k] * s;
    return r;
}

template < typename T, Basis B, IndexScheme Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator*(
    std::type_identity_t< T > s, const Expansion< T, B, Scheme >& a ) noexcept
{
    return a * s;
}

template < typename T, Basis B, IndexScheme Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator/( const Expansion< T, B, Scheme >& a,
                                                             std::type_identity_t< T > s ) noexcept
{
    return a * ( T( 1 ) / s );
}

// ---------------------------------------------------------------------------
// Bilinear product (the basis-defined Cauchy/convolution product)
// ---------------------------------------------------------------------------

template < typename T, Basis B, IndexScheme Scheme >
[[nodiscard]] constexpr Expansion< T, B, Scheme > operator*(
    const Expansion< T, B, Scheme >& a, const Expansion< T, B, Scheme >& b ) noexcept
{
    Expansion< T, B, Scheme > r;
    B::template product< T, Scheme >( r.coefficients(), a.coefficients(), b.coefficients() );
    return r;
}

// ---------------------------------------------------------------------------
// Compound assignment (dense)
// ---------------------------------------------------------------------------

template < typename T, Basis B, IndexScheme Scheme >
constexpr Expansion< T, B, Scheme >& operator+=( Expansion< T, B, Scheme >& a,
                                                 const Expansion< T, B, Scheme >& b ) noexcept
{
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) a[k] += b[k];
    return a;
}

template < typename T, Basis B, IndexScheme Scheme >
constexpr Expansion< T, B, Scheme >& operator-=( Expansion< T, B, Scheme >& a,
                                                 const Expansion< T, B, Scheme >& b ) noexcept
{
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) a[k] -= b[k];
    return a;
}

template < typename T, Basis B, IndexScheme Scheme >
constexpr Expansion< T, B, Scheme >& operator+=( Expansion< T, B, Scheme >& a,
                                                 std::type_identity_t< T > s ) noexcept
{
    a[0] += s;
    return a;
}

template < typename T, Basis B, IndexScheme Scheme >
constexpr Expansion< T, B, Scheme >& operator-=( Expansion< T, B, Scheme >& a,
                                                 std::type_identity_t< T > s ) noexcept
{
    a[0] -= s;
    return a;
}

template < typename T, Basis B, IndexScheme Scheme >
constexpr Expansion< T, B, Scheme >& operator*=( Expansion< T, B, Scheme >& a,
                                                 std::type_identity_t< T > s ) noexcept
{
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) a[k] *= s;
    return a;
}

template < typename T, Basis B, IndexScheme Scheme >
constexpr Expansion< T, B, Scheme >& operator/=( Expansion< T, B, Scheme >& a,
                                                 std::type_identity_t< T > s ) noexcept
{
    return a *= ( T( 1 ) / s );
}

/// In-place bilinear product.
template < typename T, Basis B, IndexScheme Scheme >
constexpr Expansion< T, B, Scheme >& operator*=( Expansion< T, B, Scheme >& a,
                                                 const Expansion< T, B, Scheme >& b ) noexcept
{
    a = a * b;
    return a;
}

// ===========================================================================
// Division (TaylorBasis only): the quotient and scalar reciprocal use the
// Taylor recurrence kernels. Other bases provide their own where defined.
// ===========================================================================

/// Scalar / expansion: `s / a = s * (1 / a)`.
template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator/(
    std::type_identity_t< T > s, const TaylorExpansion< T, Scheme >& a ) noexcept
{
    TaylorExpansion< T, Scheme > inv_a;
    detail::kernels::seriesReciprocal< T, Scheme >( inv_a.coefficients(), a.coefficients() );
    return inv_a * s;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator/(
    const TaylorExpansion< T, Scheme >& a, const TaylorExpansion< T, Scheme >& b ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    detail::kernels::seriesDivide< T, Scheme >( r.coefficients(), a.coefficients(),
                                                b.coefficients() );
    return r;
}

template < typename T, IndexScheme Scheme >
constexpr TaylorExpansion< T, Scheme >& operator/=( TaylorExpansion< T, Scheme >& a,
                                                    const TaylorExpansion< T, Scheme >& b ) noexcept
{
    std::array< T, Scheme::nCoeff > inv_b{};
    detail::kernels::seriesReciprocal< T, Scheme >( inv_b, b.coefficients() );
    std::array< T, Scheme::nCoeff > tmp{};
    tax::cauchyProduct< T, Scheme >( tmp, a.coefficients(), inv_b );
    a.coefficients() = tmp;
    return a;
}

// ===========================================================================
// Sparse arithmetic:  S+S, S-S, -S, S+T, T+S, S-T, T-S, S*T, T*S, S/T
// ===========================================================================

using Sparse = storage::Sparse;

/// Sparse + Sparse: two-pointer merge over sorted flat indices.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, Sparse > operator+(
    const TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& a,
    const TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& b ) noexcept
{
    TaylorExpansion< T, IsotropicScheme< N, M >, Sparse > r;
    auto& ri = r.container().rawIndices();
    auto& rv = r.container().rawValues();
    a.container().forEachPair( b.container(), [&ri, &rv]( std::size_t k, T va, T vb ) {
        const T s = va + vb;
        if ( s != T{ 0 } )
        {
            ri.push_back( storage::flat_index_t( k ) );
            rv.push_back( s );
        }
    } );
    return r;
}

/// Sparse - Sparse: two-pointer merge over sorted flat indices.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, Sparse > operator-(
    const TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& a,
    const TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& b ) noexcept
{
    TaylorExpansion< T, IsotropicScheme< N, M >, Sparse > r;
    auto& ri = r.container().rawIndices();
    auto& rv = r.container().rawValues();
    a.container().forEachPair( b.container(), [&ri, &rv]( std::size_t k, T va, T vb ) {
        const T d = va - vb;
        if ( d != T{ 0 } )
        {
            ri.push_back( storage::flat_index_t( k ) );
            rv.push_back( d );
        }
    } );
    return r;
}

/// Unary negation (support unchanged; values negated).
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, Sparse > operator-(
    const TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& a ) noexcept
{
    TaylorExpansion< T, IsotropicScheme< N, M >, Sparse > r;
    auto& ri = r.container().rawIndices();
    auto& rv = r.container().rawValues();
    a.container().forEachNonzero( [&ri, &rv]( std::size_t k, T v ) {
        ri.push_back( storage::flat_index_t( k ) );
        rv.push_back( -v );
    } );
    return r;
}

/// Sparse * scalar (support unchanged for s != 0).
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, Sparse > operator*(
    const TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& a,
    std::type_identity_t< T > s ) noexcept
{
    if ( s == T{ 0 } ) return TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >{};
    TaylorExpansion< T, IsotropicScheme< N, M >, Sparse > r;
    auto& ri = r.container().rawIndices();
    auto& rv = r.container().rawValues();
    a.container().forEachNonzero( [&ri, &rv, s]( std::size_t k, T v ) {
        ri.push_back( storage::flat_index_t( k ) );
        rv.push_back( v * s );
    } );
    return r;
}

/// Scalar * Sparse.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, Sparse > operator*(
    std::type_identity_t< T > s,
    const TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& a ) noexcept
{
    return a * s;
}

/// Sparse / scalar.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, Sparse > operator/(
    const TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& a,
    std::type_identity_t< T > s ) noexcept
{
    return a * ( T{ 1 } / s );
}

/// Sparse + scalar: add to constant term.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, Sparse > operator+(
    const TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& a,
    std::type_identity_t< T > s ) noexcept
{
    if ( s == T{ 0 } ) return a;
    const auto ai = a.container().support();
    const auto av = a.container().values();

    TaylorExpansion< T, IsotropicScheme< N, M >, Sparse > r;
    auto& ri = r.container().rawIndices();
    auto& rv = r.container().rawValues();
    ri.reserve( ai.size() + 1 );
    rv.reserve( av.size() + 1 );

    // Emit the constant term first, then bulk-append the (already sorted)
    // remainder — avoids the O(nnz) front-insert of accumulate(0, s).
    std::size_t b = 0;
    if ( !ai.empty() && ai.front() == 0 )
    {
        const T c = av.front() + s;
        if ( c != T{ 0 } )
        {
            ri.push_back( storage::flat_index_t( 0 ) );
            rv.push_back( c );
        }
        b = 1;
    } else
    {
        ri.push_back( storage::flat_index_t( 0 ) );
        rv.push_back( s );
    }
    ri.insert( ri.end(), ai.begin() + std::ptrdiff_t( b ), ai.end() );
    rv.insert( rv.end(), av.begin() + std::ptrdiff_t( b ), av.end() );
    return r;
}

/// Scalar + Sparse.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, Sparse > operator+(
    std::type_identity_t< T > s,
    const TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& a ) noexcept
{
    return a + s;
}

/// Sparse - scalar.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, Sparse > operator-(
    const TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& a,
    std::type_identity_t< T > s ) noexcept
{
    return a + ( -s );
}

/// Scalar - Sparse.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, Sparse > operator-(
    std::type_identity_t< T > s,
    const TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& a ) noexcept
{
    return ( -a ) + s;
}

/// Sparse * Sparse: truncated Cauchy product via the sparse kernel (may allocate).
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, Sparse > operator*(
    const TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& a,
    const TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& b )
{
    TaylorExpansion< T, IsotropicScheme< N, M >, Sparse > r;
    detail::kernels::sparseCauchyProduct< T, N, M >( r.container(), a.container(), b.container() );
    return r;
}

/// Sparse / Sparse: Cauchy product of a and 1/b.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, Sparse > operator/(
    const TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& a,
    const TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& b )
{
    TaylorExpansion< T, IsotropicScheme< N, M >, Sparse > inv_b;
    detail::kernels::seriesReciprocalSparse< T, N, M >( inv_b.container(), b.container() );
    TaylorExpansion< T, IsotropicScheme< N, M >, Sparse > r;
    detail::kernels::sparseCauchyProduct< T, N, M >( r.container(), a.container(),
                                                     inv_b.container() );
    return r;
}

}  // namespace tax
