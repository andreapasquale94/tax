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

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator+(
    const TaylorExpansion< T, Scheme >& a, const TaylorExpansion< T, Scheme >& b ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) r[k] = a[k] + b[k];
    return r;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator+(
    const TaylorExpansion< T, Scheme >& a, std::type_identity_t< T > s ) noexcept
{
    TaylorExpansion< T, Scheme > r = a;
    r[0] += s;
    return r;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator+(
    std::type_identity_t< T > s, const TaylorExpansion< T, Scheme >& a ) noexcept
{
    return a + s;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator-(
    const TaylorExpansion< T, Scheme >& a, const TaylorExpansion< T, Scheme >& b ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) r[k] = a[k] - b[k];
    return r;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator-(
    const TaylorExpansion< T, Scheme >& a, std::type_identity_t< T > s ) noexcept
{
    TaylorExpansion< T, Scheme > r = a;
    r[0] -= s;
    return r;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator-(
    std::type_identity_t< T > s, const TaylorExpansion< T, Scheme >& a ) noexcept
{
    TaylorExpansion< T, Scheme > r = -a;
    r[0] += s;
    return r;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator-(
    const TaylorExpansion< T, Scheme >& a ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) r[k] = -a[k];
    return r;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator*(
    const TaylorExpansion< T, Scheme >& a, std::type_identity_t< T > s ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) r[k] = a[k] * s;
    return r;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator*(
    std::type_identity_t< T > s, const TaylorExpansion< T, Scheme >& a ) noexcept
{
    return a * s;
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator/(
    const TaylorExpansion< T, Scheme >& a, std::type_identity_t< T > s ) noexcept
{
    return a * ( T( 1 ) / s );
}

template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator/(
    std::type_identity_t< T > s, const TaylorExpansion< T, Scheme >& a ) noexcept
{
    TaylorExpansion< T, Scheme > inv_a;
    detail::kernels::seriesReciprocal< T, Scheme >( inv_a.coefficients(), a.coefficients() );
    return inv_a * s;
}

// Cauchy (TE x TE) product.
template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > operator*(
    const TaylorExpansion< T, Scheme >& a, const TaylorExpansion< T, Scheme >& b ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    tax::cauchyProduct< T, Scheme >( r.coefficients(), a.coefficients(), b.coefficients() );
    return r;
}

// TE / TE division via reciprocal.
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
constexpr TaylorExpansion< T, Scheme >& operator+=( TaylorExpansion< T, Scheme >& a,
                                                    const TaylorExpansion< T, Scheme >& b ) noexcept
{
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) a[k] += b[k];
    return a;
}

template < typename T, IndexScheme Scheme >
constexpr TaylorExpansion< T, Scheme >& operator-=( TaylorExpansion< T, Scheme >& a,
                                                    const TaylorExpansion< T, Scheme >& b ) noexcept
{
    for ( std::size_t k = 0; k < a.nCoefficients; ++k ) a[k] -= b[k];
    return a;
}

template < typename T, IndexScheme Scheme >
constexpr TaylorExpansion< T, Scheme >& operator+=( TaylorExpansion< T, Scheme >& a,
                                                    std::type_identity_t< T > s ) noexcept
{
    a[0] += s;
    return a;
}

template < typename T, IndexScheme Scheme >
constexpr TaylorExpansion< T, Scheme >& operator-=( TaylorExpansion< T, Scheme >& a,
                                                    std::type_identity_t< T > s ) noexcept
{
    a[0] -= s;
    return a;
}

template < typename T, IndexScheme Scheme >
constexpr TaylorExpansion< T, Scheme >& operator*=( TaylorExpansion< T, Scheme >& a,
                                                    std::type_identity_t< T > s ) noexcept
{
    for ( T& ak : a.coefficients() ) ak *= s;
    return a;
}

template < typename T, IndexScheme Scheme >
constexpr TaylorExpansion< T, Scheme >& operator/=( TaylorExpansion< T, Scheme >& a,
                                                    std::type_identity_t< T > s ) noexcept
{
    return a *= ( T( 1 ) / s );
}

/// In-place Cauchy product.
template < typename T, IndexScheme Scheme >
constexpr TaylorExpansion< T, Scheme >& operator*=( TaylorExpansion< T, Scheme >& a,
                                                    const TaylorExpansion< T, Scheme >& b ) noexcept
{
    std::array< T, Scheme::nCoeff > tmp{};
    tax::cauchyProduct< T, Scheme >( tmp, a.coefficients(), b.coefficients() );
    a.coefficients() = tmp;
    return a;
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

// Sparse storage overloads.

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

template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, Sparse > operator*(
    std::type_identity_t< T > s,
    const TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& a ) noexcept
{
    return a * s;
}

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

template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, Sparse > operator+(
    std::type_identity_t< T > s,
    const TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& a ) noexcept
{
    return a + s;
}

template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, Sparse > operator-(
    const TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& a,
    std::type_identity_t< T > s ) noexcept
{
    return a + ( -s );
}

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

// Sparse compound assignment. A sparse result reshapes the support, so these
// rebuild `a` from the corresponding binary operator rather than mutating in
// place; the binary forms already emit a sorted, zero-free container.

#define TAX_SPARSE_COMPOUND( OP )                                                               \
    template < typename T, int N, int M >                                                       \
    TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& operator OP##=(                      \
        TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& a,                               \
        const TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& b )                        \
    {                                                                                           \
        a = a OP b;                                                                             \
        return a;                                                                               \
    }                                                                                           \
    template < typename T, int N, int M >                                                       \
    TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& operator OP##=(                      \
        TaylorExpansion< T, IsotropicScheme< N, M >, Sparse >& a, std::type_identity_t< T > s ) \
    {                                                                                           \
        a = a OP s;                                                                             \
        return a;                                                                               \
    }

TAX_SPARSE_COMPOUND( +)
TAX_SPARSE_COMPOUND( -)
TAX_SPARSE_COMPOUND( * )
TAX_SPARSE_COMPOUND( / )

#undef TAX_SPARSE_COMPOUND

}  // namespace tax
