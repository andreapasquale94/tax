#pragma once

// All sparse-storage TaylorExpansion operators (arithmetic + the sparse math
// overloads sqrt / reciprocal / pow). Consolidated here so the dense operator
// headers carry only dense overloads. Pure relocation — every overload is
// unchanged and stays in namespace tax, preserving ADL.

#include <cstddef>
#include <tax/expansion/detail/sparse_cauchy.hpp>
#include <tax/expansion/detail/sparse_subs.hpp>
#include <tax/expansion/expansion.hpp>
#include <tax/expansion/scheme/isotropic.hpp>
#include <tax/expansion/storage/sparse.hpp>

namespace tax
{

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

// ===========================================================================
// Sparse overloads: sqrt, reciprocal
// ===========================================================================

/// Sparse `sqrt(f)` via support-set forward substitution.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > sqrt(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x )
{
    TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > r;
    detail::kernels::seriesSqrtSparse< T, N, M >( r.container(), x.container() );
    return r;
}

/// Sparse `1/f` via support-set forward substitution.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > reciprocal(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x )
{
    TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > r;
    detail::kernels::seriesReciprocalSparse< T, N, M >( r.container(), x.container() );
    return r;
}

/// Sparse `f^n` via binary exponentiation of the Cauchy product.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > pow(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x, int n )
{
    TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > r;
    detail::kernels::seriesPowIntSparse< T, N, M >( r.container(), x.container(), n );
    return r;
}

}  // namespace tax
