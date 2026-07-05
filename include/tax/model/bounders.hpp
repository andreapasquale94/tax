#pragma once

#include <array>
#include <cstddef>
#include <tax/core/multi_index.hpp>
#include <tax/core/scheme/isotropic.hpp>
#include <tax/core/storage/dense.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/model/interval.hpp>

namespace tax::model
{

/// Polynomial range-bounding strategy for `TaylorModel::bound()` /
/// `polynomialBound()`. See thesis §5.4 for the algorithm ladder.
enum class Bounder
{
    /// Order-sum of the monomial enclosures (thesis "algorithm 0/1"). Exact
    /// for orders 0-1; cheap; always a valid enclosure.
    Naive,
    /// Diagonal exact-quadratic tightening (a rigorous subset of thesis
    /// §5.4.3): each variable's `q_ii*h_i^2 + g_i*h_i` is bounded exactly by
    /// the 1-D vertex analysis, cross terms and orders >= 3 stay naive, and
    /// the result is intersected with the naive bound so it can never be
    /// wider. This is the default for user-facing bounds.
    Quadratic
};

namespace detail
{

// ===========================================================================
// Shared bounding machinery (used by TaylorModel and the operator layer).
// ===========================================================================

/// Compile-time table mapping flat index k to its multi-index.
template < int N, int M >
struct MultiIndexTable
{
    static constexpr std::size_t size = numMonomials( N, M );
    std::array< MultiIndex< M >, size > value{};

    constexpr MultiIndexTable() noexcept
    {
        for ( std::size_t k = 0; k < size; ++k ) value[k] = unflatIndex< M >( k );
    }
};

/// Interval powers of the displacement domain: pw[i][j] encloses D_i^j.
/// Each entry is computed by `pow` directly (not by repeated multiplication)
/// so even powers keep the sharp non-negative lower bound of (5.4).
template < std::floating_point T, int M, int P >
struct DomainPowers
{
    std::array< std::array< Interval< T >, std::size_t( P ) + 1 >, std::size_t( M ) > pw{};

    explicit constexpr DomainPowers( const std::array< Interval< T >, std::size_t( M ) >& D )
    {
        for ( std::size_t i = 0; i < std::size_t( M ); ++i )
        {
            for ( int j = 0; j <= P; ++j ) pw[i][std::size_t( j )] = model::pow( D[i], j );
        }
    }
};

/// Naive range bound B(P) of a dense polynomial over the displacement box:
/// the interval sum of the monomial enclosures c_alpha * D^alpha. Exact for
/// the constant and linear parts (interval evaluation of a linear function
/// over a box is exact); the per-order bounds I^k of higher orders are the
/// "no tightening" sums of §5.4.
template < std::floating_point T, int N, int M, int P >
[[nodiscard]] constexpr Interval< T > polyRangeBound(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Dense >& p,
    const DomainPowers< T, M, P >& pows ) noexcept
    requires( P >= N )
{
    constexpr auto kIdx = MultiIndexTable< N, M >{};
    Interval< T > r{ p[0] };
    for ( std::size_t k = 1; k < numMonomials( N, M ); ++k )
    {
        if ( p[k] == T{ 0 } ) continue;
        Interval< T > term{ p[k] };
        const auto& alpha = kIdx.value[k];
        for ( std::size_t i = 0; i < std::size_t( M ); ++i )
        {
            const int e = alpha[i];
            if ( e != 0 ) term = term * pows.pw[i][std::size_t( e )];
        }
        r = r + term;
    }
    return r;
}

/// Range bound of the homogeneous degree-`deg` part of `p` (the per-order
/// bound I^deg): graded-lex keeps each degree in one contiguous block.
template < std::floating_point T, int N, int M, int P >
[[nodiscard]] constexpr Interval< T > orderRangeBound(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Dense >& p, int deg,
    const DomainPowers< T, M, P >& pows ) noexcept
    requires( P >= N )
{
    constexpr auto kIdx = MultiIndexTable< N, M >{};
    if ( deg < 0 || deg > N ) return {};
    if ( deg == 0 ) return Interval< T >{ p[0] };
    Interval< T > r{};
    for ( std::size_t k = numMonomials( deg - 1, M ); k < numMonomials( deg, M ); ++k )
    {
        if ( p[k] == T{ 0 } ) continue;
        Interval< T > term{ p[k] };
        const auto& alpha = kIdx.value[k];
        for ( std::size_t i = 0; i < std::size_t( M ); ++i )
        {
            const int e = alpha[i];
            if ( e != 0 ) term = term * pows.pw[i][std::size_t( e )];
        }
        r = r + term;
    }
    return r;
}

/// Bound of the order-(> N) excess of the product a * b over the domain:
/// the degree-(> N) cross terms of the Cauchy convolution that the truncated
/// polynomial product drops, folded into the remainder (§4.3 / §5.3.2).
template < std::floating_point T, int N, int M, int P >
[[nodiscard]] constexpr Interval< T > excessProductBound(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Dense >& a,
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Dense >& b,
    const DomainPowers< T, M, P >& pows ) noexcept
    requires( P >= 2 * N )
{
    constexpr auto kIdx = MultiIndexTable< N, M >{};
    constexpr auto kDeg = DegreeOf< N, M >{};
    constexpr std::size_t n_coeff = numMonomials( N, M );

    Interval< T > r{};
    for ( std::size_t i = 0; i < n_coeff; ++i )
    {
        if ( a[i] == T{ 0 } ) continue;
        const int da = kDeg.value[i];
        if ( da == 0 ) continue;  // partner degree would need to exceed N
        // Partners of degree db with da + db > N start the contiguous block
        // of degree N - da + 1.
        for ( std::size_t j = numMonomials( N - da, M ); j < n_coeff; ++j )
        {
            if ( b[j] == T{ 0 } ) continue;
            Interval< T > term = Interval< T >{ a[i] } * Interval< T >{ b[j] };
            const auto& alpha = kIdx.value[i];
            const auto& beta = kIdx.value[j];
            for ( std::size_t v = 0; v < std::size_t( M ); ++v )
            {
                const int e = alpha[v] + beta[v];
                if ( e != 0 ) term = term * pows.pw[v][std::size_t( e )];
            }
            r = r + term;
        }
    }
    return r;
}

// ===========================================================================
// Exact quadratic bounder (thesis §5.4.3, diagonal form)
// ===========================================================================

/// Rigorous enclosure of the univariate quadratic `a2*h^2 + a1*h` over the
/// displacement interval `D` (the constant term is handled by the caller).
///
/// Endpoint values give the monotone case exactly; when the parabola's
/// vertex `h* = -a1/(2 a2)` may lie inside `D`, the vertex value
/// `-a1^2/(4 a2)` — an actually attained value — is folded in. Including it
/// when the vertex is (interval-uncertainly) outside `D` only widens the
/// enclosure, never invalidates it.
template < std::floating_point T >
[[nodiscard]] constexpr Interval< T > boundDiagonalQuadratic( T a2, T a1,
                                                              const Interval< T >& D ) noexcept
{
    const Interval< T > lo{ D.lower() };
    const Interval< T > hi{ D.upper() };
    const Interval< T > a2i{ a2 };
    const Interval< T > a1i{ a1 };
    const Interval< T > f_lo = a2i * lo * lo + a1i * lo;
    const Interval< T > f_hi = a2i * hi * hi + a1i * hi;
    Interval< T > r = hull( f_lo, f_hi );
    if ( a2 == T{ 0 } ) return r;  // linear: endpoints are exact

    // Vertex location h* = -a1 / (2 a2), as an enclosure.
    const Interval< T > h_star = Interval< T >{ -a1 } / ( Interval< T >{ T{ 2 } } * a2i );
    if ( h_star.upper() >= D.lower() && h_star.lower() <= D.upper() )
    {
        // Vertex value -a1^2 / (4 a2); a genuinely attained extremum value.
        const Interval< T > v = Interval< T >{ -a1 } * a1i / ( Interval< T >{ T{ 4 } } * a2i );
        r = hull( r, v );
    }
    return r;
}

/// Diagonal exact-quadratic range bound (thesis §5.4.3). Bounds each
/// variable's `q_ii*h_i^2 + g_i*h_i` exactly, keeps cross terms and orders
/// >= 3 on the naive order-sum, and intersects with the naive bound so the
/// result is always a valid enclosure and never wider than `polyRangeBound`.
template < std::floating_point T, int N, int M, int P >
[[nodiscard]] constexpr Interval< T > quadraticRangeBound(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Dense >& p,
    const DomainPowers< T, M, P >& pows,
    const std::array< Interval< T >, std::size_t( M ) >& disp ) noexcept
    requires( P >= N )
{
    const Interval< T > naive = polyRangeBound( p, pows );
    if constexpr ( N < 2 )
    {
        return naive;  // orders 0-1 are already exact
    } else
    {
        Interval< T > acc{ p[0] };  // constant term

        // Diagonal quadratic + linear, treated exactly per variable.
        for ( int i = 0; i < M; ++i )
        {
            MultiIndex< M > lin{};
            lin[std::size_t( i )] = 1;
            MultiIndex< M > quad{};
            quad[std::size_t( i )] = 2;
            const T g = p[flatIndex< M >( lin )];
            const T q = p[flatIndex< M >( quad )];
            acc = acc + boundDiagonalQuadratic( q, g, disp[std::size_t( i )] );
        }

        // Order-2 cross terms h_i*h_j (i < j): exact via the interval product
        // of the two independent domains.
        for ( int i = 0; i < M; ++i )
        {
            for ( int j = i + 1; j < M; ++j )
            {
                MultiIndex< M > mix{};
                mix[std::size_t( i )] = 1;
                mix[std::size_t( j )] = 1;
                const T c = p[flatIndex< M >( mix )];
                if ( c != T{ 0 } )
                    acc =
                        acc + Interval< T >{ c } * disp[std::size_t( i )] * disp[std::size_t( j )];
            }
        }

        // Orders >= 3: naive per-order sums.
        for ( int deg = 3; deg <= N; ++deg ) acc = acc + orderRangeBound( p, deg, pows );

        // Both `naive` and `acc` are valid enclosures of the same range, so
        // their intersection is valid and no wider than either. Fall back to
        // naive on the (numerically impossible) empty case.
        const T lo = naive.lower() > acc.lower() ? naive.lower() : acc.lower();
        const T hi = naive.upper() < acc.upper() ? naive.upper() : acc.upper();
        if ( lo > hi ) return naive;
        return Interval< T >{ lo, hi };
    }
}

/// Dispatch a range bound over the chosen strategy.
template < std::floating_point T, int N, int M, int P >
[[nodiscard]] constexpr Interval< T > rangeBound(
    Bounder which, const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Dense >& p,
    const DomainPowers< T, M, P >& pows,
    const std::array< Interval< T >, std::size_t( M ) >& disp ) noexcept
    requires( P >= N )
{
    if ( which == Bounder::Quadratic ) return quadraticRangeBound( p, pows, disp );
    return polyRangeBound( p, pows );
}

}  // namespace detail
}  // namespace tax::model
