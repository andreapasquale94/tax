#pragma once

#include <stdexcept>
#include <tax/model/taylor_model.hpp>
#include <tax/operators/arithmetic.hpp>
#include <type_traits>

namespace tax::model
{

namespace detail
{

template < std::floating_point T, int N, int M >
constexpr void checkCompatible( const TaylorModel< T, N, M >& a, const TaylorModel< T, N, M >& b )
{
    if ( !a.compatibleWith( b ) )
        throw std::invalid_argument(
            "tax::model: operands defined over different expansion points or domains" );
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Addition / subtraction (component-wise, thesis rule (4.5))
// ---------------------------------------------------------------------------

template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > operator+( const TaylorModel< T, N, M >& a,
                                                          const TaylorModel< T, N, M >& b )
{
    detail::checkCompatible( a, b );
    return { a.polynomial() + b.polynomial(), a.remainder() + b.remainder(), a.expansionPoint(),
             a.domain() };
}

template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > operator-( const TaylorModel< T, N, M >& a,
                                                          const TaylorModel< T, N, M >& b )
{
    detail::checkCompatible( a, b );
    return { a.polynomial() - b.polynomial(), a.remainder() - b.remainder(), a.expansionPoint(),
             a.domain() };
}

template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > operator-( const TaylorModel< T, N, M >& a )
{
    return { -a.polynomial(), -a.remainder(), a.expansionPoint(), a.domain() };
}

// ---------------------------------------------------------------------------
// Scalar addition / subtraction
// ---------------------------------------------------------------------------

template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > operator+( const TaylorModel< T, N, M >& a,
                                                          std::type_identity_t< T > s )
{
    return { a.polynomial() + s, a.remainder(), a.expansionPoint(), a.domain() };
}

template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > operator+( std::type_identity_t< T > s,
                                                          const TaylorModel< T, N, M >& a )
{
    return a + s;
}

template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > operator-( const TaylorModel< T, N, M >& a,
                                                          std::type_identity_t< T > s )
{
    return { a.polynomial() - s, a.remainder(), a.expansionPoint(), a.domain() };
}

template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > operator-( std::type_identity_t< T > s,
                                                          const TaylorModel< T, N, M >& a )
{
    return { s - a.polynomial(), -a.remainder(), a.expansionPoint(), a.domain() };
}

// ---------------------------------------------------------------------------
// Interval addition / subtraction: the interval is an unknown constant s in J.
// The midpoint goes into the polynomial; the residual widens the remainder.
// ---------------------------------------------------------------------------

template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > operator+( const TaylorModel< T, N, M >& a,
                                                          const Interval< T >& j )
{
    const T m = j.mid();
    return { a.polynomial() + m, a.remainder() + ( j - m ), a.expansionPoint(), a.domain() };
}

template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > operator+( const Interval< T >& j,
                                                          const TaylorModel< T, N, M >& a )
{
    return a + j;
}

template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > operator-( const TaylorModel< T, N, M >& a,
                                                          const Interval< T >& j )
{
    return a + ( -j );
}

template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > operator-( const Interval< T >& j,
                                                          const TaylorModel< T, N, M >& a )
{
    return ( -a ) + j;
}

// ---------------------------------------------------------------------------
// Scalar / interval multiplication and scalar division
// ---------------------------------------------------------------------------

template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > operator*( const TaylorModel< T, N, M >& a,
                                                          std::type_identity_t< T > s )
{
    return { a.polynomial() * s, a.remainder() * Interval< T >{ s }, a.expansionPoint(),
             a.domain() };
}

template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > operator*( std::type_identity_t< T > s,
                                                          const TaylorModel< T, N, M >& a )
{
    return a * s;
}

template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > operator/( const TaylorModel< T, N, M >& a,
                                                          std::type_identity_t< T > s )
{
    return { a.polynomial() / s, a.remainder() / Interval< T >{ s }, a.expansionPoint(),
             a.domain() };
}

/// s * f for an unknown constant s in J: the midpoint scales the polynomial;
/// the residual (J - m) * B(P) + J * I widens the remainder.
template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > operator*( const TaylorModel< T, N, M >& a,
                                                          const Interval< T >& j )
{
    const T m = j.mid();
    const Interval< T > rem = ( j - m ) * a.polynomialBound() + j * a.remainder();
    return { a.polynomial() * m, rem, a.expansionPoint(), a.domain() };
}

template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > operator*( const Interval< T >& j,
                                                          const TaylorModel< T, N, M >& a )
{
    return a * j;
}

// ---------------------------------------------------------------------------
// Taylor-model multiplication (§4.3 / §5.3.2)
//
// (P_a + e_a)(P_b + e_b) = trunc(P_a P_b)             -> polynomial part
//                        + excess(P_a P_b)            -> degree > N cross terms
//                        + P_a e_b + P_b e_a + e_a e_b
// with e_a in I_a, e_b in I_b and P_a(x) in B(P_a), P_b(x) in B(P_b).
// ---------------------------------------------------------------------------

template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > operator*( const TaylorModel< T, N, M >& a,
                                                          const TaylorModel< T, N, M >& b )
{
    detail::checkCompatible( a, b );
    const detail::DomainPowers< T, M, 2 * N > pows{ a.displacementDomain() };
    const Interval< T > excess = detail::excessProductBound( a.polynomial(), b.polynomial(), pows );
    const Interval< T > bound_a = detail::polyRangeBound( a.polynomial(), pows );
    const Interval< T > bound_b = detail::polyRangeBound( b.polynomial(), pows );
    const Interval< T > rem =
        excess + bound_a * b.remainder() + bound_b * a.remainder() + a.remainder() * b.remainder();
    return { a.polynomial() * b.polynomial(), rem, a.expansionPoint(), a.domain() };
}

// ---------------------------------------------------------------------------
// Compound assignment (division lives in <tax/model/math.hpp>)
// ---------------------------------------------------------------------------

#define TAX_MODEL_COMPOUND( OP )                                                                 \
    template < std::floating_point T, int N, int M, typename Rhs >                               \
    constexpr TaylorModel< T, N, M >& operator OP##=( TaylorModel< T, N, M >& a, const Rhs & b ) \
        requires requires( const TaylorModel< T, N, M >& x, const Rhs& y ) { x OP y; }           \
    {                                                                                            \
        return a = a OP b;                                                                       \
    }

TAX_MODEL_COMPOUND( +)
TAX_MODEL_COMPOUND( -)
TAX_MODEL_COMPOUND( * )

#undef TAX_MODEL_COMPOUND

template < std::floating_point T, int N, int M >
constexpr TaylorModel< T, N, M >& operator/=( TaylorModel< T, N, M >& a,
                                              std::type_identity_t< T > s )
{
    return a = a / s;
}

}  // namespace tax::model
