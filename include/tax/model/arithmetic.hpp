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

/// Reconcile a binary pair for the abstract-constant path: a domain-agnostic
/// constant adopts its partner's expansion point/domain; two concrete
/// operands must match. `abstract` is set when *both* are abstract, so the
/// result stays domain-agnostic.
template < std::floating_point T, int N, int M >
struct Reconciled
{
    TaylorModel< T, N, M > a;
    TaylorModel< T, N, M > b;
    bool abstract;
};

template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr Reconciled< T, N, M > reconcile( const TaylorModel< T, N, M >& a,
                                                         const TaylorModel< T, N, M >& b )
{
    const bool aa = a.isAbstractConstant();
    const bool bb = b.isAbstractConstant();
    if ( aa && bb ) return { a, b, true };
    if ( aa ) return { a.overDomain( b.expansionPoint(), b.domain() ), b, false };
    if ( bb ) return { a, b.overDomain( a.expansionPoint(), a.domain() ), false };
    checkCompatible( a, b );
    return { a, b, false };
}

/// Propagate the domain-agnostic flag of `src` onto a result: an operation
/// between an abstract constant and a scalar/interval stays a constant.
template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > keepAbstract(
    TaylorModel< T, N, M > r, const TaylorModel< T, N, M >& src ) noexcept
{
    return src.isAbstractConstant() ? r.asAbstractConstant() : r;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Addition / subtraction (component-wise, thesis rule (4.5))
// ---------------------------------------------------------------------------

template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > operator+( const TaylorModel< T, N, M >& a,
                                                          const TaylorModel< T, N, M >& b )
{
    if ( a.isAbstractConstant() || b.isAbstractConstant() )
    {
        const auto rc = detail::reconcile( a, b );
        TaylorModel< T, N, M > r{ rc.a.polynomial() + rc.b.polynomial(),
                                  rc.a.remainder() + rc.b.remainder(), rc.a.expansionPoint(),
                                  rc.a.domain() };
        return rc.abstract ? r.asAbstractConstant() : r;
    }
    detail::checkCompatible( a, b );
    return { a.polynomial() + b.polynomial(), a.remainder() + b.remainder(), a.expansionPoint(),
             a.domain() };
}

template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > operator-( const TaylorModel< T, N, M >& a,
                                                          const TaylorModel< T, N, M >& b )
{
    if ( a.isAbstractConstant() || b.isAbstractConstant() )
    {
        const auto rc = detail::reconcile( a, b );
        TaylorModel< T, N, M > r{ rc.a.polynomial() - rc.b.polynomial(),
                                  rc.a.remainder() - rc.b.remainder(), rc.a.expansionPoint(),
                                  rc.a.domain() };
        return rc.abstract ? r.asAbstractConstant() : r;
    }
    detail::checkCompatible( a, b );
    return { a.polynomial() - b.polynomial(), a.remainder() - b.remainder(), a.expansionPoint(),
             a.domain() };
}

template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > operator-( const TaylorModel< T, N, M >& a )
{
    TaylorModel< T, N, M > r{ -a.polynomial(), -a.remainder(), a.expansionPoint(), a.domain() };
    return a.isAbstractConstant() ? r.asAbstractConstant() : r;
}

// ---------------------------------------------------------------------------
// Scalar addition / subtraction
// ---------------------------------------------------------------------------

template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > operator+( const TaylorModel< T, N, M >& a,
                                                          std::type_identity_t< T > s )
{
    return detail::keepAbstract< T, N, M >(
        { a.polynomial() + s, a.remainder(), a.expansionPoint(), a.domain() }, a );
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
    return detail::keepAbstract< T, N, M >(
        { a.polynomial() - s, a.remainder(), a.expansionPoint(), a.domain() }, a );
}

template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > operator-( std::type_identity_t< T > s,
                                                          const TaylorModel< T, N, M >& a )
{
    return detail::keepAbstract< T, N, M >(
        { s - a.polynomial(), -a.remainder(), a.expansionPoint(), a.domain() }, a );
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
    return detail::keepAbstract< T, N, M >(
        { a.polynomial() + m, a.remainder() + ( j - m ), a.expansionPoint(), a.domain() }, a );
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
    return detail::keepAbstract< T, N, M >(
        { a.polynomial() * s, a.remainder() * Interval< T >{ s }, a.expansionPoint(), a.domain() },
        a );
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
    return detail::keepAbstract< T, N, M >(
        { a.polynomial() / s, a.remainder() / Interval< T >{ s }, a.expansionPoint(), a.domain() },
        a );
}

/// s * f for an unknown constant s in J: the midpoint scales the polynomial;
/// the residual (J - m) * B(P) + J * I widens the remainder.
template < std::floating_point T, int N, int M >
[[nodiscard]] constexpr TaylorModel< T, N, M > operator*( const TaylorModel< T, N, M >& a,
                                                          const Interval< T >& j )
{
    const T m = j.mid();
    const Interval< T > rem = ( j - m ) * a.polynomialBound() + j * a.remainder();
    return detail::keepAbstract< T, N, M >(
        { a.polynomial() * m, rem, a.expansionPoint(), a.domain() }, a );
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
    // Abstract-constant fast paths: a constant scales the other operand
    // exactly (no domain interaction), which keeps Eigen's scalar-times-matrix
    // products cheap and domain-agnostic.
    if ( a.isAbstractConstant() && b.isAbstractConstant() )
        return TaylorModel< T, N, M >{ a.value() * b.value() };
    if ( a.isAbstractConstant() ) return b * a.value();  // scalar * model
    if ( b.isAbstractConstant() ) return a * b.value();
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
