#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <numbers>
#include <ostream>
#include <stdexcept>
#include <type_traits>

namespace tax::model
{

// ===========================================================================
// Outward rounding primitives
//
// Interval arithmetic is only rigorous if every computed endpoint is rounded
// *away* from the true result. Instead of toggling the FPU rounding mode
// (non-constexpr, thread-hostile), every operation below evaluates in the
// default round-to-nearest mode and then widens the result by one ulp per
// side: since round-to-nearest is within 1/2 ulp of the exact value, the
// widened interval is a guaranteed enclosure. Transcendental endpoints
// (libm) are widened by two ulps, assuming a faithful (<= 1 ulp) libm.
// ===========================================================================

namespace detail
{

template < std::floating_point T >
constexpr T nextUp( T x ) noexcept
{
    if ( x != x || x == std::numeric_limits< T >::infinity() ) return x;
    if ( x == T{ 0 } ) return std::numeric_limits< T >::denorm_min();
    using U = std::conditional_t< sizeof( T ) == 8, std::uint64_t, std::uint32_t >;
    U bits = std::bit_cast< U >( x );
    constexpr U sign_mask = U{ 1 } << ( sizeof( T ) * 8 - 1 );
    if ( bits & sign_mask )
        --bits;
    else
        ++bits;
    return std::bit_cast< T >( bits );
}

template < std::floating_point T >
constexpr T nextDown( T x ) noexcept
{
    return -nextUp( -x );
}

}  // namespace detail

// ===========================================================================
// Interval
// ===========================================================================

/// Closed interval [lo, hi] with outward-rounded arithmetic (Makino §4.2).
///
/// All arithmetic follows Table 4.1 of the thesis. Endpoints given to the
/// constructor are taken exactly; endpoints *computed* by any operation are
/// widened outward (see above), so every operation returns a rigorous
/// enclosure of the exact-arithmetic result. NaN endpoints are not supported.
template < std::floating_point T >
class Interval
{
   public:
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// The degenerate interval [0, 0].
    constexpr Interval() noexcept = default;

    /// Degenerate point interval [v, v].
    /*implicit*/ constexpr Interval( T v ) noexcept : lo_{ v }, hi_{ v } {}

    /// Interval [lo, hi]. Throws std::invalid_argument if lo > hi.
    constexpr Interval( T lo, T hi ) : lo_{ lo }, hi_{ hi }
    {
        if ( lo > hi ) throw std::invalid_argument( "tax::model::Interval: lo > hi" );
    }

    [[nodiscard]] static constexpr Interval zero() noexcept { return {}; }

    /// Enclosure of `v` widened outward by `ulps` ulps per side.
    [[nodiscard]] static constexpr Interval padded( T v, int ulps = 1 ) noexcept
    {
        T lo = v, hi = v;
        for ( int i = 0; i < ulps; ++i )
        {
            lo = detail::nextDown( lo );
            hi = detail::nextUp( hi );
        }
        Interval r;
        r.lo_ = lo;
        r.hi_ = hi;
        return r;
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    [[nodiscard]] constexpr T lower() const noexcept { return lo_; }
    [[nodiscard]] constexpr T upper() const noexcept { return hi_; }

    /// Approximate midpoint (not necessarily the exact midpoint; any point
    /// of the interval is acceptable wherever mid() is used internally).
    [[nodiscard]] constexpr T mid() const noexcept { return ( lo_ + hi_ ) / T{ 2 }; }

    /// Upper bound of the width hi - lo.
    [[nodiscard]] constexpr T width() const noexcept { return detail::nextUp( hi_ - lo_ ); }

    /// Magnitude: max |x| over the interval.
    [[nodiscard]] constexpr T mag() const noexcept
    {
        const T a = lo_ < T{ 0 } ? -lo_ : lo_;
        const T b = hi_ < T{ 0 } ? -hi_ : hi_;
        return a > b ? a : b;
    }

    /// Mignitude: min |x| over the interval (0 if the interval contains 0).
    [[nodiscard]] constexpr T mig() const noexcept
    {
        if ( contains( T{ 0 } ) ) return T{ 0 };
        const T a = lo_ < T{ 0 } ? -lo_ : lo_;
        const T b = hi_ < T{ 0 } ? -hi_ : hi_;
        return a < b ? a : b;
    }

    [[nodiscard]] constexpr bool contains( T v ) const noexcept { return lo_ <= v && v <= hi_; }

    [[nodiscard]] constexpr bool contains( const Interval& other ) const noexcept
    {
        return lo_ <= other.lo_ && other.hi_ <= hi_;
    }

    [[nodiscard]] constexpr bool operator==( const Interval& ) const noexcept = default;

    // ------------------------------------------------------------------
    // Arithmetic (Table 4.1), outward-rounded
    // ------------------------------------------------------------------

    [[nodiscard]] constexpr Interval operator-() const noexcept
    {
        Interval r;
        r.lo_ = -hi_;
        r.hi_ = -lo_;
        return r;
    }

    [[nodiscard]] friend constexpr Interval operator+( const Interval& a,
                                                       const Interval& b ) noexcept
    {
        return outward( a.lo_ + b.lo_, a.hi_ + b.hi_ );
    }

    [[nodiscard]] friend constexpr Interval operator-( const Interval& a,
                                                       const Interval& b ) noexcept
    {
        return outward( a.lo_ - b.hi_, a.hi_ - b.lo_ );
    }

    [[nodiscard]] friend constexpr Interval operator*( const Interval& a,
                                                       const Interval& b ) noexcept
    {
        const T p1 = a.lo_ * b.lo_;
        const T p2 = a.lo_ * b.hi_;
        const T p3 = a.hi_ * b.lo_;
        const T p4 = a.hi_ * b.hi_;
        return outward( std::min( { p1, p2, p3, p4 } ), std::max( { p1, p2, p3, p4 } ) );
    }

    /// Division. Throws std::domain_error if 0 is contained in the divisor.
    [[nodiscard]] friend constexpr Interval operator/( const Interval& a, const Interval& b )
    {
        if ( b.contains( T{ 0 } ) )
            throw std::domain_error( "tax::model::Interval: division by interval containing 0" );
        const T q1 = a.lo_ / b.lo_;
        const T q2 = a.lo_ / b.hi_;
        const T q3 = a.hi_ / b.lo_;
        const T q4 = a.hi_ / b.hi_;
        return outward( std::min( { q1, q2, q3, q4 } ), std::max( { q1, q2, q3, q4 } ) );
    }

    // Mixed interval/scalar arithmetic — routed through the point interval.
    [[nodiscard]] friend constexpr Interval operator+( const Interval& a,
                                                       std::type_identity_t< T > s ) noexcept
    {
        return a + Interval{ s };
    }
    [[nodiscard]] friend constexpr Interval operator+( std::type_identity_t< T > s,
                                                       const Interval& a ) noexcept
    {
        return Interval{ s } + a;
    }
    [[nodiscard]] friend constexpr Interval operator-( const Interval& a,
                                                       std::type_identity_t< T > s ) noexcept
    {
        return a - Interval{ s };
    }
    [[nodiscard]] friend constexpr Interval operator-( std::type_identity_t< T > s,
                                                       const Interval& a ) noexcept
    {
        return Interval{ s } - a;
    }
    [[nodiscard]] friend constexpr Interval operator*( const Interval& a,
                                                       std::type_identity_t< T > s ) noexcept
    {
        return a * Interval{ s };
    }
    [[nodiscard]] friend constexpr Interval operator*( std::type_identity_t< T > s,
                                                       const Interval& a ) noexcept
    {
        return Interval{ s } * a;
    }
    [[nodiscard]] friend constexpr Interval operator/( const Interval& a,
                                                       std::type_identity_t< T > s )
    {
        return a / Interval{ s };
    }
    [[nodiscard]] friend constexpr Interval operator/( std::type_identity_t< T > s,
                                                       const Interval& a )
    {
        return Interval{ s } / a;
    }

    constexpr Interval& operator+=( const Interval& o ) noexcept { return *this = *this + o; }
    constexpr Interval& operator-=( const Interval& o ) noexcept { return *this = *this - o; }
    constexpr Interval& operator*=( const Interval& o ) noexcept { return *this = *this * o; }
    constexpr Interval& operator/=( const Interval& o ) { return *this = *this / o; }

    // ------------------------------------------------------------------
    // Set operations
    // ------------------------------------------------------------------

    /// Smallest interval containing both operands.
    [[nodiscard]] friend constexpr Interval hull( const Interval& a, const Interval& b ) noexcept
    {
        Interval r;
        r.lo_ = a.lo_ < b.lo_ ? a.lo_ : b.lo_;
        r.hi_ = a.hi_ > b.hi_ ? a.hi_ : b.hi_;
        return r;
    }

    /// Intersection. Throws std::domain_error if the operands are disjoint.
    [[nodiscard]] friend constexpr Interval intersect( const Interval& a, const Interval& b )
    {
        const T lo = a.lo_ > b.lo_ ? a.lo_ : b.lo_;
        const T hi = a.hi_ < b.hi_ ? a.hi_ : b.hi_;
        if ( lo > hi )
            throw std::domain_error( "tax::model::Interval: intersect of disjoint intervals" );
        Interval r;
        r.lo_ = lo;
        r.hi_ = hi;
        return r;
    }

    friend std::ostream& operator<<( std::ostream& os, const Interval& x )
    {
        return os << "[" << x.lo_ << ", " << x.hi_ << "]";
    }

    /// Endpoints widened by one ulp per side — for internal use by the
    /// elementary-function enclosures below.
    [[nodiscard]] static constexpr Interval outward( T lo, T hi ) noexcept
    {
        Interval r;
        r.lo_ = detail::nextDown( lo );
        r.hi_ = detail::nextUp( hi );
        return r;
    }

   private:
    T lo_{ 0 };
    T hi_{ 0 };
};

// ===========================================================================
// Powers
// ===========================================================================

namespace detail
{

/// Rigorous enclosure of the integer power v^n of a point, n >= 1, by
/// outward-rounded binary exponentiation.
template < std::floating_point T >
[[nodiscard]] constexpr Interval< T > pointPow( T v, int n ) noexcept
{
    Interval< T > r{ T{ 1 } };
    Interval< T > base{ v };
    while ( n > 0 )
    {
        if ( n & 1 ) r = r * base;
        base = base * base;
        n >>= 1;
    }
    return r;
}

}  // namespace detail

/// Interval square with the sharp rule (5.4): never dips below 0.
template < std::floating_point T >
[[nodiscard]] constexpr Interval< T > sqr( const Interval< T >& x ) noexcept
{
    const Interval< T > lo2 = detail::pointPow( x.lower(), 2 );
    const Interval< T > hi2 = detail::pointPow( x.upper(), 2 );
    const T hi = std::max( lo2.upper(), hi2.upper() );
    // The zero lower bound is exact — do not pad it outward.
    if ( x.contains( T{ 0 } ) ) return Interval< T >{ T{ 0 }, hi };
    return Interval< T >{ std::min( lo2.lower(), hi2.lower() ), hi };
}

/// Interval integer power. Even powers use the sharp (5.4)-style rule; odd
/// powers are monotone. Negative exponents divide (throws if 0 is inside).
template < std::floating_point T >
[[nodiscard]] constexpr Interval< T > pow( const Interval< T >& x, int n )
{
    if ( n < 0 ) return Interval< T >{ T{ 1 } } / pow( x, -n );
    if ( n == 0 ) return Interval< T >{ T{ 1 } };
    if ( n == 1 ) return x;
    if ( n % 2 != 0 )
    {
        return Interval< T >::outward( detail::pointPow( x.lower(), n ).lower(),
                                       detail::pointPow( x.upper(), n ).upper() );
    }
    const T hi = detail::pointPow( x.mag(), n ).upper();
    // Even powers: the zero lower bound is exact — do not pad it outward.
    if ( x.contains( T{ 0 } ) ) return Interval< T >{ T{ 0 }, hi };
    return Interval< T >{ detail::pointPow( x.mig(), n ).lower(), hi };
}

// ===========================================================================
// Elementary function enclosures (runtime only — they evaluate libm at the
// endpoints and widen by two ulps; see the header comment for the rounding
// contract).
// ===========================================================================

namespace detail
{

template < std::floating_point T >
inline constexpr Interval< T > kPi = Interval< T >::padded( std::numbers::pi_v< T > );

/// True if some integer lies in the enclosure `t` (conservative: may report
/// true on the boundary of representability, never false negatives).
template < std::floating_point T >
[[nodiscard]] inline bool containsInteger( const Interval< T >& t ) noexcept
{
    return std::floor( t.upper() ) >= std::ceil( t.lower() );
}

}  // namespace detail

/// exp over an interval (monotone).
template < std::floating_point T >
[[nodiscard]] inline Interval< T > exp( const Interval< T >& x ) noexcept
{
    const T lo = Interval< T >::padded( std::exp( x.lower() ), 2 ).lower();
    const T hi = Interval< T >::padded( std::exp( x.upper() ), 2 ).upper();
    // Monotone clamp of both endpoints: exp is positive.
    return Interval< T >{ std::max( lo, T{ 0 } ), std::max( hi, T{ 0 } ) };
}

/// log over an interval (monotone). Throws std::domain_error unless x > 0.
template < std::floating_point T >
[[nodiscard]] inline Interval< T > log( const Interval< T >& x )
{
    if ( !( x.lower() > T{ 0 } ) )
        throw std::domain_error( "tax::model::log(Interval): argument must be positive" );
    return Interval< T >{ Interval< T >::padded( std::log( x.lower() ), 2 ).lower(),
                          Interval< T >::padded( std::log( x.upper() ), 2 ).upper() };
}

/// sqrt over an interval (monotone; IEEE sqrt is correctly rounded, 1 ulp
/// padding suffices). Throws std::domain_error if x has negative points.
template < std::floating_point T >
[[nodiscard]] inline Interval< T > sqrt( const Interval< T >& x )
{
    if ( x.lower() < T{ 0 } )
        throw std::domain_error( "tax::model::sqrt(Interval): argument must be non-negative" );
    const T lo = detail::nextDown( std::sqrt( x.lower() ) );
    const T hi = detail::nextUp( std::sqrt( x.upper() ) );
    return Interval< T >{ std::max( lo, T{ 0 } ), hi };
}

/// sinh over an interval (monotone).
template < std::floating_point T >
[[nodiscard]] inline Interval< T > sinh( const Interval< T >& x ) noexcept
{
    return Interval< T >{ Interval< T >::padded( std::sinh( x.lower() ), 2 ).lower(),
                          Interval< T >::padded( std::sinh( x.upper() ), 2 ).upper() };
}

/// cosh over an interval (even, minimum 1 at 0).
template < std::floating_point T >
[[nodiscard]] inline Interval< T > cosh( const Interval< T >& x ) noexcept
{
    const T hi = Interval< T >::padded( std::cosh( x.mag() ), 2 ).upper();
    const T lo =
        x.contains( T{ 0 } ) ? T{ 1 } : Interval< T >::padded( std::cosh( x.mig() ), 2 ).lower();
    // Monotone clamp of both endpoints: cosh >= 1.
    return Interval< T >{ std::max( lo, T{ 1 } ), std::max( hi, T{ 1 } ) };
}

/// sin over an interval: checks whether a maximum (pi/2 + 2 pi k) or minimum
/// (-pi/2 + 2 pi k) is enclosed; otherwise takes padded endpoint values.
template < std::floating_point T >
[[nodiscard]] inline Interval< T > sin( const Interval< T >& x ) noexcept
{
    const Interval< T > two_pi = T{ 2 } * detail::kPi< T >;
    const Interval< T > half_pi = detail::kPi< T > / T{ 2 };
    if ( !( x.width() < two_pi.upper() ) ) return Interval< T >{ T{ -1 }, T{ 1 } };

    const Interval< T > slo = Interval< T >::padded( std::sin( x.lower() ), 2 );
    const Interval< T > shi = Interval< T >::padded( std::sin( x.upper() ), 2 );
    T hi = std::max( slo.upper(), shi.upper() );
    T lo = std::min( slo.lower(), shi.lower() );
    if ( detail::containsInteger( ( x - half_pi ) / two_pi ) ) hi = T{ 1 };
    if ( detail::containsInteger( ( x + half_pi ) / two_pi ) ) lo = T{ -1 };
    // Monotone clamp of both endpoints into [-1, 1] (keeps lo <= hi).
    const auto clamp = []( T v ) { return std::min( std::max( v, T{ -1 } ), T{ 1 } ); };
    return Interval< T >{ clamp( lo ), clamp( hi ) };
}

/// cos over an interval: checks whether a maximum (2 pi k) or minimum
/// (pi + 2 pi k) is enclosed; otherwise takes padded endpoint values.
template < std::floating_point T >
[[nodiscard]] inline Interval< T > cos( const Interval< T >& x ) noexcept
{
    const Interval< T > two_pi = T{ 2 } * detail::kPi< T >;
    if ( !( x.width() < two_pi.upper() ) ) return Interval< T >{ T{ -1 }, T{ 1 } };

    const Interval< T > clo = Interval< T >::padded( std::cos( x.lower() ), 2 );
    const Interval< T > chi = Interval< T >::padded( std::cos( x.upper() ), 2 );
    T hi = std::max( clo.upper(), chi.upper() );
    T lo = std::min( clo.lower(), chi.lower() );
    if ( detail::containsInteger( x / two_pi ) ) hi = T{ 1 };
    if ( detail::containsInteger( (x - detail::kPi< T >) / two_pi ) ) lo = T{ -1 };
    // Monotone clamp of both endpoints into [-1, 1] (keeps lo <= hi).
    const auto clamp = []( T v ) { return std::min( std::max( v, T{ -1 } ), T{ 1 } ); };
    return Interval< T >{ clamp( lo ), clamp( hi ) };
}

}  // namespace tax::model
