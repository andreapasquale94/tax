#pragma once

// Compile-time string + type-list primitives shared by the named and mixed axis layers.

#include <cstddef>

namespace tax::named
{

// ---------------------------------------------------------------------------
// FixedString — a structural compile-time string usable as an NTTP
// ---------------------------------------------------------------------------

/// A null-terminated compile-time string suitable as a non-type template parameter (e.g. `Axis<
/// "x", 3 >`).
template < std::size_t K >
struct FixedString
{
    char data[K]{};

    /*implicit*/ constexpr FixedString( const char ( &s )[K] ) noexcept
    {
        for ( std::size_t i = 0; i < K; ++i ) data[i] = s[i];
    }

    /// Length of the string excluding the terminating null.
    [[nodiscard]] static constexpr std::size_t size() noexcept { return K - 1; }

    [[nodiscard]] constexpr char operator[]( std::size_t i ) const noexcept { return data[i]; }
};

/// Three-way lexicographic comparison of two FixedStrings (-1 / 0 / 1).
template < std::size_t A, std::size_t B >
[[nodiscard]] constexpr int compareFixed( const FixedString< A >& a,
                                          const FixedString< B >& b ) noexcept
{
    const std::size_t la = a.size();
    const std::size_t lb = b.size();
    const std::size_t n = la < lb ? la : lb;
    for ( std::size_t i = 0; i < n; ++i )
    {
        // Compare as unsigned char: plain char signedness is implementation-defined,
        // so signed comparison would order non-ASCII axis names inconsistently and
        // could make the canonical merged-type ordering platform-dependent.
        const unsigned char ca = static_cast< unsigned char >( a[i] );
        const unsigned char cb = static_cast< unsigned char >( b[i] );
        if ( ca != cb ) return ca < cb ? -1 : 1;
    }
    if ( la == lb ) return 0;
    return la < lb ? -1 : 1;
}

/// Three-way comparison of two FixedString NTTP values (-1 / 0 / 1).
template < FixedString A, FixedString B >
inline constexpr int fixedCompare = compareFixed( A, B );

// ---------------------------------------------------------------------------
// Compile-time type-list primitives
// ---------------------------------------------------------------------------

namespace detail
{

/// A list of axis types.
template < typename... Ts >
struct TypeList
{
    static constexpr std::size_t size = sizeof...( Ts );
};

template < typename Head, typename List >
struct Prepend;
template < typename Head, typename... Ts >
struct Prepend< Head, TypeList< Ts... > >
{
    using type = TypeList< Head, Ts... >;
};

}  // namespace detail

}  // namespace tax::named
