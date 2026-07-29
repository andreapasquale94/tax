#pragma once

#include <array>
#include <cstddef>
#include <tax/core/multi_index.hpp>
#include <utility>

namespace tax::storage
{

/// Dense coefficient container for a TaylorExpansion, sized by `Size` slots.
template < typename T, std::size_t Size >
struct DenseContainer
{
    using value_type = T;
    using coeffs_type = std::array< T, Size >;
    static constexpr std::size_t nCoefficients = Size;

    coeffs_type data{};

    [[nodiscard]] constexpr T value() const noexcept { return data[0]; }

    [[nodiscard]] constexpr T operator[]( std::size_t k ) const noexcept { return data[k]; }
    [[nodiscard]] constexpr T& operator[]( std::size_t k ) noexcept { return data[k]; }

    constexpr void set( std::size_t k, T v ) noexcept { data[k] = v; }
    constexpr void accumulate( std::size_t k, T v ) noexcept { data[k] += v; }

    /// Visit every coefficient slot in flat-index order: fn(k, data[k]).
    /// Conditionally noexcept: propagates the callable's own exception specification
    /// rather than calling std::terminate when a throwing `fn` is supplied.
    template < typename Fn >
    constexpr void forEach( Fn&& fn ) const
        noexcept( noexcept( fn( std::size_t{ 0 }, std::declval< T >() ) ) )
    {
        for ( std::size_t k = 0; k < nCoefficients; ++k ) fn( k, data[k] );
    }
};

}  // namespace tax::storage
