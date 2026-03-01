#pragma once

#include <tax/utils/fwd.hpp>
#include <array>
#include <ostream>
#include <string_view>

namespace tax::detail
{

inline constexpr std::array< std::string_view, 10 > superscript_digits{
    "⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹" };
inline constexpr std::array< std::string_view, 10 > subscript_digits{
    "₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉" };

inline void writeMappedDigits( std::ostream& os, int value,
                               const std::array< std::string_view, 10 >& digits )
{
    char buf[16];
    int n = 0;
    do
    {
        buf[n++] = char( '0' + ( value % 10 ) );
        value /= 10;
    } while ( value > 0 );
    for ( int i = n - 1; i >= 0; --i ) os << digits[buf[i] - '0'];
}

template < int M >
inline void writeMonomial( std::ostream& os, const MultiIndex< M >& alpha )
{
    bool wrote_factor = false;
    for ( int i = 0; i < M; ++i )
    {
        const int pow = alpha[i];
        if ( pow == 0 ) continue;
        if ( wrote_factor ) os << "·";
        if constexpr ( M == 1 )
        {
            os << "dt";
        } else
        {
            os << "dx";
            writeMappedDigits( os, i, subscript_digits );
        }
        if ( pow > 1 ) writeMappedDigits( os, pow, superscript_digits );
        wrote_factor = true;
    }
}

template < int M >
inline void writeTruncationRemainder( std::ostream& os, int order )
{
    if constexpr ( M == 1 )
    {
        os << "O(dt";
        writeMappedDigits( os, order, superscript_digits );
        os << ')';
    } else
    {
        os << "O(||dx||";
        writeMappedDigits( os, order, superscript_digits );
        os << ')';
    }
}

}  // namespace tax::detail
