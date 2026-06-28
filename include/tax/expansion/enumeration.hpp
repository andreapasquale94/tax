#pragma once

#include <tax/expansion/multi_index.hpp>

namespace tax
{

/// Call `func(alpha)` for every multi-index of total degree exactly `degree`, in graded-lex order.
template < int M, typename Func >
constexpr void forEachMonomialOfDegree( int degree, Func&& func )
{
    MultiIndex< M > alpha{};
    auto fill = [&]( auto& self, int var, int rem ) constexpr -> void {
        if ( var == M - 1 )
        {
            alpha[std::size_t( var )] = rem;
            func( alpha );
            return;
        }
        for ( int k = rem; k >= 0; --k )
        {
            alpha[std::size_t( var )] = k;
            self( self, var + 1, rem - k );
        }
    };
    fill( fill, 0, degree );
}

/// Call `func(alpha)` for every multi-index of total degree in `[0, N]`, in graded-lex order.
template < int M, int N, typename Func >
constexpr void forEachMonomial( Func&& func )
{
    for ( int deg = 0; deg <= N; ++deg )
    {
        forEachMonomialOfDegree< M >( deg, func );
    }
}

/// Call `func(beta, alpha-beta)` for every beta componentwise `<= alpha`.
template < int M, typename Func >
constexpr void forEachSubIndex( const MultiIndex< M >& alpha, Func&& func )
{
    MultiIndex< M > beta{};
    auto fill = [&]( auto& self, int var ) constexpr -> void {
        if ( var == M )
        {
            MultiIndex< M > gamma{};
            for ( int i = 0; i < M; ++i )
                gamma[std::size_t( i )] = alpha[std::size_t( i )] - beta[std::size_t( i )];
            func( beta, gamma );
            return;
        }
        for ( int b = 0; b <= alpha[std::size_t( var )]; ++b )
        {
            beta[std::size_t( var )] = b;
            self( self, var + 1 );
        }
    };
    fill( fill, 0 );
}

}  // namespace tax
