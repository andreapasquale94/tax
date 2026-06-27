#include <gtest/gtest.h>

#include <cmath>
#include <tax/tax.hpp>

// Chebyshev on a non-canonical interval [0, 3], with the domain carried in the
// basis type.
template < int N >
using Cheb03 =
    tax::Expansion< double, tax::ChebyshevBasisOn< 0.0, 3.0 >, tax::IsotropicScheme< N, 1 > >;

TEST( ChebyshevDomain, InterpolatesOnInterval )
{
    auto f = tax::chebyshevInterpolate< 18, tax::ChebyshevBasisOn< 0.0, 3.0 > >(
        []( double x ) { return std::exp( x ); } );
    for ( double x = 0.0; x <= 3.0; x += 0.1 ) EXPECT_NEAR( f.eval( x ), std::exp( x ), 1e-8 );
}

TEST( ChebyshevDomain, DerivativeUsesChainRule )
{
    // d/dx sin(x) = cos(x) on [0, 3] — exercises the du/dx = 2/(b-a) factor.
    auto f = tax::chebyshevInterpolate< 20, tax::ChebyshevBasisOn< 0.0, 3.0 > >(
        []( double x ) { return std::sin( x ); } );
    auto df = f.deriv();
    for ( double x = 0.0; x <= 3.0; x += 0.1 ) EXPECT_NEAR( df.eval( x ), std::cos( x ), 1e-7 );
}

TEST( ChebyshevDomain, IntegralInvertsDerivative )
{
    auto f = tax::chebyshevInterpolate< 20, tax::ChebyshevBasisOn< 0.0, 3.0 > >(
        []( double x ) { return std::sin( x ); } );
    auto rt = f.integ().deriv();
    for ( int k = 0; k < int( Cheb03< 20 >::nCoefficients ); ++k )
        EXPECT_NEAR( rt[std::size_t( k )], f[std::size_t( k )], 1e-9 );
}

TEST( ChebyshevDomain, MathOnInterval )
{
    // exp composed on [0, 3] via interpolation, then compared to the host.
    auto x = tax::chebyshevInterpolate< 20, tax::ChebyshevBasisOn< 0.0, 3.0 > >(
        []( double t ) { return t; } );  // identity on [0,3]
    auto e = exp( x );
    for ( double t = 0.0; t <= 3.0; t += 0.1 ) EXPECT_NEAR( e.eval( t ), std::exp( t ), 1e-7 );
}
