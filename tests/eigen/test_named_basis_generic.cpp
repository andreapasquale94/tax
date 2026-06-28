#include <gtest/gtest.h>

#include <Eigen/Core>
#include <array>
#include <tax/tax.hpp>

// Named axes work for any basis, not just Taylor: composition across axis sets,
// axis-addressed deriv/slice, and point-form per-axis gradient/Jacobian.

template < typename Basis >
static void checkNamedAlgebra()
{
    auto x = tax::named::variable< "x", 5, Basis >( 0.0 );  // axis "x"
    auto p = tax::named::variable< "p", 5, Basis >( 0.0 );  // axis "p"
    auto f = x * p + 2.0 * x;                               // union axes {p, x}

    using F = decltype( f );
    static_assert( std::is_same_v< typename F::basis, Basis > );
    static_assert( F::vars_v == 2 );

    // Joint variable order is name-sorted: p = var 0, x = var 1.
    typename F::Input pt{ 0.3, -0.4 };  // p = 0.3, x = -0.4
    const double pv = pt[0], xv = pt[1];

    EXPECT_NEAR( f.inner().eval( pt ), xv * pv + 2.0 * xv, 1e-10 );

    // ∂f/∂x = p + 2, evaluated along the named "x" axis.
    auto gx = tax::named::gradient< "x" >( f, Eigen::Vector2d{ pv, xv } );
    EXPECT_NEAR( gx( 0 ), pv + 2.0, 1e-10 );
    // ∂f/∂p = x.
    auto gp = tax::named::gradient< "p" >( f, Eigen::Vector2d{ pv, xv } );
    EXPECT_NEAR( gp( 0 ), xv, 1e-10 );

    // Axis-addressed derivative polynomial.
    auto dfx = f.template deriv< "x" >();
    EXPECT_NEAR( dfx.inner().eval( pt ), pv + 2.0, 1e-10 );

    // Slice onto "x": keeps terms with p-exponent 0, i.e. 2x.
    auto sx = f.template slice< "x" >();
    using SX = decltype( sx );
    static_assert( SX::vars_v == 1 );
    EXPECT_NEAR( sx.inner().eval( typename SX::Input{ xv } ), 2.0 * xv, 1e-10 );
}

TEST( NamedBasisGeneric, AlgebraChebyshev ) { checkNamedAlgebra< tax::ChebyshevBasis >(); }
TEST( NamedBasisGeneric, AlgebraLegendre ) { checkNamedAlgebra< tax::LegendreBasis >(); }
TEST( NamedBasisGeneric, AlgebraHermite ) { checkNamedAlgebra< tax::HermiteBasis >(); }

// Jacobian w.r.t. a named axis for a vector of named expansions, any basis.
template < typename Basis >
static void checkNamedJacobian()
{
    auto x = tax::named::variable< "x", 5, Basis >( 0.0 );
    auto p = tax::named::variable< "p", 5, Basis >( 0.0 );

    using F = decltype( x * p );  // union {p, x}
    Eigen::Matrix< F, 2, 1 > G;
    G( 0 ) = x * p;        // ∂/∂x = p
    G( 1 ) = x + 3.0 * p;  // ∂/∂x = 1

    Eigen::Vector2d at{ 0.5, -0.2 };  // p = 0.5, x = -0.2
    auto J = tax::named::jacobian< "x" >( G, at );
    EXPECT_NEAR( J( 0, 0 ), 0.5, 1e-10 );  // ∂(xp)/∂x = p
    EXPECT_NEAR( J( 1, 0 ), 1.0, 1e-10 );  // ∂(x+3p)/∂x = 1
}

TEST( NamedBasisGeneric, JacobianChebyshev ) { checkNamedJacobian< tax::ChebyshevBasis >(); }
TEST( NamedBasisGeneric, JacobianLegendre ) { checkNamedJacobian< tax::LegendreBasis >(); }
TEST( NamedBasisGeneric, JacobianHermite ) { checkNamedJacobian< tax::HermiteBasis >(); }

// Eigen NumTraits for named non-Taylor expansions.
template < typename Basis >
static void checkNamedNumTraits()
{
    auto x = tax::named::variable< "x", 4, Basis >( 0.0 );
    using NX = decltype( x );  // single axis "x"

    Eigen::Matrix< NX, 2, 2 > A;
    A( 0, 0 ) = 1.0 + x;
    A( 0, 1 ) = x;
    A( 1, 0 ) = x;
    A( 1, 1 ) = NX{ 1.0 };
    Eigen::Matrix< NX, 2, 1 > v;
    v( 0 ) = x;
    v( 1 ) = NX{ 2.0 };

    Eigen::Matrix< NX, 2, 1 > w = A * v;
    const double xv = 0.6;
    EXPECT_NEAR( w( 0 ).inner().eval( typename NX::Input{ xv } ), ( 1.0 + xv ) * xv + xv * 2.0,
                 1e-10 );
    EXPECT_NEAR( w( 1 ).inner().eval( typename NX::Input{ xv } ), xv * xv + 1.0 * 2.0, 1e-10 );
}

TEST( NamedBasisGeneric, NumTraitsChebyshev ) { checkNamedNumTraits< tax::ChebyshevBasis >(); }
TEST( NamedBasisGeneric, NumTraitsLegendre ) { checkNamedNumTraits< tax::LegendreBasis >(); }
TEST( NamedBasisGeneric, NumTraitsHermite ) { checkNamedNumTraits< tax::HermiteBasis >(); }

// Transcendental functions on a named NON-Taylor expansion: the named unary math
// surface must be basis-generic, forwarding to the inner basis' own math (here
// Chebyshev), not hardcoded to TaylorBasis.
TEST( NamedBasisGeneric, TranscendentalChebyshev )
{
    auto x = tax::named::variable< "x", 8, tax::ChebyshevBasis >( 0.0 );
    using X = decltype( x );

    auto fx = tax::exp( x );           // basis-generic named unary surface
    auto ref = tax::exp( x.inner() );  // bare Chebyshev exp
    static_assert( std::is_same_v< typename decltype( fx )::basis, tax::ChebyshevBasis > );
    for ( std::size_t k = 0; k < X::Inner::nCoefficients; ++k )
        EXPECT_NEAR( fx.inner()[k], ref[k], 1e-12 );

    auto sx = tax::sin( x );
    auto sref = tax::sin( x.inner() );
    for ( std::size_t k = 0; k < X::Inner::nCoefficients; ++k )
        EXPECT_NEAR( sx.inner()[k], sref[k], 1e-12 );
}

// The Taylor named surface is unchanged: NE / NamedTaylorExpansion still alias
// the merged class, and value() works generically.
TEST( NamedBasisGeneric, TaylorAliasIntact )
{
    static_assert(
        std::is_same_v< tax::NE< 4, tax::Axis< "x", 2 > >,
                        tax::NamedExpansion< double, tax::TaylorBasis, 4, tax::Axis< "x", 2 > > > );
    auto x = tax::named::variable< "x", 4, tax::LegendreBasis >( 0.0 );
    EXPECT_DOUBLE_EQ( tax::named::value( x ), 0.0 );
}
