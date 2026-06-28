#include <gtest/gtest.h>

#include <concepts>
#include <tax/la.hpp>

#include "../testUtils.hpp"

// Axis types hoisted into aliases: a string-literal NTTP nested several levels
// deep inside std::same_as<...> trips the parser, so name them up front.
using PAxis = tax::Axis< "p", 2 >;
using XAxis = tax::Axis< "X", 4 >;
using POrderedAx = tax::OrderedAxis< "p", 2, 4 >;

// -----------------------------------------------------------------------------
// Type-identity: each alias expands to a VecNT (== Eigen column vector) of the
// matching expansion scalar.
// -----------------------------------------------------------------------------

static_assert( std::same_as< tax::la::TEVec< 3, 4, 2 >, tax::la::VecNT< 3, tax::TE< 4, 2 > > > );
static_assert( std::same_as< tax::la::TEVec< 3, 4, 2 >, Eigen::Matrix< tax::TE< 4, 2 >, 3, 1 > > );

// Univariate default (M == 1).
static_assert( std::same_as< tax::la::TEVec< 2, 5 >, tax::la::VecNT< 2, tax::TE< 5 > > > );

static_assert(
    std::same_as< tax::la::NEVec< 2, 4, PAxis >, tax::la::VecNT< 2, tax::NE< 4, PAxis > > > );

static_assert(
    std::same_as< tax::la::MTEVec< 2, POrderedAx >, tax::la::VecNT< 2, tax::MTE< POrderedAx > > > );

// MTE scalar alias mirrors NE: a double-valued MixedTaylorExpansion.
static_assert(
    std::same_as< tax::MTE< POrderedAx >, tax::MixedTaylorExpansion< double, POrderedAx > > );

// -----------------------------------------------------------------------------
// promote_t — the common (union-of-axes) expansion type operands promote into.
// "X" < "p" lexicographically, so the merged axis list stays sorted.
// -----------------------------------------------------------------------------

// Two named expansions over disjoint axes -> the union, sorted/unique.
static_assert( std::same_as< tax::promote_t< tax::NE< 2, XAxis >, tax::NE< 2, PAxis > >,
                             tax::NE< 2, XAxis, PAxis > > );
// Scalar promotes into the accompanying expansion (either operand order).
static_assert( std::same_as< tax::promote_t< tax::NE< 2, XAxis >, double >, tax::NE< 2, XAxis > > );
static_assert( std::same_as< tax::promote_t< double, tax::NE< 2, XAxis > >, tax::NE< 2, XAxis > > );
// Plain TE and a scalar.
static_assert( std::same_as< tax::promote_t< tax::TE< 4, 2 >, double >, tax::TE< 4, 2 > > );
// A single operand yields itself.
static_assert( std::same_as< tax::promote_t< tax::NE< 2, XAxis > >, tax::NE< 2, XAxis > > );

// -----------------------------------------------------------------------------
// Free tax::truncate< N2 > — order-reducing, scalar and Eigen overloads.
// -----------------------------------------------------------------------------

// Scalar: result is the lower-order type (dense and named).
static_assert( std::same_as< decltype( tax::truncate< 3 >( std::declval< tax::TE< 5, 2 > >() ) ),
                             tax::TE< 3, 2 > > );
static_assert(
    std::same_as< decltype( tax::truncate< 2 >( std::declval< tax::NE< 4, XAxis > >() ) ),
                  tax::NE< 2, XAxis > > );
// Eigen vector: same shape, lower-order element type.
static_assert( std::same_as< decltype( tax::truncate< 2 >(
                                 std::declval< tax::la::TEVec< 2, 5, 2 > >() ) )::Scalar,
                             tax::TE< 2, 2 > > );

// -----------------------------------------------------------------------------
// Runtime: the aliases are usable as ordinary Eigen vectors of expansions.
// -----------------------------------------------------------------------------

TEST( ExpansionVectors, TEVecHoldsVariables )
{
    Eigen::Vector2d x0{ 3.0, -1.0 };
    tax::la::TEVec< 2, 4, 2 > v = tax::la::variables< tax::TE< 4, 2 > >( x0 );
    EXPECT_NEAR( v( 0 ).value(), 3.0, 1e-15 );
    EXPECT_NEAR( v( 1 ).value(), -1.0, 1e-15 );
    EXPECT_NEAR( ( v( 0 ).coeff< 1, 0 >() ), 1.0, 1e-15 );
    EXPECT_NEAR( ( v( 1 ).coeff< 0, 1 >() ), 1.0, 1e-15 );
}

TEST( ExpansionVectors, NEVecHoldsNamedVariables )
{
    auto p = tax::variables< "p", 4 >( std::array< double, 2 >{ 1.0, 2.0 } );
    tax::la::NEVec< 2, 4, PAxis > v;
    v( 0 ) = p[0];
    v( 1 ) = p[1];
    EXPECT_NEAR( v( 0 ).value(), 1.0, 1e-15 );
    EXPECT_NEAR( v( 1 ).value(), 2.0, 1e-15 );
}

TEST( ExpansionVectors, MTEVecHoldsMixedVariables )
{
    auto p = tax::mixed::variables< "p", 4 >( std::array< double, 2 >{ 5.0, 6.0 } );
    tax::la::MTEVec< 2, POrderedAx > v;
    v( 0 ) = p[0];
    v( 1 ) = p[1];
    EXPECT_NEAR( v( 0 ).value(), 5.0, 1e-15 );
    EXPECT_NEAR( v( 1 ).value(), 6.0, 1e-15 );
}

TEST( ExpansionVectors, TruncateDropsHighOrderTerms )
{
    // f(x) = (1 + x)^2 truncated to order 1 keeps only 1 + 2x; the x^2 term goes.
    auto x = tax::TE< 3 >::variable( 0.0 );
    auto f = ( 1.0 + x ) * ( 1.0 + x );
    auto g = tax::truncate< 1 >( f );
    EXPECT_NEAR( g.value(), 1.0, 1e-15 );
    EXPECT_NEAR( g.template coeff< 1 >(), 2.0, 1e-15 );

    // Eigen overload truncates each element.
    Eigen::Vector2d x0{ 1.0, 2.0 };
    auto v = tax::la::variables< tax::TE< 4, 2 > >( x0 );
    auto vt = tax::truncate< 1 >( v );
    EXPECT_NEAR( vt( 0 ).value(), 1.0, 1e-15 );
    EXPECT_NEAR( ( vt( 0 ).template coeff< 1, 0 >() ), 1.0, 1e-15 );
}
