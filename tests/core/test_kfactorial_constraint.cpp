#include <gtest/gtest.h>

#include <tax/tax.hpp>

// The k!-scaled value accessors apply Taylor semantics and are wrong for
// orthogonal bases, so they must be callable on a Taylor expansion and
// NON-callable (SFINAE-disabled) on Chebyshev/Legendre/Hermite.

template < typename E >
concept HasValueDerivative =
    requires( const E e, tax::MultiIndex< E::scheme::vars > a ) { e.derivative( a ); };

using TaylorE = tax::TE< 3, 2 >;
using ChebE = tax::ChebyshevExpansion< 3, 2 >;
using LegE = tax::LegendreSeries< 3, 2 >;
using HermE = tax::HermiteSeries< 3, 2 >;

static_assert( HasValueDerivative< TaylorE >,
               "Taylor expansion must expose the k!-scaled derivative()" );
static_assert( !HasValueDerivative< ChebE >,
               "Chebyshev expansion must NOT expose the k!-scaled derivative()" );
static_assert( !HasValueDerivative< LegE >,
               "Legendre expansion must NOT expose the k!-scaled derivative()" );
static_assert( !HasValueDerivative< HermE >,
               "Hermite expansion must NOT expose the k!-scaled derivative()" );

// Also prove the compile-time template overload derivative<Alpha...>() is
// equally constrained to TaylorBasis.
template < typename E >
concept HasTemplateDerivative = requires( const E e ) {
    e.template derivative< 1, 0 >();  // 2-variable form matching TE<3,2>
};

static_assert( HasTemplateDerivative< TaylorE >,
               "Taylor expansion must expose derivative<Alpha...>()" );
static_assert( !HasTemplateDerivative< ChebE >,
               "Chebyshev expansion must NOT expose derivative<Alpha...>()" );
static_assert( !HasTemplateDerivative< LegE >,
               "Legendre expansion must NOT expose derivative<Alpha...>()" );
static_assert( !HasTemplateDerivative< HermE >,
               "Hermite expansion must NOT expose derivative<Alpha...>()" );

TEST( KFactorialConstraint, TaylorOnly )
{
    // The static_asserts above are the real test; this keeps a runtime hook so
    // the file is a normal gtest TU.
    SUCCEED();
}
