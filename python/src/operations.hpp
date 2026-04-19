#pragma once

// ---------------------------------------------------------------------------
// operations.hpp — declarations for PyTE operations.
//
// Implementations live in the ops_*.cpp translation units so that
// template instantiations for the full (N, M) grid compile in parallel.
// ---------------------------------------------------------------------------

#include <string>
#include <vector>
#include "runtime_te.hpp"

namespace tax::py
{

// ---- Factories ------------------------------------------------------------
PyTE makeConstant( int n, int m, double value );
PyTE makeVariable( int n, int m, int i, double x0 );
std::vector< PyTE > makeVariables( int n, int m, const std::vector< double >& x0 );

// ---- Arithmetic ----------------------------------------------------------
PyTE add( const PyTE& a, const PyTE& b );
PyTE sub( const PyTE& a, const PyTE& b );
PyTE mul( const PyTE& a, const PyTE& b );
PyTE div( const PyTE& a, const PyTE& b );
PyTE neg( const PyTE& a );

// ---- TE op scalar --------------------------------------------------------
PyTE addScalar( const PyTE& a, double s );
PyTE subScalar( const PyTE& a, double s );
PyTE mulScalar( const PyTE& a, double s );
PyTE divScalar( const PyTE& a, double s );
PyTE scalarSub( double s, const PyTE& a );
PyTE scalarDiv( double s, const PyTE& a );

// ---- Trigonometric -------------------------------------------------------
PyTE sin( const PyTE& a );
PyTE cos( const PyTE& a );
PyTE tan( const PyTE& a );
PyTE asin( const PyTE& a );
PyTE acos( const PyTE& a );
PyTE atan( const PyTE& a );
PyTE atan2( const PyTE& y, const PyTE& x );

// ---- Hyperbolic ----------------------------------------------------------
PyTE sinh( const PyTE& a );
PyTE cosh( const PyTE& a );
PyTE tanh( const PyTE& a );
PyTE asinh( const PyTE& a );
PyTE acosh( const PyTE& a );
PyTE atanh( const PyTE& a );

// ---- Misc transcendental & algebraic -------------------------------------
PyTE exp( const PyTE& a );
PyTE log( const PyTE& a );
PyTE log10( const PyTE& a );
PyTE sqrt( const PyTE& a );
PyTE cbrt( const PyTE& a );
PyTE square( const PyTE& a );
PyTE cube( const PyTE& a );
PyTE abs( const PyTE& a );
PyTE erf( const PyTE& a );
PyTE hypot2( const PyTE& x, const PyTE& y );

// ---- Powers --------------------------------------------------------------
PyTE powInt( const PyTE& a, int p );
PyTE powDouble( const PyTE& a, double p );
PyTE powTE( const PyTE& a, const PyTE& b );

// ---- Access / evaluation / deriv / integ ---------------------------------
double evalAt( const PyTE& a, const std::vector< double >& dx );
PyTE derivVar( const PyTE& a, int var );
PyTE integVar( const PyTE& a, int var );
double derivativeAt( const PyTE& a, const std::vector< int >& alpha );
double coeffAt( const PyTE& a, const std::vector< int >& alpha );
std::vector< int > unflat( const PyTE& a, std::size_t k );
std::string repr( const PyTE& a );

}  // namespace tax::py
