// ---------------------------------------------------------------------------
// bindings.cpp — pybind11 module exposing the runtime TE class.
// ---------------------------------------------------------------------------

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include "operations.hpp"
#include "runtime_te.hpp"

namespace py = pybind11;
using tax::py::PyTE;

PYBIND11_MODULE( _tax, m )
{
    m.doc() = "tax — runtime-dimensioned Truncated Algebraic eXpansions";

#ifdef TAX_PY_VERSION
    m.attr( "__version__" ) = TAX_PY_VERSION;
#endif

    m.attr( "TAX_PY_MAX_N" ) = int( TAX_PY_MAX_N );
    m.attr( "TAX_PY_MAX_M" ) = int( TAX_PY_MAX_M );
    m.attr( "TAX_PY_MAX_COEFFS" ) = int( TAX_PY_MAX_COEFFS );

    auto cls = py::class_< PyTE >( m, "TE", R"doc(
Truncated algebraic expansion (Taylor basis) with runtime-chosen order N
and number of variables M.

Use :func:`tax.variable` or :func:`tax.variables` to build variables, then
arbitrary arithmetic and math expressions propagate the expansion.
)doc" );

    cls.def( py::init( []( int order, int nvars ) { return PyTE( order, nvars ); } ),
             py::arg( "order" ), py::arg( "nvars" ),
             "Construct a zero polynomial with the given order and variable count." );

    cls.def_readonly( "order", &PyTE::order, "Truncation order N." );
    cls.def_readonly( "nvars", &PyTE::nvars, "Number of variables M." );

    cls.def( "value", &PyTE::value, "Constant term (value at the expansion point)." );
    cls.def( "size", &PyTE::size, "Number of coefficients stored." );

    cls.def(
        "coeffs",
        []( const PyTE& a ) { return a.coeffs; },
        "Copy of the coefficient vector (graded-lex order)." );
    cls.def( "eval", &tax::py::evalAt, py::arg( "dx" ),
             "Evaluate the polynomial at point dx (length M)." );

    cls.def(
        "deriv",
        []( const PyTE& a, int var ) { return tax::py::derivVar( a, var ); },
        py::arg( "var" ), "Partial derivative polynomial wrt variable `var`." );
    cls.def(
        "integ",
        []( const PyTE& a, int var ) { return tax::py::integVar( a, var ); },
        py::arg( "var" ), "Indefinite integral polynomial wrt variable `var`." );

    cls.def( "derivative", &tax::py::derivativeAt, py::arg( "alpha" ),
             "Partial derivative value at the expansion point for multi-index alpha." );
    cls.def( "coeff", &tax::py::coeffAt, py::arg( "alpha" ),
             "Raw coefficient for the given multi-index." );
    cls.def( "multi_index", &tax::py::unflat, py::arg( "k" ),
             "Multi-index for the coefficient stored at flat index `k`." );

    cls.def( "__repr__", &tax::py::repr );
    cls.def( "__str__", &tax::py::repr );
    cls.def( "__len__", &PyTE::size );

    // --- Arithmetic operators ---------------------------------------------
    cls.def( "__pos__", []( const PyTE& a ) { return a; } );
    cls.def( "__neg__", &tax::py::neg );

    cls.def( "__add__", &tax::py::add, py::is_operator() );
    cls.def( "__sub__", &tax::py::sub, py::is_operator() );
    cls.def( "__mul__", &tax::py::mul, py::is_operator() );
    cls.def( "__truediv__", &tax::py::div, py::is_operator() );

    cls.def( "__add__", &tax::py::addScalar, py::is_operator() );
    cls.def( "__sub__", &tax::py::subScalar, py::is_operator() );
    cls.def( "__mul__", &tax::py::mulScalar, py::is_operator() );
    cls.def( "__truediv__", &tax::py::divScalar, py::is_operator() );

    cls.def(
        "__radd__", []( const PyTE& a, double s ) { return tax::py::addScalar( a, s ); },
        py::is_operator() );
    cls.def(
        "__rsub__", []( const PyTE& a, double s ) { return tax::py::scalarSub( s, a ); },
        py::is_operator() );
    cls.def(
        "__rmul__", []( const PyTE& a, double s ) { return tax::py::mulScalar( a, s ); },
        py::is_operator() );
    cls.def(
        "__rtruediv__", []( const PyTE& a, double s ) { return tax::py::scalarDiv( s, a ); },
        py::is_operator() );

    cls.def(
        "__pow__",
        []( const PyTE& a, int p ) { return tax::py::powInt( a, p ); },
        py::is_operator() );
    cls.def(
        "__pow__",
        []( const PyTE& a, double p ) { return tax::py::powDouble( a, p ); },
        py::is_operator() );
    cls.def(
        "__pow__",
        []( const PyTE& a, const PyTE& b ) { return tax::py::powTE( a, b ); },
        py::is_operator() );

    // --- Factories --------------------------------------------------------
    m.def( "constant", &tax::py::makeConstant, py::arg( "value" ), py::arg( "order" ),
           py::arg( "nvars" ), "Constant TE of the given value, order and variable count." );

    m.def( "variable", &tax::py::makeVariable, py::arg( "order" ), py::arg( "nvars" ),
           py::arg( "index" ), py::arg( "x0" ),
           "Coordinate variable x_index expanded about x0 (constant term = x0, "
           "linear term in x_index = 1)." );

    m.def( "variables", &tax::py::makeVariables, py::arg( "order" ), py::arg( "nvars" ),
           py::arg( "x0" ),
           "Build all M coordinate variables expanded about the point `x0` (length M)." );

    // --- Free-function math (module-level, accepted by arrays of TEs) -----
#define TAX_PY_BIND_UNARY( NAME )                                                                  \
    m.def( #NAME, &tax::py::NAME, py::arg( "x" ), "Apply " #NAME " element-wise to a TE." )

    TAX_PY_BIND_UNARY( sin );
    TAX_PY_BIND_UNARY( cos );
    TAX_PY_BIND_UNARY( tan );
    TAX_PY_BIND_UNARY( asin );
    TAX_PY_BIND_UNARY( acos );
    TAX_PY_BIND_UNARY( atan );
    TAX_PY_BIND_UNARY( sinh );
    TAX_PY_BIND_UNARY( cosh );
    TAX_PY_BIND_UNARY( tanh );
    TAX_PY_BIND_UNARY( asinh );
    TAX_PY_BIND_UNARY( acosh );
    TAX_PY_BIND_UNARY( atanh );
    TAX_PY_BIND_UNARY( exp );
    TAX_PY_BIND_UNARY( log );
    TAX_PY_BIND_UNARY( log10 );
    TAX_PY_BIND_UNARY( sqrt );
    TAX_PY_BIND_UNARY( cbrt );
    TAX_PY_BIND_UNARY( square );
    TAX_PY_BIND_UNARY( cube );
    TAX_PY_BIND_UNARY( abs );
    TAX_PY_BIND_UNARY( erf );
#undef TAX_PY_BIND_UNARY

    m.def( "atan2", &tax::py::atan2, py::arg( "y" ), py::arg( "x" ) );
    m.def( "hypot", &tax::py::hypot2, py::arg( "x" ), py::arg( "y" ) );
    m.def(
        "pow",
        []( const PyTE& a, int p ) { return tax::py::powInt( a, p ); },
        py::arg( "x" ), py::arg( "p" ) );
    m.def(
        "pow",
        []( const PyTE& a, double p ) { return tax::py::powDouble( a, p ); },
        py::arg( "x" ), py::arg( "p" ) );
    m.def(
        "pow",
        []( const PyTE& a, const PyTE& b ) { return tax::py::powTE( a, b ); },
        py::arg( "x" ), py::arg( "y" ) );
}
