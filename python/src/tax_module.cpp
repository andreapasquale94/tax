// SPDX-License-Identifier: BSD-3-Clause
//
// Python bindings for tax via nanobind.
//
// Per the architectural brief, Python sees `DynTE<double>` only — no
// std::variant dispatch over a (Order, Vars) grid, no JIT instantiation
// tricks.  Construction goes through module-level utility functions
// (`tax.zero`, `tax.one`, `tax.constant`, `tax.variable`,
// `tax.variables`); the class itself isn't directly constructible from
// Python.  Arithmetic and math operators evaluate the underlying ET
// expressions into a fresh DynTE on every call (because Python
// expressions cannot meaningfully own lazy ET temporaries across
// statements).

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <span>
#include <vector>

#include "tax/tax.hpp"

namespace nb = nanobind;
using DynTE = tax::DynTE< double >;

namespace
{

// Wrap a Python list of ints into a span<const std::size_t>.  Stored on
// the stack of the calling frame so it stays alive for the duration of
// the binding call.
struct MultiIndexBuf
{
    std::vector< std::size_t > data;
    [[nodiscard]] std::span< const std::size_t > view() const noexcept
    {
        return std::span< const std::size_t >( data.data(), data.size() );
    }
};

[[nodiscard]] MultiIndexBuf toMultiIndex( const std::vector< std::size_t >& alpha )
{
    return MultiIndexBuf{ alpha };
}

// Realise an ET expression into a fresh DynTE.  Just forwards to the
// expression's own `.eval()` — we keep this thin wrapper purely so the
// binding lambdas read uniformly across paths that don't pass through
// an ET (e.g. operator+(DynTE, double) which is itself an ET that we
// then eval).
template < class Expr >
[[nodiscard]] DynTE realise( const DynTE& /*like*/, Expr&& expr )
{
    return std::forward< Expr >( expr ).eval();
}

}  // namespace

NB_MODULE( _tax, m )
{
    m.doc() = "Truncated multivariate Taylor expansions (DynTE<double>).";

    nb::class_< DynTE >( m, "DynTE",
                         "A truncated Taylor expansion in M variables of order N.\n"
                         "Construct via the module-level factories: tax.zero, tax.one,\n"
                         "tax.constant, tax.variable, tax.variables." )
        // ---- accessors -------------------------------------------------
        .def_prop_ro( "order", &DynTE::order )
        .def_prop_ro( "nvars", &DynTE::nvars )
        .def( "value", &DynTE::value )
        .def(
            "coeff",
            []( const DynTE& self, const std::vector< std::size_t >& alpha ) {
                auto buf = toMultiIndex( alpha );
                return self.coeff( buf.view() );
            },
            nb::arg( "alpha" ) )
        .def(
            "derivative",
            []( const DynTE& self, const std::vector< std::size_t >& alpha ) {
                auto buf = toMultiIndex( alpha );
                return self.derivative( buf.view() );
            },
            nb::arg( "alpha" ) )
        .def(
            "eval",
            []( const DynTE& self, const std::vector< double >& dx ) {
                return self.eval( dx );
            },
            nb::arg( "dx" ) )

        // ---- norms -----------------------------------------------------
        .def( "coeffs_norm_inf", &DynTE::coeffsNormInf )
        .def( "coeffs_norm_1", []( const DynTE& self ) { return self.coeffsNorm< 1 >(); } )
        .def( "coeffs_norm_2", []( const DynTE& self ) { return self.coeffsNorm< 2 >(); } )

        // ---- arithmetic ------------------------------------------------
        .def( "__add__",
              []( const DynTE& a, const DynTE& b ) { return realise( a, a + b ); } )
        .def( "__add__",
              []( const DynTE& a, double c ) { return realise( a, a + c ); } )
        .def( "__radd__",
              []( const DynTE& a, double c ) { return realise( a, c + a ); } )
        .def( "__sub__",
              []( const DynTE& a, const DynTE& b ) { return realise( a, a - b ); } )
        .def( "__sub__",
              []( const DynTE& a, double c ) { return realise( a, a - c ); } )
        .def( "__rsub__",
              []( const DynTE& a, double c ) { return realise( a, c - a ); } )
        .def( "__mul__",
              []( const DynTE& a, const DynTE& b ) { return realise( a, a * b ); } )
        .def( "__mul__",
              []( const DynTE& a, double c ) { return realise( a, a * c ); } )
        .def( "__rmul__",
              []( const DynTE& a, double c ) { return realise( a, c * a ); } )
        .def( "__truediv__",
              []( const DynTE& a, const DynTE& b ) { return realise( a, a / b ); } )
        .def( "__truediv__",
              []( const DynTE& a, double c ) { return realise( a, a / c ); } )
        .def( "__neg__",
              []( const DynTE& a ) { return realise( a, -a ); } )

        // ---- repr ------------------------------------------------------
        .def( "__repr__", []( const DynTE& self ) {
            return std::string( "DynTE(order=" ) + std::to_string( self.order() )
                   + ", nvars=" + std::to_string( self.nvars() )
                   + ", value=" + std::to_string( self.value() ) + ")";
        } );

    // ---- module-level factories ----------------------------------------
    //
    // Wrapped in lambdas because the unified TaylorExpansionT exposes both
    // static-only and dynamic-only overloads of each factory name; a bare
    // `&DynTE::zero` is then ambiguous to overload-resolve.  The lambdas
    // pin down the dynamic signature.
    m.def(
        "zero",
        []( std::size_t order, std::size_t nvars ) {
            return DynTE::zero( order, nvars );
        },
        nb::arg( "order" ), nb::arg( "nvars" ),
        "Allocate a zero TTE of the given order and number of variables." );
    m.def(
        "one",
        []( std::size_t order, std::size_t nvars ) {
            return DynTE::one( order, nvars );
        },
        nb::arg( "order" ), nb::arg( "nvars" ),
        "TTE whose constant term is 1 and all other coefficients are 0." );
    m.def(
        "constant",
        []( double value, std::size_t order, std::size_t nvars ) {
            return DynTE::constant( value, order, nvars );
        },
        nb::arg( "value" ), nb::arg( "order" ), nb::arg( "nvars" ),
        "TTE whose constant term is `value` and whose linear part is 0." );
    m.def(
        "variable",
        []( double value, std::size_t order, std::size_t nvars,
            std::size_t var_idx ) {
            return DynTE::variable( value, order, nvars, var_idx );
        },
        nb::arg( "value" ), nb::arg( "order" ), nb::arg( "nvars" ),
        nb::arg( "var_idx" ),
        "x_i = value + dx_i, with the dx-seed placed at multi-index var_idx." );
    m.def(
        "variables",
        []( const std::vector< double >& x0, std::size_t order ) {
            return DynTE::variables( x0, order );
        },
        nb::arg( "x0" ), nb::arg( "order" ),
        "Return a list of M = len(x0) independent variables sharing the\n"
        "same truncation order, each seeded against its own dx_i." );

    // ---- math free functions -------------------------------------------
    m.def( "sin", []( const DynTE& a ) { return realise( a, tax::sin( a ) ); } );
    m.def( "cos", []( const DynTE& a ) { return realise( a, tax::cos( a ) ); } );
    m.def( "tan", []( const DynTE& a ) { return realise( a, tax::tan( a ) ); } );
    m.def( "sinh", []( const DynTE& a ) { return realise( a, tax::sinh( a ) ); } );
    m.def( "cosh", []( const DynTE& a ) { return realise( a, tax::cosh( a ) ); } );
    m.def( "tanh", []( const DynTE& a ) { return realise( a, tax::tanh( a ) ); } );
    m.def( "asin", []( const DynTE& a ) { return realise( a, tax::asin( a ) ); } );
    m.def( "acos", []( const DynTE& a ) { return realise( a, tax::acos( a ) ); } );
    m.def( "atan", []( const DynTE& a ) { return realise( a, tax::atan( a ) ); } );
    m.def( "asinh", []( const DynTE& a ) { return realise( a, tax::asinh( a ) ); } );
    m.def( "acosh", []( const DynTE& a ) { return realise( a, tax::acosh( a ) ); } );
    m.def( "atanh", []( const DynTE& a ) { return realise( a, tax::atanh( a ) ); } );
    m.def( "exp", []( const DynTE& a ) { return realise( a, tax::exp( a ) ); } );
    m.def( "log", []( const DynTE& a ) { return realise( a, tax::log( a ) ); } );
    m.def( "log10", []( const DynTE& a ) { return realise( a, tax::log10( a ) ); } );
    m.def( "sqrt", []( const DynTE& a ) { return realise( a, tax::sqrt( a ) ); } );
    m.def( "cbrt", []( const DynTE& a ) { return realise( a, tax::cbrt( a ) ); } );
    m.def( "square", []( const DynTE& a ) { return realise( a, tax::square( a ) ); } );
    m.def( "cube", []( const DynTE& a ) { return realise( a, tax::cube( a ) ); } );
    m.def( "erf", []( const DynTE& a ) { return realise( a, tax::erf( a ) ); } );

    // pow(a, p) — runtime real exponent.
    m.def(
        "pow",
        []( const DynTE& a, double p ) { return realise( a, tax::pow( a, p ) ); },
        nb::arg( "a" ), nb::arg( "p" ) );

    // atan2(y, x).
    m.def(
        "atan2",
        []( const DynTE& y, const DynTE& x ) { return realise( y, tax::atan2( y, x ) ); },
        nb::arg( "y" ), nb::arg( "x" ) );

    // hypot — 2 and 3 argument forms.
    m.def(
        "hypot",
        []( const DynTE& x, const DynTE& y ) { return realise( x, tax::hypot( x, y ) ); },
        nb::arg( "x" ), nb::arg( "y" ) );
    m.def(
        "hypot",
        []( const DynTE& x, const DynTE& y, const DynTE& z ) {
            return realise( x, tax::hypot( x, y, z ) );
        },
        nb::arg( "x" ), nb::arg( "y" ), nb::arg( "z" ) );

    // sincos / sinhcosh — return (sin, cos) / (sinh, cosh) tuples.  Python
    // doesn't carry lazy ETs across statements, but the C++ pair owner
    // still saves the second streaming sweep (the second realise() call
    // copies from an already-populated buffer).
    m.def(
        "sincos",
        []( const DynTE& a ) {
            auto p = tax::sincos( a );
            return std::pair< DynTE, DynTE >{ realise( a, p.sin() ), realise( a, p.cos() ) };
        },
        nb::arg( "a" ),
        "Return (sin(a), cos(a)).  Internally shares the paired sin/cos buffers." );
    m.def(
        "sinhcosh",
        []( const DynTE& a ) {
            auto p = tax::sinhcosh( a );
            return std::pair< DynTE, DynTE >{ realise( a, p.sinh() ), realise( a, p.cosh() ) };
        },
        nb::arg( "a" ),
        "Return (sinh(a), cosh(a)).  Internally shares the paired sinh/cosh buffers." );
}
