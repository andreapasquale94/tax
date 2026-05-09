// SPDX-License-Identifier: BSD-3-Clause
//
// Python bindings for tax via nanobind.
//
// Per the architectural brief, Python sees `DynTE<double>` only — no
// std::variant dispatch over a (Order, Vars) grid, no JIT instantiation
// tricks.  The dynamic storage type is what users get; arithmetic and
// math operators evaluate the underlying ET expressions into a fresh
// DynTE on every call (because Python expressions cannot meaningfully
// own lazy ET temporaries across statements).

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <span>
#include <vector>

#include "tax/tax.hpp"

namespace nb = nanobind;
using DynTE = tax::DynamicTaylorExpansion< double >;

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

// Realise an ET expression into a fresh DynTE matching `like` in shape.
template < class Expr >
[[nodiscard]] DynTE realise( const DynTE& like, Expr&& expr )
{
    DynTE out( like.order(), like.nvars() );
    out <<= std::forward< Expr >( expr );
    return out;
}

}  // namespace

NB_MODULE( _tax, m )
{
    m.doc() = "Truncated multivariate Taylor expansions (DynTE<double>).";

    nb::class_< DynTE >( m, "DynTE" )
        // ---- ctors -----------------------------------------------------
        .def( nb::init<>() )
        .def( nb::init< std::size_t, std::size_t >(), nb::arg( "order" ), nb::arg( "nvars" ),
              "Allocate a zero DynTE with the given truncation order and number of variables." )

        // ---- factories -------------------------------------------------
        .def_static( "zero", &DynTE::zero, nb::arg( "order" ), nb::arg( "nvars" ) )
        .def_static( "one", &DynTE::one, nb::arg( "order" ), nb::arg( "nvars" ) )
        .def_static( "constant", &DynTE::constant, nb::arg( "value" ), nb::arg( "order" ),
                     nb::arg( "nvars" ) )
        .def_static( "variable", &DynTE::variable, nb::arg( "value" ), nb::arg( "order" ),
                     nb::arg( "nvars" ), nb::arg( "var_idx" ),
                     "x_i = value + dx_i where var_idx selects the seeded variable." )
        .def_static(
            "variables",
            []( const std::vector< double >& x0, std::size_t order ) {
                return DynTE::variables( x0, order );
            },
            nb::arg( "x0" ), nb::arg( "order" ),
            "Return a list of M = len(x0) independent variables." )

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

    // ---- math free functions -------------------------------------------
    m.def( "sin", []( const DynTE& a ) { return realise( a, tax::sin( a ) ); } );
    m.def( "cos", []( const DynTE& a ) { return realise( a, tax::cos( a ) ); } );
    m.def( "tan", []( const DynTE& a ) { return realise( a, tax::tan( a ) ); } );
    m.def( "sinh", []( const DynTE& a ) { return realise( a, tax::sinh( a ) ); } );
    m.def( "cosh", []( const DynTE& a ) { return realise( a, tax::cosh( a ) ); } );
    m.def( "tanh", []( const DynTE& a ) { return realise( a, tax::tanh( a ) ); } );
    m.def( "exp", []( const DynTE& a ) { return realise( a, tax::exp( a ) ); } );
    m.def( "log", []( const DynTE& a ) { return realise( a, tax::log( a ) ); } );
    m.def( "sqrt", []( const DynTE& a ) { return realise( a, tax::sqrt( a ) ); } );
    m.def( "square", []( const DynTE& a ) { return realise( a, tax::square( a ) ); } );
    m.def( "cube", []( const DynTE& a ) { return realise( a, tax::cube( a ) ); } );
}
