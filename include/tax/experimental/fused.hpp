#pragma once

// ---------------------------------------------------------------------------
// Experimental: fused elementwise expression templates (opt-in).
//
// `tax` is eager: `2*a + 3*b - c` materialises one full coefficient array per
// operator. For large coefficient counts (many variables, high order) those
// intermediate passes spill to memory and dominate. This header adds a lazy
// expression layer for the *elementwise* family — `+`, `-`, unary `-`,
// scalar `*` / `/`, and constant shift — so a whole linear combination fuses
// into a single pass with no temporaries.
//
// It does NOT replace the eager operators (no ODR / overload-resolution
// conflict): the lazy operators only engage once an operand is a fused node,
// which you opt into with `tax::fuse(x)`:
//
//     #include <tax/experimental/fused.hpp>
//     using tax::fuse;
//     TEn<6,4> r = 2.0 * fuse(a) + 3.0 * fuse(b) - fuse(c) + 0.5 * fuse(d);
//
// The Cauchy product (`expansion * expansion`) is NOT elementwise and stays
// eager; evaluate it first and `fuse()` the result as a leaf.
//
// Scope (prototype): dense `TaylorExpansion` only (any IndexScheme); named /
// mixed-named wrappers are not yet covered. Nodes hold operands by value and
// leaves reference the expansion, so — as with all expression templates —
// evaluate the expression in the same full statement that builds it.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <tax/core/taylor_expansion.hpp>
#include <type_traits>
#include <utility>

namespace tax::fused
{

/// A fused node exposes `scalar_type`, `scheme`, `nCoefficients`, `operator[]`,
/// and the `fused_node` marker. CRTP base supplies materialisation.
template < typename T >
concept Node = requires { typename std::remove_cvref_t< T >::fused_node; };

/// CRTP base: every node materialises into the matching dense expansion via one
/// fused loop, either explicitly (`.eval()`) or by implicit conversion.
template < typename Derived >
struct ExprBase
{
    using fused_node = void;

    [[nodiscard]] constexpr const Derived& derived() const noexcept
    {
        return static_cast< const Derived& >( *this );
    }

    // The materialised type is computed lazily (inside the member templates), so
    // CRTP does not require `Derived` complete while ExprBase is instantiated as
    // a base subobject.
    [[nodiscard]] constexpr auto eval() const noexcept
    {
        TaylorExpansion< typename Derived::scalar_type, typename Derived::scheme > r;
        const Derived& d = derived();
        for ( std::size_t k = 0; k < Derived::nCoefficients; ++k ) r[k] = d[k];
        return r;
    }

    /// Implicit materialisation into the matching dense expansion.
    template < typename E >
        requires std::same_as<
            E, TaylorExpansion< typename Derived::scalar_type, typename Derived::scheme > >
    /*implicit*/ constexpr operator E() const noexcept
    {
        return eval();
    }
};

// ---------------------------------------------------------------------------
// Leaf: a reference to a concrete expansion.
// ---------------------------------------------------------------------------
template < typename E >
struct Leaf : ExprBase< Leaf< E > >
{
    using scalar_type = typename E::scalar_type;
    using scheme = typename E::scheme;
    static constexpr std::size_t nCoefficients = E::nCoefficients;

    const E* e;

    [[nodiscard]] constexpr scalar_type operator[]( std::size_t k ) const noexcept
    {
        return ( *e )[k];
    }
};

/// Lift a dense expansion into the fused expression layer.
template < typename T, typename Scheme >
[[nodiscard]] constexpr Leaf< TaylorExpansion< T, Scheme > > fuse(
    const TaylorExpansion< T, Scheme >& e ) noexcept
{
    return Leaf< TaylorExpansion< T, Scheme > >{ {}, &e };
}

// `as_node`: identity on nodes, `fuse` on expansions. Used to normalise mixed
// operands (node + bare expansion) without forcing the user to wrap every leaf.
template < Node N >
[[nodiscard]] constexpr N as_node( const N& n ) noexcept
{
    return n;
}
template < typename T, typename Scheme >
[[nodiscard]] constexpr auto as_node( const TaylorExpansion< T, Scheme >& e ) noexcept
{
    return fuse( e );
}

/// True for operands the binary node operators accept (a node or an expansion).
template < typename X >
concept Leafable = Node< X > || requires {
    typename X::scalar_type;
    typename X::scheme;
    { X::nCoefficients } -> std::convertible_to< std::size_t >;
};

// ---------------------------------------------------------------------------
// Binary / unary elementwise nodes.
// ---------------------------------------------------------------------------
template < typename L, typename R >
struct Add : ExprBase< Add< L, R > >
{
    using scalar_type = typename L::scalar_type;
    using scheme = typename L::scheme;
    static constexpr std::size_t nCoefficients = L::nCoefficients;
    L l;
    R r;
    [[nodiscard]] constexpr scalar_type operator[]( std::size_t k ) const noexcept
    {
        return l[k] + r[k];
    }
};

template < typename L, typename R >
struct Sub : ExprBase< Sub< L, R > >
{
    using scalar_type = typename L::scalar_type;
    using scheme = typename L::scheme;
    static constexpr std::size_t nCoefficients = L::nCoefficients;
    L l;
    R r;
    [[nodiscard]] constexpr scalar_type operator[]( std::size_t k ) const noexcept
    {
        return l[k] - r[k];
    }
};

template < typename A >
struct Neg : ExprBase< Neg< A > >
{
    using scalar_type = typename A::scalar_type;
    using scheme = typename A::scheme;
    static constexpr std::size_t nCoefficients = A::nCoefficients;
    A a;
    [[nodiscard]] constexpr scalar_type operator[]( std::size_t k ) const noexcept { return -a[k]; }
};

/// Scalar scaling `s * node`.
template < typename A >
struct Scale : ExprBase< Scale< A > >
{
    using scalar_type = typename A::scalar_type;
    using scheme = typename A::scheme;
    static constexpr std::size_t nCoefficients = A::nCoefficients;
    A a;
    scalar_type s;
    [[nodiscard]] constexpr scalar_type operator[]( std::size_t k ) const noexcept
    {
        return s * a[k];
    }
};

/// Add a scalar to the constant term only.
template < typename A >
struct ConstantShift : ExprBase< ConstantShift< A > >
{
    using scalar_type = typename A::scalar_type;
    using scheme = typename A::scheme;
    static constexpr std::size_t nCoefficients = A::nCoefficients;
    A a;
    scalar_type s;
    [[nodiscard]] constexpr scalar_type operator[]( std::size_t k ) const noexcept
    {
        return k == 0 ? a[k] + s : a[k];
    }
};

// ---------------------------------------------------------------------------
// Operators. Each requires at least one fused-node operand, so bare
// `expansion OP expansion` keeps resolving to the eager operators.
// ---------------------------------------------------------------------------
template < typename L, typename R >
    requires( ( Node< L > || Node< R > ) && Leafable< L > && Leafable< R > )
[[nodiscard]] constexpr auto operator+( const L& l, const R& r ) noexcept
{
    auto nl = as_node( l );
    auto nr = as_node( r );
    return Add< decltype( nl ), decltype( nr ) >{ {}, nl, nr };
}

template < typename L, typename R >
    requires( ( Node< L > || Node< R > ) && Leafable< L > && Leafable< R > )
[[nodiscard]] constexpr auto operator-( const L& l, const R& r ) noexcept
{
    auto nl = as_node( l );
    auto nr = as_node( r );
    return Sub< decltype( nl ), decltype( nr ) >{ {}, nl, nr };
}

template < Node A >
[[nodiscard]] constexpr Neg< A > operator-( const A& a ) noexcept
{
    return Neg< A >{ {}, a };
}

template < Node A >
[[nodiscard]] constexpr Scale< A > operator*( const A& a, typename A::scalar_type s ) noexcept
{
    return Scale< A >{ {}, a, s };
}
template < Node A >
[[nodiscard]] constexpr Scale< A > operator*( typename A::scalar_type s, const A& a ) noexcept
{
    return Scale< A >{ {}, a, s };
}
template < Node A >
[[nodiscard]] constexpr Scale< A > operator/( const A& a, typename A::scalar_type s ) noexcept
{
    return Scale< A >{ {}, a, typename A::scalar_type{ 1 } / s };
}

template < Node A >
[[nodiscard]] constexpr ConstantShift< A > operator+( const A& a,
                                                      typename A::scalar_type s ) noexcept
{
    return ConstantShift< A >{ {}, a, s };
}
template < Node A >
[[nodiscard]] constexpr ConstantShift< A > operator+( typename A::scalar_type s,
                                                      const A& a ) noexcept
{
    return ConstantShift< A >{ {}, a, s };
}
template < Node A >
[[nodiscard]] constexpr ConstantShift< A > operator-( const A& a,
                                                      typename A::scalar_type s ) noexcept
{
    return ConstantShift< A >{ {}, a, -s };
}

}  // namespace tax::fused

namespace tax
{
using fused::fuse;
}  // namespace tax
