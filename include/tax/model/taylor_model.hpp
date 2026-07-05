#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <tax/core/multi_index.hpp>
#include <tax/core/scheme/isotropic.hpp>
#include <tax/core/storage/dense.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/model/bounders.hpp>
#include <tax/model/interval.hpp>
#include <utility>

namespace tax::model
{

// ===========================================================================
// Taylor models (Makino, ch. 4: "Remainder-Enhanced DA Operations")
//
// A Taylor model T = (P, I) over the parameter alpha = (N, x0, [a, b])
// encloses a function f on the domain [a, b]:
//
//     f(x)  in  P(x - x0) + I     for all x in [a, b].
//
// The polynomial part is an ordinary dense TaylorExpansion in the
// displacement dx = x - x0; the remainder bound I is an Interval.
//
// Rigor contract: every *interval* computation (range bounds, remainder
// propagation, Lagrange remainders, domain checks) is outward-rounded and
// therefore a guaranteed enclosure. Floating-point rounding of the
// *polynomial coefficients* themselves (~1 ulp per coefficient operation)
// is NOT folded into the remainder — the bounds are rigorous in exact
// coefficient arithmetic, matching the presentation of ch. 4 (the
// coefficient-error sweeping of COSY's RD type, ch. 5, is future work).
// ===========================================================================

// The shared bounding machinery (DomainPowers, polyRangeBound,
// orderRangeBound, excessProductBound) and the exact quadratic bounder live
// in <tax/model/bounders.hpp>, in namespace tax::model::detail.

// ===========================================================================
// TaylorModel
// ===========================================================================

/// An order-N Taylor model of an M-variate function (Makino §4.3):
/// polynomial part + remainder bound interval + expansion point + domain.
template < std::floating_point T, int N, int M = 1 >
    requires( N >= 0 && M >= 1 )
class TaylorModel
{
   public:
    // ------------------------------------------------------------------
    // Associated types
    // ------------------------------------------------------------------
    using scheme = IsotropicScheme< N, M >;
    using scalar_type = T;
    using interval_type = Interval< T >;
    using Poly = TaylorExpansion< T, scheme, storage::Dense >;
    using Input = typename Poly::Input;
    using Point = std::array< T, std::size_t( M ) >;
    using Domain = std::array< Interval< T >, std::size_t( M ) >;

    // ------------------------------------------------------------------
    // Compile-time properties
    // ------------------------------------------------------------------
    static constexpr int order_v = N;
    static constexpr int vars_v = M;
    static constexpr std::size_t nCoefficients = scheme::nCoeff;

    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Zero function over the degenerate domain {0}^M.
    constexpr TaylorModel() noexcept = default;

    /// Full constructor (P, I, x0, [a, b]). Throws std::invalid_argument if
    /// the expansion point lies outside the domain.
    constexpr TaylorModel( Poly poly, Interval< T > remainder, Point x0, Domain dom )
        : poly_{ std::move( poly ) }, rem_{ remainder }, x0_{ x0 }, dom_{ dom }
    {
        for ( std::size_t i = 0; i < std::size_t( M ); ++i )
        {
            if ( !dom_[i].contains( x0_[i] ) )
                throw std::invalid_argument(
                    "tax::model::TaylorModel: expansion point outside domain" );
        }
    }

    /// Domain-agnostic constant `c` — the constant function, which is valid
    /// over *every* domain. Such a value carries no concrete expansion
    /// point/domain of its own; in any binary operation it adopts its
    /// partner's, and two concrete operands still require matching domains.
    /// This is what lets `TaylorModel` act as an Eigen scalar: the `0` and `1`
    /// literals Eigen synthesises (in `setZero`, `Identity`, reductions, …)
    /// have no domain, yet compose correctly with real domain-carrying models.
    /*implicit*/ constexpr TaylorModel( T c ) noexcept
        : poly_{ Poly::constant( c ) }, abstract_{ true }
    {
    }

    // ------------------------------------------------------------------
    // Named factories
    // ------------------------------------------------------------------

    /// Constant function `v` — exact, remainder [0, 0].
    [[nodiscard]] static constexpr TaylorModel constant( T v, const Point& x0, const Domain& dom )
    {
        return TaylorModel{ Poly::constant( v ), Interval< T >{}, x0, dom };
    }

    /// Univariate identity `x = x0 + dx` on `dom` — exact, remainder [0, 0].
    [[nodiscard]] static constexpr TaylorModel variable( T x0, Interval< T > dom )
        requires( M == 1 && N >= 1 )
    {
        return TaylorModel{ Poly::variable( x0 ), Interval< T >{}, Point{ x0 }, Domain{ dom } };
    }

    /// Coordinate variable `x_I = x0_I + dx_I` (compile-time index).
    template < int I >
    [[nodiscard]] static constexpr TaylorModel variable( const Point& x0, const Domain& dom )
        requires( N >= 1 && I >= 0 && I < M )
    {
        return TaylorModel{ Poly::template variable< I >( x0 ), Interval< T >{}, x0, dom };
    }

    /// Runtime-indexed coordinate variable. Throws std::out_of_range.
    [[nodiscard]] static TaylorModel variable( const Point& x0, const Domain& dom, int var_idx )
        requires( N >= 1 )
    {
        if ( var_idx < 0 || var_idx >= M )
            throw std::out_of_range( "tax::model::TaylorModel::variable: index out of range" );
        return TaylorModel{ Poly::variable( x0[std::size_t( var_idx )], var_idx ), Interval< T >{},
                            x0, dom };
    }

    /// All M coordinate variables at once.
    [[nodiscard]] static constexpr std::array< TaylorModel, std::size_t( M ) > variables(
        const Point& x0, const Domain& dom )
        requires( N >= 1 )
    {
        return [&]< std::size_t... Is >( std::index_sequence< Is... > ) {
            return std::array< TaylorModel, std::size_t( M ) >{
                variable< int( Is ) >( x0, dom )... };
        }( std::make_index_sequence< std::size_t( M ) >{} );
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    [[nodiscard]] constexpr const Poly& polynomial() const noexcept { return poly_; }
    [[nodiscard]] constexpr Poly& polynomial() noexcept { return poly_; }

    [[nodiscard]] constexpr const Interval< T >& remainder() const noexcept { return rem_; }
    [[nodiscard]] constexpr Interval< T >& remainder() noexcept { return rem_; }

    [[nodiscard]] constexpr const Point& expansionPoint() const noexcept { return x0_; }
    [[nodiscard]] constexpr const Domain& domain() const noexcept { return dom_; }

    /// Constant part of the polynomial (COSY's CONS).
    [[nodiscard]] constexpr T value() const noexcept { return poly_.value(); }

    /// Displacement domain D_i = [a_i - x0_i, b_i - x0_i]; always contains 0.
    [[nodiscard]] constexpr Domain displacementDomain() const noexcept
    {
        Domain d{};
        for ( std::size_t i = 0; i < std::size_t( M ); ++i ) d[i] = dom_[i] - x0_[i];
        return d;
    }

    /// True iff both models share the expansion point and domain (binary
    /// operations require this).
    [[nodiscard]] constexpr bool compatibleWith( const TaylorModel& o ) const noexcept
    {
        return x0_ == o.x0_ && dom_ == o.dom_;
    }

    /// True for a domain-agnostic constant (see the `TaylorModel(T)` ctor).
    [[nodiscard]] constexpr bool isAbstractConstant() const noexcept { return abstract_; }

    /// Re-home a domain-agnostic constant onto a concrete expansion point and
    /// domain (a constant is valid over any domain). The full constructor
    /// still validates `x0 in dom`.
    [[nodiscard]] constexpr TaylorModel overDomain( const Point& x0, const Domain& dom ) const
    {
        return TaylorModel{ poly_, rem_, x0, dom };
    }

    /// A copy flagged as a domain-agnostic constant — used to propagate the
    /// flag when two abstract constants are combined.
    [[nodiscard]] constexpr TaylorModel asAbstractConstant() const noexcept
    {
        TaylorModel r = *this;
        r.abstract_ = true;
        return r;
    }

    // ------------------------------------------------------------------
    // Range bounds
    // ------------------------------------------------------------------

    /// Rigorous range bound B(P) of the polynomial part over the domain.
    /// Defaults to the diagonal exact-quadratic bounder (§5.4.3), which is
    /// never wider than the naive order-sum; pass `Bounder::Naive` for the
    /// cheaper order-sum. Both are valid enclosures.
    [[nodiscard]] constexpr Interval< T > polynomialBound(
        Bounder which = Bounder::Quadratic ) const noexcept
    {
        const Domain disp = displacementDomain();
        const detail::DomainPowers< T, M, N > pows{ disp };
        return detail::rangeBound( which, poly_, pows, disp );
    }

    /// Per-order bound I^k: range of the homogeneous degree-k part (§5.4).
    [[nodiscard]] constexpr Interval< T > orderBound( int deg ) const noexcept
    {
        const detail::DomainPowers< T, M, N > pows{ displacementDomain() };
        return detail::orderRangeBound( poly_, deg, pows );
    }

    /// Total enclosure of the modeled function over the domain:
    /// B(P) + I (COSY's IN). See `polynomialBound` for the `which` strategy.
    [[nodiscard]] constexpr Interval< T > bound( Bounder which = Bounder::Quadratic ) const noexcept
    {
        return polynomialBound( which ) + rem_;
    }

    // ------------------------------------------------------------------
    // Evaluation
    // ------------------------------------------------------------------

    /// Guaranteed enclosure of f(x0 + dx): interval evaluation of P at dx,
    /// plus the remainder. Throws std::domain_error if dx leaves the
    /// displacement domain (the enclosure property only holds inside).
    [[nodiscard]] constexpr Interval< T > eval( const Input& dx ) const
    {
        const Domain disp = displacementDomain();
        Domain pt{};
        for ( std::size_t i = 0; i < std::size_t( M ); ++i )
        {
            if ( !disp[i].contains( dx[i] ) )
                throw std::domain_error(
                    "tax::model::TaylorModel::eval: point outside the domain" );
            pt[i] = Interval< T >{ dx[i] };
        }
        const detail::DomainPowers< T, M, N > pows{ pt };
        return detail::polyRangeBound( poly_, pows ) + rem_;
    }

    // ------------------------------------------------------------------
    // Antiderivation (4.12)
    // ------------------------------------------------------------------

    /// Taylor model of the indefinite integral from x0_I to x_I:
    /// the polynomial part integrates P_{N-1}; the freed order-N part joins
    /// the remainder, scaled by the displacement range of x_I.
    template < int I >
    [[nodiscard]] constexpr TaylorModel integ() const noexcept
        requires( I >= 0 && I < M )
    {
        return integImpl( poly_.template integ< I >(), I );
    }

    /// Runtime-indexed antiderivation. Throws std::out_of_range.
    [[nodiscard]] TaylorModel integ( int var ) const
    {
        if ( var < 0 || var >= M )
            throw std::out_of_range( "tax::model::TaylorModel::integ: var must be in [0, M)" );
        return integImpl( poly_.integ( var ), var );
    }

    // ------------------------------------------------------------------
    // Partial evaluation (variable fixing)
    // ------------------------------------------------------------------

    /// Evaluate variable `I` at the absolute coordinate `value`, keeping the
    /// other variables symbolic. The axis collapses exactly (the fixed
    /// coordinate is a point, so no remainder growth): every monomial in x_I
    /// folds into the lower-degree monomials, and the model becomes constant
    /// in variable I over the degenerate domain `[value, value]`. `value`
    /// must lie in the variable's domain, else the enclosure would not hold.
    ///
    /// This is the step-continuation primitive for a Taylor-model ODE
    /// integrator: fixing the time variable at the step endpoint yields the
    /// end-of-step state as a function of the remaining (initial-condition)
    /// variables. Throws std::domain_error if `value` is out of domain.
    template < int I >
    [[nodiscard]] constexpr TaylorModel fix( T value ) const
        requires( I >= 0 && I < M )
    {
        return fixImpl( I, value );
    }

    /// Runtime-indexed partial evaluation. Throws std::out_of_range /
    /// std::domain_error.
    [[nodiscard]] TaylorModel fix( int var, T value ) const
    {
        if ( var < 0 || var >= M )
            throw std::out_of_range( "tax::model::TaylorModel::fix: var must be in [0, M)" );
        return fixImpl( var, value );
    }

    /// Reset variable `I`'s expansion point and domain. Only valid when the
    /// model does not depend on variable I (all monomials containing x_I are
    /// zero) — e.g. after `fix` has collapsed that axis — so that reusing the
    /// slot as a fresh coordinate keeps the enclosure valid. This lets an ODE
    /// integrator recycle the (collapsed) time slot for the next step's
    /// domain [0, h]. Throws std::invalid_argument on violation.
    template < int I >
    [[nodiscard]] constexpr TaylorModel retarget( T x0_new, Interval< T > dom_new ) const
        requires( I >= 0 && I < M )
    {
        return retargetImpl( I, x0_new, dom_new );
    }

    /// Runtime-indexed retarget. Throws std::out_of_range /
    /// std::invalid_argument.
    [[nodiscard]] TaylorModel retarget( int var, T x0_new, Interval< T > dom_new ) const
    {
        if ( var < 0 || var >= M )
            throw std::out_of_range( "tax::model::TaylorModel::retarget: var must be in [0, M)" );
        return retargetImpl( var, x0_new, dom_new );
    }

   private:
    [[nodiscard]] constexpr TaylorModel fixImpl( int var, T value ) const
    {
        if ( !dom_[std::size_t( var )].contains( value ) )
            throw std::domain_error(
                "tax::model::TaylorModel::fix: value outside the variable's domain" );
        const T d = value - x0_[std::size_t( var )];
        Poly out{};
        for ( std::size_t k = 0; k < nCoefficients; ++k )
        {
            const T c = poly_[k];
            if ( c == T{ 0 } ) continue;
            auto alpha = scheme::multiOf( k );
            const int e = alpha[std::size_t( var )];
            T dp = T{ 1 };
            for ( int j = 0; j < e; ++j ) dp *= d;
            alpha[std::size_t( var )] = 0;
            out[scheme::flatOf( alpha )] += c * dp;
        }
        TaylorModel r{};
        r.poly_ = out;
        r.rem_ = rem_;  // fixing to a point does not enlarge the remainder
        r.x0_ = x0_;
        r.x0_[std::size_t( var )] = value;
        r.dom_ = dom_;
        r.dom_[std::size_t( var )] = Interval< T >{ value };
        return r;
    }

    [[nodiscard]] constexpr TaylorModel retargetImpl( int var, T x0_new,
                                                      Interval< T > dom_new ) const
    {
        for ( std::size_t k = 0; k < nCoefficients; ++k )
        {
            if ( poly_[k] != T{ 0 } && scheme::multiOf( k )[std::size_t( var )] > 0 )
                throw std::invalid_argument(
                    "tax::model::TaylorModel::retarget: model depends on the variable" );
        }
        if ( !dom_new.contains( x0_new ) )
            throw std::invalid_argument(
                "tax::model::TaylorModel::retarget: expansion point outside new domain" );
        TaylorModel r = *this;
        r.x0_[std::size_t( var )] = x0_new;
        r.dom_[std::size_t( var )] = dom_new;
        return r;
    }

    [[nodiscard]] constexpr TaylorModel integImpl( const Poly& raw_integral,
                                                   int var ) const noexcept
    {
        // TE::integ integrates every monomial that stays inside order N, i.e.
        // exactly the P_{N-1} part; the order-N block of `poly_` is dropped
        // there and accounted for here instead:
        //   I_result = (B(P_N - P_{N-1}) + I) * hull(0, D_var)   (cf. (4.12)).
        const Domain disp = displacementDomain();
        const detail::DomainPowers< T, M, N > pows{ disp };
        const Interval< T > top = detail::orderRangeBound( poly_, N, pows );
        const Interval< T > span = hull( Interval< T >{}, disp[std::size_t( var )] );

        TaylorModel r{};
        r.poly_ = raw_integral;
        r.rem_ = ( top + rem_ ) * span;
        r.x0_ = x0_;
        r.dom_ = dom_;
        return r;
    }

    Poly poly_{};
    Interval< T > rem_{};
    Point x0_{};
    Domain dom_{};
    /// Domain-agnostic constant marker (Eigen-scalar literals); see ctor.
    bool abstract_ = false;
};

// ---------------------------------------------------------------------------
// Convenience alias
// ---------------------------------------------------------------------------

/// `TM<N, M>` — order-N, M-variate `double` Taylor model.
template < int N, int M = 1 >
using TM = TaylorModel< double, N, M >;

}  // namespace tax::model
