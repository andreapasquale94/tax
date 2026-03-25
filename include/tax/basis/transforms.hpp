#pragma once

#include <Eigen/Core>
#include <array>
#include <tax/utils/aliases.hpp>
#include <tax/utils/combinatorics.hpp>
#include <tax/utils/enumeration.hpp>
#include <tax/utils/fwd.hpp>

namespace tax::detail
{

// =============================================================================
// Univariate Chebyshev <-> Monomial conversion matrices
// =============================================================================

/**
 * @brief Compute the (N+1)x(N+1) matrix converting Chebyshev to monomial coefficients.
 * @details Row i, column j gives the coefficient of x^i in T_j(x).
 *          Uses the recurrence T_0=1, T_1=x, T_n = 2x*T_{n-1} - T_{n-2}.
 */
template < typename T, int N >
[[nodiscard]] static TransformMatrix< T, N > chebyshevToMonomialMatrix()
{
    TransformMatrix< T, N > mat = TransformMatrix< T, N >::Zero();

    // T_0(x) = 1
    mat( 0, 0 ) = T{ 1 };

    if constexpr ( N >= 1 )
    {
        // T_1(x) = x
        mat( 1, 1 ) = T{ 1 };
    }

    // T_n(x) = 2x * T_{n-1}(x) - T_{n-2}(x)
    for ( int n = 2; n <= N; ++n )
    {
        // Use col() expressions to avoid aliasing issues at -O3
        mat.col( n ) = -mat.col( n - 2 );
        mat.col( n ).tail( N ) += T{ 2 } * mat.col( n - 1 ).head( N );
    }
    return mat;
}

/**
 * @brief Compute the (N+1)x(N+1) matrix converting monomial to Chebyshev coefficients.
 * @details This is the inverse of chebyshevToMonomialMatrix.
 */
template < typename T, int N >
[[nodiscard]] static TransformMatrix< T, N > monomialToChebyshevMatrix()
{
    return chebyshevToMonomialMatrix< T, N >().inverse();
}

// =============================================================================
// Univariate Legendre <-> Monomial conversion matrices
// =============================================================================

/**
 * @brief Compute the (N+1)x(N+1) matrix converting Legendre to monomial coefficients.
 * @details Row i, column j gives the coefficient of x^i in P_j(x).
 *          Uses the recurrence P_0=1, P_1=x, P_n = ((2n-1)/n)*x*P_{n-1} - ((n-1)/n)*P_{n-2}.
 */
template < typename T, int N >
[[nodiscard]] static TransformMatrix< T, N > legendreToMonomialMatrix()
{
    TransformMatrix< T, N > mat = TransformMatrix< T, N >::Zero();

    // P_0(x) = 1
    mat( 0, 0 ) = T{ 1 };

    if constexpr ( N >= 1 )
    {
        // P_1(x) = x
        mat( 1, 1 ) = T{ 1 };
    }

    // P_n(x) = ((2n-1)/n) * x * P_{n-1}(x) - ((n-1)/n) * P_{n-2}(x)
    for ( int n = 2; n <= N; ++n )
    {
        const T a = T( 2 * n - 1 ) / T( n );
        const T b = T( n - 1 ) / T( n );
        // Use col() expressions to avoid aliasing issues at -O3
        mat.col( n ) = -b * mat.col( n - 2 );
        mat.col( n ).tail( N ) += a * mat.col( n - 1 ).head( N );
    }
    return mat;
}

/**
 * @brief Compute the (N+1)x(N+1) matrix converting monomial to Legendre coefficients.
 * @details This is the inverse of legendreToMonomialMatrix.
 */
template < typename T, int N >
[[nodiscard]] static TransformMatrix< T, N > monomialToLegendreMatrix()
{
    return legendreToMonomialMatrix< T, N >().inverse();
}

// =============================================================================
// Univariate Hermite (probabilist's) <-> Monomial conversion matrices
// =============================================================================

/**
 * @brief Compute the (N+1)x(N+1) matrix converting probabilist's Hermite to monomial
 * coefficients.
 * @details Row i, column j gives the coefficient of x^i in He_j(x).
 *          Uses the recurrence He_0=1, He_1=x, He_n = x*He_{n-1} - (n-1)*He_{n-2}.
 */
template < typename T, int N >
[[nodiscard]] static TransformMatrix< T, N > hermiteToMonomialMatrix()
{
    TransformMatrix< T, N > mat = TransformMatrix< T, N >::Zero();

    // He_0(x) = 1
    mat( 0, 0 ) = T{ 1 };

    if constexpr ( N >= 1 )
    {
        // He_1(x) = x
        mat( 1, 1 ) = T{ 1 };
    }

    // He_n(x) = x * He_{n-1}(x) - (n-1) * He_{n-2}(x)
    for ( int n = 2; n <= N; ++n )
    {
        // Use col() expressions to avoid aliasing issues at -O3
        mat.col( n ) = -T( n - 1 ) * mat.col( n - 2 );
        mat.col( n ).tail( N ) += mat.col( n - 1 ).head( N );
    }
    return mat;
}

/**
 * @brief Compute the (N+1)x(N+1) matrix converting monomial to probabilist's Hermite
 * coefficients.
 * @details This is the inverse of hermiteToMonomialMatrix.
 */
template < typename T, int N >
[[nodiscard]] static TransformMatrix< T, N > monomialToHermiteMatrix()
{
    return hermiteToMonomialMatrix< T, N >().inverse();
}

// =============================================================================
// Apply univariate transform via Eigen matrix-vector multiply
// =============================================================================

/**
 * @brief Apply a (N+1)x(N+1) transformation matrix to a coefficient vector.
 * @details Maps the std::array coefficient vector to Eigen for the multiply.
 */
template < typename T, int S >
static void applyTransformMatrix( std::array< T, S >& out,
                                  const std::array< T, S >& in,
                                  const la::MatNT< T, S >& mat ) noexcept
{
    Eigen::Map< la::VecNT< T, S > > out_map( out.data() );
    Eigen::Map< const la::VecNT< T, S > > in_map( in.data() );
    out_map.noalias() = mat * in_map;
}

// =============================================================================
// Multivariate transforms via tensor product (dimension-by-dimension)
// =============================================================================

/**
 * @brief Apply a dimension-by-dimension basis transform to multivariate coefficients.
 * @details For M=1, delegates to applyTransformMatrix.
 *          For M>1, applies the univariate matrix along each variable axis using ping-pong
 *          buffers.
 * @param mat The (N+1)x(N+1) univariate transformation matrix.
 */
template < typename T, int N, int M >
static void applyMultivariateTransform( CoeffArray< T, N, M >& out,
                                        const CoeffArray< T, N, M >& in,
                                        const TransformMatrix< T, N >& mat ) noexcept
{
    constexpr auto nC = numMonomials( N, M );

    if constexpr ( M == 1 )
    {
        applyTransformMatrix< T, static_cast< int >( nC ) >( out, in, mat );
    }
    else
    {
        std::array< T, nC > buf_a = in;
        std::array< T, nC > buf_b{};

        for ( int var = 0; var < M; ++var )
        {
            auto& src = ( var % 2 == 0 ) ? buf_a : buf_b;
            auto& dst = ( var % 2 == 0 ) ? buf_b : buf_a;
            for ( std::size_t i = 0; i < nC; ++i ) dst[i] = T{};

            for ( std::size_t i = 0; i < nC; ++i )
            {
                if ( src[i] == T{} ) continue;
                auto alpha = unflatIndex< M >( i );
                const int j_src = alpha[var];

                for ( int i_dst = 0; i_dst <= N; ++i_dst )
                {
                    const T mat_val = mat( i_dst, j_src );
                    if ( mat_val == T{} ) continue;

                    auto beta = alpha;
                    beta[var] = i_dst;
                    if ( totalDegree< M >( beta ) > N ) continue;
                    dst[flatIndex< M >( beta )] += mat_val * src[i];
                }
            }

            src = dst;
        }

        out = ( M % 2 == 0 ) ? buf_a : buf_b;
    }
}

// =============================================================================
// Chebyshev <-> Monomial multivariate transforms
// =============================================================================

template < typename T, int N, int M >
static void chebyshevToMonomial( CoeffArray< T, N, M >& out,
                                 const CoeffArray< T, N, M >& in ) noexcept
{
    static const auto mat = chebyshevToMonomialMatrix< T, N >();
    applyMultivariateTransform< T, N, M >( out, in, mat );
}

template < typename T, int N, int M >
static void monomialToChebyshev( CoeffArray< T, N, M >& out,
                                 const CoeffArray< T, N, M >& in ) noexcept
{
    static const auto mat = monomialToChebyshevMatrix< T, N >();
    applyMultivariateTransform< T, N, M >( out, in, mat );
}

// =============================================================================
// Legendre <-> Monomial multivariate transforms
// =============================================================================

template < typename T, int N, int M >
static void legendreToMonomial( CoeffArray< T, N, M >& out,
                                const CoeffArray< T, N, M >& in ) noexcept
{
    static const auto mat = legendreToMonomialMatrix< T, N >();
    applyMultivariateTransform< T, N, M >( out, in, mat );
}

template < typename T, int N, int M >
static void monomialToLegendre( CoeffArray< T, N, M >& out,
                                const CoeffArray< T, N, M >& in ) noexcept
{
    static const auto mat = monomialToLegendreMatrix< T, N >();
    applyMultivariateTransform< T, N, M >( out, in, mat );
}

// =============================================================================
// Hermite <-> Monomial multivariate transforms
// =============================================================================

template < typename T, int N, int M >
static void hermiteToMonomial( CoeffArray< T, N, M >& out,
                               const CoeffArray< T, N, M >& in ) noexcept
{
    static const auto mat = hermiteToMonomialMatrix< T, N >();
    applyMultivariateTransform< T, N, M >( out, in, mat );
}

template < typename T, int N, int M >
static void monomialToHermite( CoeffArray< T, N, M >& out,
                               const CoeffArray< T, N, M >& in ) noexcept
{
    static const auto mat = monomialToHermiteMatrix< T, N >();
    applyMultivariateTransform< T, N, M >( out, in, mat );
}

}  // namespace tax::detail
