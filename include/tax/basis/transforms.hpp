#pragma once

#include <Eigen/Core>
#include <array>
#include <cmath>
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
[[nodiscard]] static constexpr TransformMatrix< T, N > chebyshevToMonomialMatrix()
{
    constexpr int S = N + 1;
    TransformMatrix< T, N > mat{};

    // T_0(x) = 1
    mat[0 * S + 0] = T{ 1 };

    if constexpr ( N >= 1 )
    {
        // T_1(x) = x
        mat[1 * S + 1] = T{ 1 };
    }

    // T_n(x) = 2x * T_{n-1}(x) - T_{n-2}(x)
    for ( int n = 2; n <= N; ++n )
    {
        // -T_{n-2}
        for ( int i = 0; i <= N; ++i ) mat[i * S + n] = -mat[i * S + ( n - 2 )];
        // + 2x * T_{n-1}  (shift up by 1 in power of x)
        for ( int i = 1; i <= N; ++i ) mat[i * S + n] += T{ 2 } * mat[( i - 1 ) * S + ( n - 1 )];
    }
    return mat;
}

/**
 * @brief Compute the (N+1)x(N+1) matrix converting monomial to Chebyshev coefficients.
 * @details This is the inverse of chebyshevToMonomialMatrix.
 *          Computed by Gauss elimination (constexpr-compatible).
 */
template < typename T, int N >
[[nodiscard]] static constexpr TransformMatrix< T, N > monomialToChebyshevMatrix()
{
    constexpr int S = N + 1;

    // Start with [C2M | I] and do Gauss-Jordan to get [I | C2M^{-1}]
    auto c2m = chebyshevToMonomialMatrix< T, N >();

    // Augmented matrix [c2m | identity]
    std::array< T, S * 2 * S > aug{};
    for ( int i = 0; i < S; ++i )
    {
        for ( int j = 0; j < S; ++j )
        {
            aug[i * ( 2 * S ) + j] = c2m[i * S + j];
            aug[i * ( 2 * S ) + S + j] = ( i == j ) ? T{ 1 } : T{ 0 };
        }
    }

    // Forward elimination with partial pivoting
    for ( int col = 0; col < S; ++col )
    {
        // Find pivot
        int pivot = col;
        T best = ( aug[col * ( 2 * S ) + col] < T{ 0 } ) ? -aug[col * ( 2 * S ) + col]
                                                            : aug[col * ( 2 * S ) + col];
        for ( int row = col + 1; row < S; ++row )
        {
            T val = ( aug[row * ( 2 * S ) + col] < T{ 0 } ) ? -aug[row * ( 2 * S ) + col]
                                                              : aug[row * ( 2 * S ) + col];
            if ( val > best )
            {
                best = val;
                pivot = row;
            }
        }

        // Swap rows
        if ( pivot != col )
        {
            for ( int j = 0; j < 2 * S; ++j )
            {
                T tmp = aug[col * ( 2 * S ) + j];
                aug[col * ( 2 * S ) + j] = aug[pivot * ( 2 * S ) + j];
                aug[pivot * ( 2 * S ) + j] = tmp;
            }
        }

        // Scale pivot row
        T scale = T{ 1 } / aug[col * ( 2 * S ) + col];
        for ( int j = 0; j < 2 * S; ++j ) aug[col * ( 2 * S ) + j] *= scale;

        // Eliminate
        for ( int row = 0; row < S; ++row )
        {
            if ( row == col ) continue;
            T factor = aug[row * ( 2 * S ) + col];
            if ( factor == T{ 0 } ) continue;
            for ( int j = 0; j < 2 * S; ++j )
                aug[row * ( 2 * S ) + j] -= factor * aug[col * ( 2 * S ) + j];
        }
    }

    // Extract inverse
    TransformMatrix< T, N > inv{};
    for ( int i = 0; i < S; ++i )
        for ( int j = 0; j < S; ++j ) inv[i * S + j] = aug[i * ( 2 * S ) + S + j];

    return inv;
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
[[nodiscard]] static constexpr TransformMatrix< T, N > legendreToMonomialMatrix()
{
    constexpr int S = N + 1;
    TransformMatrix< T, N > mat{};

    // P_0(x) = 1
    mat[0 * S + 0] = T{ 1 };

    if constexpr ( N >= 1 )
    {
        // P_1(x) = x
        mat[1 * S + 1] = T{ 1 };
    }

    // P_n(x) = ((2n-1)/n) * x * P_{n-1}(x) - ((n-1)/n) * P_{n-2}(x)
    for ( int n = 2; n <= N; ++n )
    {
        const T a = T( 2 * n - 1 ) / T( n );
        const T b = T( n - 1 ) / T( n );
        // -((n-1)/n) * P_{n-2}
        for ( int i = 0; i <= N; ++i ) mat[i * S + n] = -b * mat[i * S + ( n - 2 )];
        // + ((2n-1)/n) * x * P_{n-1}  (shift up by 1 in power of x)
        for ( int i = 1; i <= N; ++i ) mat[i * S + n] += a * mat[( i - 1 ) * S + ( n - 1 )];
    }
    return mat;
}

/**
 * @brief Compute the (N+1)x(N+1) matrix converting monomial to Legendre coefficients.
 * @details This is the inverse of legendreToMonomialMatrix.
 */
template < typename T, int N >
[[nodiscard]] static constexpr TransformMatrix< T, N > monomialToLegendreMatrix()
{
    constexpr int S = N + 1;

    auto l2m = legendreToMonomialMatrix< T, N >();

    std::array< T, S * 2 * S > aug{};
    for ( int i = 0; i < S; ++i )
    {
        for ( int j = 0; j < S; ++j )
        {
            aug[i * ( 2 * S ) + j] = l2m[i * S + j];
            aug[i * ( 2 * S ) + S + j] = ( i == j ) ? T{ 1 } : T{ 0 };
        }
    }

    for ( int col = 0; col < S; ++col )
    {
        int pivot = col;
        T best = ( aug[col * ( 2 * S ) + col] < T{ 0 } ) ? -aug[col * ( 2 * S ) + col]
                                                            : aug[col * ( 2 * S ) + col];
        for ( int row = col + 1; row < S; ++row )
        {
            T val = ( aug[row * ( 2 * S ) + col] < T{ 0 } ) ? -aug[row * ( 2 * S ) + col]
                                                              : aug[row * ( 2 * S ) + col];
            if ( val > best )
            {
                best = val;
                pivot = row;
            }
        }

        if ( pivot != col )
        {
            for ( int j = 0; j < 2 * S; ++j )
            {
                T tmp = aug[col * ( 2 * S ) + j];
                aug[col * ( 2 * S ) + j] = aug[pivot * ( 2 * S ) + j];
                aug[pivot * ( 2 * S ) + j] = tmp;
            }
        }

        T scale = T{ 1 } / aug[col * ( 2 * S ) + col];
        for ( int j = 0; j < 2 * S; ++j ) aug[col * ( 2 * S ) + j] *= scale;

        for ( int row = 0; row < S; ++row )
        {
            if ( row == col ) continue;
            T factor = aug[row * ( 2 * S ) + col];
            if ( factor == T{ 0 } ) continue;
            for ( int j = 0; j < 2 * S; ++j )
                aug[row * ( 2 * S ) + j] -= factor * aug[col * ( 2 * S ) + j];
        }
    }

    TransformMatrix< T, N > inv{};
    for ( int i = 0; i < S; ++i )
        for ( int j = 0; j < S; ++j ) inv[i * S + j] = aug[i * ( 2 * S ) + S + j];

    return inv;
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
[[nodiscard]] static constexpr TransformMatrix< T, N > hermiteToMonomialMatrix()
{
    constexpr int S = N + 1;
    TransformMatrix< T, N > mat{};

    // He_0(x) = 1
    mat[0 * S + 0] = T{ 1 };

    if constexpr ( N >= 1 )
    {
        // He_1(x) = x
        mat[1 * S + 1] = T{ 1 };
    }

    // He_n(x) = x * He_{n-1}(x) - (n-1) * He_{n-2}(x)
    for ( int n = 2; n <= N; ++n )
    {
        // -(n-1) * He_{n-2}
        for ( int i = 0; i <= N; ++i ) mat[i * S + n] = -T( n - 1 ) * mat[i * S + ( n - 2 )];
        // + x * He_{n-1}  (shift up by 1 in power of x)
        for ( int i = 1; i <= N; ++i ) mat[i * S + n] += mat[( i - 1 ) * S + ( n - 1 )];
    }
    return mat;
}

/**
 * @brief Compute the (N+1)x(N+1) matrix converting monomial to probabilist's Hermite
 * coefficients.
 * @details This is the inverse of hermiteToMonomialMatrix.
 */
template < typename T, int N >
[[nodiscard]] static constexpr TransformMatrix< T, N > monomialToHermiteMatrix()
{
    constexpr int S = N + 1;

    auto h2m = hermiteToMonomialMatrix< T, N >();

    std::array< T, S * 2 * S > aug{};
    for ( int i = 0; i < S; ++i )
    {
        for ( int j = 0; j < S; ++j )
        {
            aug[i * ( 2 * S ) + j] = h2m[i * S + j];
            aug[i * ( 2 * S ) + S + j] = ( i == j ) ? T{ 1 } : T{ 0 };
        }
    }

    for ( int col = 0; col < S; ++col )
    {
        int pivot = col;
        T best = ( aug[col * ( 2 * S ) + col] < T{ 0 } ) ? -aug[col * ( 2 * S ) + col]
                                                            : aug[col * ( 2 * S ) + col];
        for ( int row = col + 1; row < S; ++row )
        {
            T val = ( aug[row * ( 2 * S ) + col] < T{ 0 } ) ? -aug[row * ( 2 * S ) + col]
                                                              : aug[row * ( 2 * S ) + col];
            if ( val > best )
            {
                best = val;
                pivot = row;
            }
        }

        if ( pivot != col )
        {
            for ( int j = 0; j < 2 * S; ++j )
            {
                T tmp = aug[col * ( 2 * S ) + j];
                aug[col * ( 2 * S ) + j] = aug[pivot * ( 2 * S ) + j];
                aug[pivot * ( 2 * S ) + j] = tmp;
            }
        }

        T scale = T{ 1 } / aug[col * ( 2 * S ) + col];
        for ( int j = 0; j < 2 * S; ++j ) aug[col * ( 2 * S ) + j] *= scale;

        for ( int row = 0; row < S; ++row )
        {
            if ( row == col ) continue;
            T factor = aug[row * ( 2 * S ) + col];
            if ( factor == T{ 0 } ) continue;
            for ( int j = 0; j < 2 * S; ++j )
                aug[row * ( 2 * S ) + j] -= factor * aug[col * ( 2 * S ) + j];
        }
    }

    TransformMatrix< T, N > inv{};
    for ( int i = 0; i < S; ++i )
        for ( int j = 0; j < S; ++j ) inv[i * S + j] = aug[i * ( 2 * S ) + S + j];

    return inv;
}

// =============================================================================
// Apply univariate transform via matrix-vector multiply
// =============================================================================

/**
 * @brief Apply a (N+1)x(N+1) matrix to a coefficient vector of length N+1.
 * @details Uses Eigen::Map for matrix-vector multiply at runtime; falls back
 *          to a manual loop in constexpr contexts.
 */
template < typename T, int S >
static constexpr void applyTransformMatrix( std::array< T, S >& out,
                                            const std::array< T, S >& in,
                                            const std::array< T, S * S >& mat ) noexcept
{
    if consteval
    {
        for ( int i = 0; i < S; ++i )
        {
            T sum{};
            for ( int j = 0; j < S; ++j ) sum += mat[i * S + j] * in[j];
            out[i] = sum;
        }
    }
    else
    {
        using Vec = Eigen::Vector< T, S >;
        using Mat = Eigen::Matrix< T, S, S, Eigen::RowMajor >;
        Eigen::Map< Vec > out_map( out.data() );
        Eigen::Map< const Vec > in_map( in.data() );
        Eigen::Map< const Mat > mat_map( mat.data() );
        out_map.noalias() = mat_map * in_map;
    }
}

// =============================================================================
// Multivariate transforms via tensor product (dimension-by-dimension)
// =============================================================================

/**
 * @brief Convert multivariate Chebyshev coefficients to monomial coefficients.
 * @details Applies the univariate Chebyshev→monomial transform dimension by dimension.
 *          For M=1 this is a simple matrix-vector multiply.
 *          For M>1, we iterate over all multi-indices and transform along each variable axis.
 */
template < typename T, int N, int M >
static constexpr void chebyshevToMonomial(
    CoeffArray< T, N, M >& out, const CoeffArray< T, N, M >& in ) noexcept
{
    constexpr auto nC = numMonomials( N, M );

    if constexpr ( M == 1 )
    {
        constexpr auto mat = chebyshevToMonomialMatrix< T, N >();
        applyTransformMatrix< T, static_cast< int >( nC ) >( out, in, mat );
    }
    else
    {
        // Dimension-by-dimension transform.
        // We transform along variable 0 first, then variable 1, etc.
        // Use two buffers and ping-pong between them.
        constexpr auto mat = chebyshevToMonomialMatrix< T, N >();

        std::array< T, nC > buf_a = in;
        std::array< T, nC > buf_b{};

        for ( int var = 0; var < M; ++var )
        {
            auto& src = ( var % 2 == 0 ) ? buf_a : buf_b;
            auto& dst = ( var % 2 == 0 ) ? buf_b : buf_a;
            for ( std::size_t i = 0; i < nC; ++i ) dst[i] = T{};

            // For each multi-index alpha in the destination, accumulate contributions
            // from all source multi-indices that differ only in the `var` component.
            for ( std::size_t i = 0; i < nC; ++i )
            {
                if ( src[i] == T{} ) continue;
                auto alpha = unflatIndex< M >( i );
                const int j_src = alpha[var];  // source degree along `var`

                // Transform along dimension `var`: replace degree j_src with
                // contributions to all degrees i_dst via the C2M matrix row
                for ( int i_dst = 0; i_dst <= N; ++i_dst )
                {
                    const T mat_val = mat[i_dst * ( N + 1 ) + j_src];
                    if ( mat_val == T{} ) continue;

                    auto beta = alpha;
                    beta[var] = i_dst;
                    if ( totalDegree< M >( beta ) > N ) continue;
                    dst[flatIndex< M >( beta )] += mat_val * src[i];
                }
            }

            src = dst;  // ensure result ends up in the right buffer for next iteration
        }

        // Result is in the buffer that was last written to
        out = ( M % 2 == 0 ) ? buf_a : buf_b;
    }
}

/**
 * @brief Convert multivariate monomial coefficients to Chebyshev coefficients.
 */
template < typename T, int N, int M >
static constexpr void monomialToChebyshev(
    CoeffArray< T, N, M >& out, const CoeffArray< T, N, M >& in ) noexcept
{
    constexpr auto nC = numMonomials( N, M );

    if constexpr ( M == 1 )
    {
        constexpr auto mat = monomialToChebyshevMatrix< T, N >();
        applyTransformMatrix< T, static_cast< int >( nC ) >( out, in, mat );
    }
    else
    {
        constexpr auto mat = monomialToChebyshevMatrix< T, N >();

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
                    const T mat_val = mat[i_dst * ( N + 1 ) + j_src];
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
// Multivariate Legendre <-> Monomial transforms
// =============================================================================

template < typename T, int N, int M >
static constexpr void legendreToMonomial(
    CoeffArray< T, N, M >& out, const CoeffArray< T, N, M >& in ) noexcept
{
    constexpr auto nC = numMonomials( N, M );

    if constexpr ( M == 1 )
    {
        constexpr auto mat = legendreToMonomialMatrix< T, N >();
        applyTransformMatrix< T, static_cast< int >( nC ) >( out, in, mat );
    }
    else
    {
        constexpr auto mat = legendreToMonomialMatrix< T, N >();

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
                    const T mat_val = mat[i_dst * ( N + 1 ) + j_src];
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

template < typename T, int N, int M >
static constexpr void monomialToLegendre(
    CoeffArray< T, N, M >& out, const CoeffArray< T, N, M >& in ) noexcept
{
    constexpr auto nC = numMonomials( N, M );

    if constexpr ( M == 1 )
    {
        constexpr auto mat = monomialToLegendreMatrix< T, N >();
        applyTransformMatrix< T, static_cast< int >( nC ) >( out, in, mat );
    }
    else
    {
        constexpr auto mat = monomialToLegendreMatrix< T, N >();

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
                    const T mat_val = mat[i_dst * ( N + 1 ) + j_src];
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
// Multivariate Hermite <-> Monomial transforms
// =============================================================================

template < typename T, int N, int M >
static constexpr void hermiteToMonomial(
    CoeffArray< T, N, M >& out, const CoeffArray< T, N, M >& in ) noexcept
{
    constexpr auto nC = numMonomials( N, M );

    if constexpr ( M == 1 )
    {
        constexpr auto mat = hermiteToMonomialMatrix< T, N >();
        applyTransformMatrix< T, static_cast< int >( nC ) >( out, in, mat );
    }
    else
    {
        constexpr auto mat = hermiteToMonomialMatrix< T, N >();

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
                    const T mat_val = mat[i_dst * ( N + 1 ) + j_src];
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

template < typename T, int N, int M >
static constexpr void monomialToHermite(
    CoeffArray< T, N, M >& out, const CoeffArray< T, N, M >& in ) noexcept
{
    constexpr auto nC = numMonomials( N, M );

    if constexpr ( M == 1 )
    {
        constexpr auto mat = monomialToHermiteMatrix< T, N >();
        applyTransformMatrix< T, static_cast< int >( nC ) >( out, in, mat );
    }
    else
    {
        constexpr auto mat = monomialToHermiteMatrix< T, N >();

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
                    const T mat_val = mat[i_dst * ( N + 1 ) + j_src];
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

}  // namespace tax::detail
