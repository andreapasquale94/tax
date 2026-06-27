// Micro-benchmarks for the dense TaylorExpansion hot paths.
//
// Self-contained (only <chrono>); no external benchmark dependency. Build with
// -DTAX_BUILD_BENCHMARKS=ON and run the `bench_taylor` target. Numbers are
// ns/op; pin to a core (e.g. `taskset -c 0 ./bench_taylor`) for stable results.
//
// Coverage targets the two regimes the library is used in:
//   * high-order univariate          (N = 20, 30; M = 1)
//   * many-variable mid/high order    (M = 6, 9; N = 4..8)

#include <chrono>
#include <cstdio>
#include <string>
#include <tax/experimental/fused.hpp>
#include <tax/tax.hpp>

namespace
{
using clk = std::chrono::steady_clock;

template < typename T >
[[gnu::noinline]] void sink( const T& v )
{
#if defined( __GNUC__ )
    asm volatile( "" : : "r,m"( &v ) : "memory" );
#else
    volatile auto x = &v;
    (void)x;
#endif
}

template < typename F >
double bench( std::size_t iters, F&& f )
{
    for ( std::size_t i = 0; i < iters / 10 + 1; ++i ) f( i );
    const auto t0 = clk::now();
    for ( std::size_t i = 0; i < iters; ++i ) f( i );
    const auto t1 = clk::now();
    return std::chrono::duration< double, std::nano >( t1 - t0 ).count() / double( iters );
}

inline double jitter( std::size_t i ) { return 0.1 + 1e-6 * double( i & 1023 ); }

void row( const std::string& name, double ns )
{
    std::printf( "%-26s %12.2f ns/op\n", name.c_str(), ns );
}

template < int N >
void uni()
{
    using E = tax::TEn< N, 1 >;
    const auto mk = [&]( double j ) { return E::variable( 0.4 + j ); };
    const std::size_t it = 200000;
    const std::string t = " N" + std::to_string( N ) + " M1";
    row( "mul" + t, bench( it, [&]( std::size_t i ) {
             E a = mk( jitter( i ) ), b = mk( jitter( i ) + 0.01 );
             sink( a * b );
         } ) );
    row( "div" + t, bench( it, [&]( std::size_t i ) {
             E a = mk( jitter( i ) ), b = mk( jitter( i ) + 0.01 );
             sink( a / ( b + 2.0 ) );
         } ) );
    row( "exp" + t, bench( it, [&]( std::size_t i ) {
             E a = mk( jitter( i ) );
             sink( exp( a ) );
         } ) );
    row( "composite" + t, bench( it, [&]( std::size_t i ) {
             E x = mk( jitter( i ) );
             sink( sin( x ) * exp( x ) + sqrt( x * x + 1.0 ) );
         } ) );
}

template < int N, int M >
void multi( std::size_t it )
{
    using E = tax::TEn< N, M >;
    typename E::Input p{};
    for ( int k = 0; k < M; ++k ) p[std::size_t( k )] = 0.3 + 0.05 * k;
    const auto mk = [&]( double j, int w ) {
        typename E::Input pp = p;
        pp[0] += j + 0.01 * w;
        return E::template variable< 0 >( pp );
    };
    const std::string t = " N" + std::to_string( N ) + " M" + std::to_string( M );
    row( "mul" + t, bench( it, [&]( std::size_t i ) {
             E a = mk( jitter( i ), 0 ), b = mk( jitter( i ), 1 );
             sink( a * b );
         } ) );
    row( "exp" + t, bench( it, [&]( std::size_t i ) {
             E a = mk( jitter( i ), 0 );
             sink( exp( a ) );
         } ) );
    row( "composite" + t, bench( it, [&]( std::size_t i ) {
             E x = mk( jitter( i ), 0 ), y = mk( jitter( i ), 1 );
             sink( sin( x ) * exp( y ) + sqrt( x * x + 1.0 ) );
         } ) );
}

// Eager vs. fused (experimental expression templates) for an elementwise linear
// combination — the case where eager evaluation pays for one temporary per op.
template < int N, int M >
void fusion( std::size_t it )
{
    using tax::fuse;
    using E = tax::TEn< N, M >;
    typename E::Input p{};
    for ( int k = 0; k < M; ++k ) p[std::size_t( k )] = 0.3 + 0.05 * k;
    // Dense bases built once (transcendental results — fully populated, not the
    // foldable sparse `variable` pattern). The timed loop only perturbs the
    // constant term so the linear combination itself dominates and the
    // operand-construction cost is identical for both paths.
    typename E::Input pp = p;
    E a0 = exp( E::template variable< 0 >( pp ) );
    E b0 = exp( E::template variable< 0 >( ( pp[0] += 0.1, pp ) ) );
    E c0 = exp( E::template variable< 0 >( ( pp[0] += 0.1, pp ) ) );
    E d0 = exp( E::template variable< 0 >( ( pp[0] += 0.1, pp ) ) );

    const std::string t = " N" + std::to_string( N ) + " M" + std::to_string( M );
    row( "lincomb-eager" + t, bench( it, [&]( std::size_t i ) {
             E a = a0, b = b0, c = c0, d = d0;
             const double j = jitter( i );
             a[0] += j;
             b[0] -= j;  // cheap per-iteration variation, dense operands
             E r = 2.0 * a + 3.0 * b - c + 0.5 * d + 1.5;
             sink( r );
         } ) );
    row( "lincomb-fused" + t, bench( it, [&]( std::size_t i ) {
             E a = a0, b = b0, c = c0, d = d0;
             const double j = jitter( i );
             a[0] += j;
             b[0] -= j;
             E r = 2.0 * fuse( a ) + 3.0 * fuse( b ) - fuse( c ) + 0.5 * fuse( d ) + 1.5;
             sink( r );
         } ) );
}
}  // namespace

int main()
{
    std::printf( "=== tax dense micro-benchmarks ===\n" );
    uni< 20 >();
    uni< 30 >();
    multi< 4, 6 >( 100000 );
    multi< 6, 6 >( 40000 );
    multi< 8, 6 >( 8000 );
    multi< 4, 9 >( 40000 );
    multi< 6, 9 >( 6000 );
    multi< 8, 9 >( 600 );

    std::printf( "--- eager vs fused (elementwise linear combination) ---\n" );
    fusion< 30, 1 >( 300000 );
    fusion< 4, 6 >( 200000 );
    fusion< 6, 6 >( 80000 );
    fusion< 8, 6 >( 16000 );
    fusion< 6, 9 >( 16000 );
    fusion< 8, 9 >( 2000 );
    return 0;
}
