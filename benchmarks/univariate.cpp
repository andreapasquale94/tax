#include <benchmark/benchmark.h>
#include <tax/tax.hpp>

#ifdef TAX_BENCH_HAVE_DACE
#include <dace/dace.h>
#endif

namespace
{

template < int N, class Op >
void runTaxBenchmark( benchmark::State& state, double x0, Op&& op )
{
    const auto x = tax::DA< N >::variable( x0 );

    for ( auto _ : state )
    {
        tax::DA< N > y = op( x );
        benchmark::DoNotOptimize( y );
        benchmark::ClobberMemory();
    }
}

#ifdef TAX_BENCH_HAVE_DACE
template < int N, class Op >
void runDaceBenchmark( benchmark::State& state, Op&& op )
{
    DACE::DA::init( N, 1 );
    const DACE::DA xr( 1 );

    for ( auto _ : state )
    {
        DACE::DA yr = op( xr );
        benchmark::DoNotOptimize( yr );
        benchmark::ClobberMemory();
    }
}
#endif

template < int N >
void BM_Tax_Sin( benchmark::State& state )
{
    runTaxBenchmark< N >( state, 0.0, []( const auto& x ) { return tax::sin( x ); } );
}

template < int N >
void BM_Tax_Exp( benchmark::State& state )
{
    runTaxBenchmark< N >( state, 0.0, []( const auto& x ) { return tax::exp( x ); } );
}

template < int N >
void BM_Tax_Log( benchmark::State& state )
{
    runTaxBenchmark< N >( state, 1.0, []( const auto& x ) { return tax::log( x ); } );
}

template < int N >
void BM_Tax_Sqrt( benchmark::State& state )
{
    runTaxBenchmark< N >( state, 2.0, []( const auto& x ) { return tax::sqrt( x ); } );
}

template < int N >
void BM_Tax_IPow( benchmark::State& state )
{
    runTaxBenchmark< N >( state, 2.0, []( const auto& x ) { return tax::pow( x, 5 ); } );
}

template < int N >
void BM_Tax_Pow( benchmark::State& state )
{
    runTaxBenchmark< N >( state, 2.0, []( const auto& x ) { return tax::pow( x, 0.5 ); } );
}

#ifdef TAX_BENCH_HAVE_DACE
template < int N >
void BM_Dace_Sin( benchmark::State& state )
{
    runDaceBenchmark< N >( state, []( const DACE::DA& xr ) { return xr.sin(); } );
}

template < int N >
void BM_Dace_Exp( benchmark::State& state )
{
    runDaceBenchmark< N >( state, []( const DACE::DA& xr ) { return xr.exp(); } );
}

template < int N >
void BM_Dace_Log( benchmark::State& state )
{
    runDaceBenchmark< N >( state, []( const DACE::DA& xr ) { return ( 1.0 + xr ).log(); } );
}

template < int N >
void BM_Dace_Sqrt( benchmark::State& state )
{
    runDaceBenchmark< N >( state, []( const DACE::DA& xr ) { return ( 2.0 + xr ).sqrt(); } );
}

template < int N >
void BM_Dace_IPow( benchmark::State& state )
{
    runDaceBenchmark< N >( state, []( const DACE::DA& xr ) { return ( 2.0 + xr ).pow( 5 ); } );
}

template < int N >
void BM_Dace_Pow( benchmark::State& state )
{
    runDaceBenchmark< N >( state, []( const DACE::DA& xr ) { return ( 2.0 + xr ).pow( 0.5 ); } );
}
#endif

void registerBenchmarks()
{
    auto reg = []( const char* name, auto fn ) {
        benchmark::RegisterBenchmark( name, fn )->Unit( benchmark::kMicrosecond );
    };

    reg( "Tax/Sin/N10", &BM_Tax_Sin< 10 > );
    reg( "Tax/Sin/N20", &BM_Tax_Sin< 20 > );
    reg( "Tax/Sin/N40", &BM_Tax_Sin< 40 > );

    reg( "Tax/Exp/N10", &BM_Tax_Exp< 10 > );
    reg( "Tax/Exp/N20", &BM_Tax_Exp< 20 > );
    reg( "Tax/Exp/N40", &BM_Tax_Exp< 40 > );

    reg( "Tax/Log/N10", &BM_Tax_Log< 10 > );
    reg( "Tax/Log/N20", &BM_Tax_Log< 20 > );
    reg( "Tax/Log/N40", &BM_Tax_Log< 40 > );

    reg( "Tax/Sqrt/N10", &BM_Tax_Sqrt< 10 > );
    reg( "Tax/Sqrt/N20", &BM_Tax_Sqrt< 20 > );
    reg( "Tax/Sqrt/N40", &BM_Tax_Sqrt< 40 > );

    reg( "Tax/IPow/N10", &BM_Tax_IPow< 10 > );
    reg( "Tax/IPow/N20", &BM_Tax_IPow< 20 > );
    reg( "Tax/IPow/N40", &BM_Tax_IPow< 40 > );

    reg( "Tax/Pow/N10", &BM_Tax_Pow< 10 > );
    reg( "Tax/Pow/N20", &BM_Tax_Pow< 20 > );
    reg( "Tax/Pow/N40", &BM_Tax_Pow< 40 > );

#ifdef TAX_BENCH_HAVE_DACE
    reg( "Dace/Sin/N10", &BM_Dace_Sin< 10 > );
    reg( "Dace/Sin/N20", &BM_Dace_Sin< 20 > );
    reg( "Dace/Sin/N40", &BM_Dace_Sin< 40 > );

    reg( "Dace/Exp/N10", &BM_Dace_Exp< 10 > );
    reg( "Dace/Exp/N20", &BM_Dace_Exp< 20 > );
    reg( "Dace/Exp/N40", &BM_Dace_Exp< 40 > );

    reg( "Dace/Log/N10", &BM_Dace_Log< 10 > );
    reg( "Dace/Log/N20", &BM_Dace_Log< 20 > );
    reg( "Dace/Log/N40", &BM_Dace_Log< 40 > );

    reg( "Dace/Sqrt/N10", &BM_Dace_Sqrt< 10 > );
    reg( "Dace/Sqrt/N20", &BM_Dace_Sqrt< 20 > );
    reg( "Dace/Sqrt/N40", &BM_Dace_Sqrt< 40 > );

    reg( "Dace/IPow/N10", &BM_Dace_IPow< 10 > );
    reg( "Dace/IPow/N20", &BM_Dace_IPow< 20 > );
    reg( "Dace/IPow/N40", &BM_Dace_IPow< 40 > );

    reg( "Dace/Pow/N10", &BM_Dace_Pow< 10 > );
    reg( "Dace/Pow/N20", &BM_Dace_Pow< 20 > );
    reg( "Dace/Pow/N40", &BM_Dace_Pow< 40 > );
#endif
}

}  // namespace

int main( int argc, char** argv )
{
    benchmark::Initialize( &argc, argv );
    if ( benchmark::ReportUnrecognizedArguments( argc, argv ) ) return 1;

    registerBenchmarks();
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
