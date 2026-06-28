#include <tax/tax.hpp>

#include <cstdio>

int main()
{
    auto x = tax::TE< 5 >::variable( 1.0 );
    auto f = sin( x ) * exp( x );
    std::printf( "tax %s: f.value() = %g\n", TAX_VERSION_STRING, f.value() );
    return 0;
}
