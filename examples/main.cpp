#include <printf.h>

#include <iostream>
#include <tax/tax.hpp>

int main()
{
    // auto x = tax::TE< 10 >::variable( 0.0 );

    // tax::TE< 10 > y = tax::pow( x, 4 );

    // std::cout << tax::eval( y, 1.0 ) << "\n";

    auto [x, y] = tax::TEn< 4, 2 >::variables( 1.0, 1.0 );

    tax::TEn< 4, 2 > z = x * x + y * y;

    std::cout << z << "\n\n";

    std::cout << tax::value( z ) << "\n\n";
    std::cout << tax::derivative< 1 >( z ).transpose() << "\n\n";
    std::cout << tax::derivative< 2 >( z ) << "\n\n";
    std::cout << tax::derivative< 3 >( z ) << "\n\n";

    std::cout << tax::eval( z, tax::TEn< 4, 2 >::Input{ 1e-12, 1e-12 } ) << "\n\n";

    // test
    return 0;
}