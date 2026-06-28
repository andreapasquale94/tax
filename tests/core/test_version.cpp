#include <gtest/gtest.h>

#include <tax/tax.hpp>

TEST( Version, MacrosMatchProject )
{
    EXPECT_EQ( TAX_VERSION_MAJOR, 0 );
    EXPECT_EQ( TAX_VERSION_MINOR, 1 );
    EXPECT_EQ( TAX_VERSION_PATCH, 0 );
    EXPECT_STREQ( TAX_VERSION_STRING, "0.1.0" );
}
