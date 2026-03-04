#pragma once

#include <Eigen/Core>
#include <tax/tte.hpp>

namespace Eigen
{

template < typename T, int N, int M >
struct NumTraits< tax::TruncatedTaylorExpansionT< T, N, M > > : NumTraits< T >
{
    using Real = tax::TruncatedTaylorExpansionT< T, N, M >;
    using NonInteger = tax::TruncatedTaylorExpansionT< T, N, M >;
    using Literal = tax::TruncatedTaylorExpansionT< T, N, M >;
    using Nested = tax::TruncatedTaylorExpansionT< T, N, M >;

    enum
    {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = NumTraits< T >::ReadCost,
        AddCost = NumTraits< T >::AddCost,
        MulCost = NumTraits< T >::MulCost
    };
};

}  // namespace Eigen
