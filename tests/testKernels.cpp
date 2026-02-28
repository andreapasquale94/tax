#include "testUtils.hpp"

using namespace tax::detail;

// Helper: make a univariate coefficient array for 1 + x (order N).
template <int N>
static auto make1PlusX() {
    std::array<double, numMonomials(N, 1)> a{};
    a[0] = 1.0;
    if constexpr (N >= 1) a[1] = 1.0;
    return a;
}

// =============================================================================
// Element-wise arithmetic kernels
// =============================================================================

TEST(AddInPlace, BasicAdd) {
    std::array<double, 4> a{1, 2, 3, 4};
    std::array<double, 4> b{4, 3, 2, 1};
    addInPlace<double, 4>(a, b);
    EXPECT_EQ(a[0], 5.0); EXPECT_EQ(a[1], 5.0);
    EXPECT_EQ(a[2], 5.0); EXPECT_EQ(a[3], 5.0);
}

TEST(SubInPlace, BasicSub) {
    std::array<double, 3> a{5, 6, 7};
    std::array<double, 3> b{1, 2, 3};
    subInPlace<double, 3>(a, b);
    EXPECT_EQ(a[0], 4.0); EXPECT_EQ(a[1], 4.0); EXPECT_EQ(a[2], 4.0);
}

TEST(NegateInPlace, FlipsAllSigns) {
    std::array<double, 3> a{1, -2, 3};
    negateInPlace<double, 3>(a);
    EXPECT_EQ(a[0], -1.0); EXPECT_EQ(a[1], 2.0); EXPECT_EQ(a[2], -3.0);
}

TEST(ScaleInPlace, ScaleBy2) {
    std::array<double, 3> a{1, 2, 3};
    scaleInPlace<double, 3>(a, 2.0);
    EXPECT_EQ(a[0], 2.0); EXPECT_EQ(a[1], 4.0); EXPECT_EQ(a[2], 6.0);
}

// =============================================================================
// cauchyProduct — (1+x)*(1+x) = 1 + 2x + x^2
// =============================================================================

TEST(CauchyProduct, Univariate_LinTimesLin) {
    constexpr int N = 3, M = 1;
    using arr = std::array<double, numMonomials(N, M)>;
    arr f = make1PlusX<N>();   // 1+x
    arr g = f;
    arr out{};
    cauchyProduct<double, N, M>(out, f, g);
    // (1+x)^2 = 1 + 2x + x^2
    EXPECT_NEAR(out[0], 1.0, kTol);
    EXPECT_NEAR(out[1], 2.0, kTol);
    EXPECT_NEAR(out[2], 1.0, kTol);
    EXPECT_NEAR(out[3], 0.0, kTol);
}

TEST(CauchyProduct, Univariate_Identity) {
    // f * 1 = f
    constexpr int N = 4, M = 1;
    using arr = std::array<double, numMonomials(N, M)>;
    arr f = make1PlusX<N>();
    arr one{}; one[0] = 1.0;
    arr out{};
    cauchyProduct<double, N, M>(out, f, one);
    for (std::size_t i = 0; i < numMonomials(N, M); ++i)
        EXPECT_NEAR(out[i], f[i], kTol) << "i=" << i;
}

TEST(CauchyProduct, Bivariate_LinTimesLin) {
    // (1+x+y)*(1+x+y) in N=2, M=2
    // = 1 + 2x + 2y + x^2 + 2xy + y^2
    constexpr int N = 2, M = 2;
    using arr = std::array<double, numMonomials(N, M)>;
    // 1+x+y: c[{0,0}]=1, c[{1,0}]=1, c[{0,1}]=1
    arr f{};
    f[flatIndex<2>({0,0})] = 1.0;
    f[flatIndex<2>({1,0})] = 1.0;
    f[flatIndex<2>({0,1})] = 1.0;
    arr out{};
    cauchyProduct<double, N, M>(out, f, f);
    EXPECT_NEAR(out[flatIndex<2>({0,0})], 1.0, kTol);
    EXPECT_NEAR(out[flatIndex<2>({1,0})], 2.0, kTol);
    EXPECT_NEAR(out[flatIndex<2>({0,1})], 2.0, kTol);
    EXPECT_NEAR(out[flatIndex<2>({2,0})], 1.0, kTol);
    EXPECT_NEAR(out[flatIndex<2>({1,1})], 2.0, kTol);
    EXPECT_NEAR(out[flatIndex<2>({0,2})], 1.0, kTol);
}

// =============================================================================
// cauchyAccumulate — accumulates rather than overwrites
// =============================================================================

TEST(CauchyAccumulate, Univariate_AccumulatesCorrectly) {
    constexpr int N = 3, M = 1;
    using arr = std::array<double, numMonomials(N, M)>;
    arr f = make1PlusX<N>();
    arr out{};  out[0] = 1.0;   // pre-existing value in out
    cauchyAccumulate<double, N, M>(out, f, f);
    // out was [1,0,0,0]; accumulate += (1+x)^2 = [1,2,1,0]
    // result should be [2,2,1,0]
    EXPECT_NEAR(out[0], 2.0, kTol);
    EXPECT_NEAR(out[1], 2.0, kTol);
    EXPECT_NEAR(out[2], 1.0, kTol);
    EXPECT_NEAR(out[3], 0.0, kTol);
}

TEST(CauchyAccumulate, VsProductPlusExisting) {
    // cauchyAccumulate(out, f, g) == out + cauchyProduct(f, g)
    constexpr int N = 4, M = 1;
    using arr = std::array<double, numMonomials(N, M)>;
    arr f = make1PlusX<N>();
    arr g{};  g[0] = 2.0; g[1] = -1.0;   // 2 - x

    arr prod{};
    cauchyProduct<double, N, M>(prod, f, g);

    arr base{};  base[0] = 5.0; base[2] = 3.0;
    arr ref = base;
    addInPlace<double, numMonomials(N, M)>(ref, prod);

    cauchyAccumulate<double, N, M>(base, f, g);
    for (std::size_t i = 0; i < numMonomials(N, M); ++i)
        EXPECT_NEAR(base[i], ref[i], kTol) << "i=" << i;
}

// =============================================================================
// seriesReciprocal — 1/(1+x) = 1 - x + x^2 - x^3 + ...
// =============================================================================

TEST(SeriesReciprocal, Univariate_GeometricSeries) {
    constexpr int N = 4, M = 1;
    using arr = std::array<double, numMonomials(N, M)>;
    arr a = make1PlusX<N>();   // 1+x
    arr out{};
    seriesReciprocal<double, N, M>(out, a);
    // 1/(1+x) = sum_{k=0}^{N} (-1)^k x^k
    for (int k = 0; k <= N; ++k)
        EXPECT_NEAR(out[k], std::pow(-1.0, k), kTol) << "k=" << k;
}

TEST(SeriesReciprocal, ConstantInput) {
    // 1/4 as a series
    constexpr int N = 3, M = 1;
    using arr = std::array<double, numMonomials(N, M)>;
    arr a{}; a[0] = 4.0;
    arr out{};
    seriesReciprocal<double, N, M>(out, a);
    EXPECT_NEAR(out[0], 0.25, kTol);
    EXPECT_NEAR(out[1], 0.0,  kTol);
    EXPECT_NEAR(out[2], 0.0,  kTol);
}

TEST(SeriesReciprocal, ProductIsIdentity) {
    // a * (1/a) = 1  (truncated at order N)
    constexpr int N = 4, M = 1;
    using arr = std::array<double, numMonomials(N, M)>;
    arr a = make1PlusX<N>();
    arr rec{};
    seriesReciprocal<double, N, M>(rec, a);
    arr prod{};
    cauchyProduct<double, N, M>(prod, a, rec);
    EXPECT_NEAR(prod[0], 1.0, kTol);
    for (int k = 1; k <= N; ++k)
        EXPECT_NEAR(prod[k], 0.0, kTol) << "k=" << k;
}

// =============================================================================
// seriesSquare — out = a²
// =============================================================================

TEST(SeriesSquare, SameAsCauchyProductWithSelf) {
    constexpr int N = 4, M = 1;
    using arr = std::array<double, numMonomials(N, M)>;
    arr a{}; a[0]=1; a[1]=2; a[2]=-1;
    arr sq{}, ref{};
    seriesSquare<double, N, M>(sq, a);
    cauchyProduct<double, N, M>(ref, a, a);
    for (std::size_t i = 0; i < numMonomials(N, M); ++i)
        EXPECT_NEAR(sq[i], ref[i], kTol) << "i=" << i;
}

TEST(SeriesSquare, KnownResult) {
    // (1+x)^2 = 1 + 2x + x^2
    constexpr int N = 3, M = 1;
    using arr = std::array<double, numMonomials(N, M)>;
    arr a = make1PlusX<N>();
    arr out{};
    seriesSquare<double, N, M>(out, a);
    EXPECT_NEAR(out[0], 1.0, kTol);
    EXPECT_NEAR(out[1], 2.0, kTol);
    EXPECT_NEAR(out[2], 1.0, kTol);
    EXPECT_NEAR(out[3], 0.0, kTol);
}

// =============================================================================
// seriesCube — out = a³
// =============================================================================

TEST(SeriesCube, KnownResult) {
    // (1+x)^3 = 1 + 3x + 3x^2 + x^3
    constexpr int N = 4, M = 1;
    using arr = std::array<double, numMonomials(N, M)>;
    arr a = make1PlusX<N>();
    arr out{};
    seriesCube<double, N, M>(out, a);
    EXPECT_NEAR(out[0], 1.0, kTol);
    EXPECT_NEAR(out[1], 3.0, kTol);
    EXPECT_NEAR(out[2], 3.0, kTol);
    EXPECT_NEAR(out[3], 1.0, kTol);
    EXPECT_NEAR(out[4], 0.0, kTol);
}

TEST(SeriesCube, VsThreeWayProduct) {
    // a^3 via seriesCube should match a * a * a via two cauchyProducts
    constexpr int N = 4, M = 1;
    using arr = std::array<double, numMonomials(N, M)>;
    arr a{}; a[0]=1; a[1]=3; a[2]=-1;
    arr cube{};
    seriesCube<double, N, M>(cube, a);
    arr tmp{}, ref{};
    cauchyProduct<double, N, M>(tmp, a, a);
    cauchyProduct<double, N, M>(ref, tmp, a);
    for (std::size_t i = 0; i < numMonomials(N, M); ++i)
        EXPECT_NEAR(cube[i], ref[i], kTol) << "i=" << i;
}

// =============================================================================
// seriesSqrt — solve g*g = a
// =============================================================================

TEST(SeriesSqrt, SqrtOf1PlusX) {
    // sqrt(1+x) at order 4:
    //   c[0]=1, c[1]=1/2, c[2]=-1/8, c[3]=1/16, c[4]=-5/128
    constexpr int N = 4, M = 1;
    using arr = std::array<double, numMonomials(N, M)>;
    arr a = make1PlusX<N>();
    arr g{};
    seriesSqrt<double, N, M>(g, a);
    EXPECT_NEAR(g[0],  1.0,           kTol);
    EXPECT_NEAR(g[1],  0.5,           kTol);
    EXPECT_NEAR(g[2], -0.125,         kTol);
    EXPECT_NEAR(g[3],  0.0625,        kTol);
    EXPECT_NEAR(g[4], -5.0 / 128.0,   kTol);
}

TEST(SeriesSqrt, SqrtSquaredIsOriginal) {
    // g*g should recover a (up to truncation)
    constexpr int N = 4, M = 1;
    using arr = std::array<double, numMonomials(N, M)>;
    arr a = make1PlusX<N>();
    arr g{};
    seriesSqrt<double, N, M>(g, a);
    arr check{};
    seriesSquare<double, N, M>(check, g);
    for (std::size_t i = 0; i < numMonomials(N, M); ++i)
        EXPECT_NEAR(check[i], a[i], kTol) << "i=" << i;
}

TEST(SeriesSqrt, ConstantInput) {
    // sqrt(4) = 2, all higher coefficients zero
    constexpr int N = 3, M = 1;
    using arr = std::array<double, numMonomials(N, M)>;
    arr a{}; a[0] = 4.0;
    arr g{};
    seriesSqrt<double, N, M>(g, a);
    EXPECT_NEAR(g[0], 2.0, kTol);
    EXPECT_NEAR(g[1], 0.0, kTol);
    EXPECT_NEAR(g[2], 0.0, kTol);
}
