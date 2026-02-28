#include <tuple>

#include <benchmark/benchmark.h>
#include <tax/tax.hpp>

#include <array>
#include <cstdint>

namespace {

template <int Order>
using DA1 = tax::DA<Order>;

template <int Order>
static DA1<Order> make_x()
{
    return DA1<Order>::template variable<0>({1.25});
}

template <int Order>
static DA1<Order> make_y()
{
    const auto x = make_x<Order>();
    return DA1<Order>(2.0 + x * 0.5);
}

template <int Order, typename Fn>
static void run_unary_bench(benchmark::State& state, Fn&& fn)
{
    const auto x = make_x<Order>();
    for (auto _ : state) {
        auto out = fn(x);
        benchmark::DoNotOptimize(out);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(DA1<Order>::ncoef));
}

template <int Order, typename Fn>
static void run_binary_bench(benchmark::State& state, Fn&& fn)
{
    const auto x = make_x<Order>();
    const auto y = make_y<Order>();
    for (auto _ : state) {
        auto out = fn(x, y);
        benchmark::DoNotOptimize(out);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(DA1<Order>::ncoef));
}

template <int Order>
static void BM_Add(benchmark::State& state)
{
    run_binary_bench<Order>(state, [](const auto& a, const auto& b) { return DA1<Order>(a + b); });
}

template <int Order>
static void BM_Sub(benchmark::State& state)
{
    run_binary_bench<Order>(state, [](const auto& a, const auto& b) { return DA1<Order>(a - b); });
}

template <int Order>
static void BM_Mul(benchmark::State& state)
{
    run_binary_bench<Order>(state, [](const auto& a, const auto& b) { return DA1<Order>(a * b); });
}

template <int Order>
static void BM_Div(benchmark::State& state)
{
    run_binary_bench<Order>(state, [](const auto& a, const auto& b) { return DA1<Order>(a / b); });
}

template <int Order>
static void BM_UnaryNeg(benchmark::State& state)
{
    run_unary_bench<Order>(state, [](const auto& a) { return DA1<Order>(-a); });
}

template <int Order>
static void BM_AddScalarR(benchmark::State& state)
{
    run_unary_bench<Order>(state, [](const auto& a) { return DA1<Order>(a + 0.75); });
}

template <int Order>
static void BM_SubScalarR(benchmark::State& state)
{
    run_unary_bench<Order>(state, [](const auto& a) { return DA1<Order>(a - 0.75); });
}

template <int Order>
static void BM_MulScalarR(benchmark::State& state)
{
    run_unary_bench<Order>(state, [](const auto& a) { return DA1<Order>(a * 0.75); });
}

template <int Order>
static void BM_DivScalarR(benchmark::State& state)
{
    run_unary_bench<Order>(state, [](const auto& a) { return DA1<Order>(a / 0.75); });
}

template <int Order>
static void BM_AddScalarL(benchmark::State& state)
{
    run_unary_bench<Order>(state, [](const auto& a) { return DA1<Order>(0.75 + a); });
}

template <int Order>
static void BM_SubScalarL(benchmark::State& state)
{
    run_unary_bench<Order>(state, [](const auto& a) { return DA1<Order>(0.75 - a); });
}

template <int Order>
static void BM_MulScalarL(benchmark::State& state)
{
    run_unary_bench<Order>(state, [](const auto& a) { return DA1<Order>(0.75 * a); });
}

template <int Order>
static void BM_DivScalarL(benchmark::State& state)
{
    run_unary_bench<Order>(state, [](const auto& a) { return DA1<Order>(0.75 / a); });
}

template <int Order>
static void BM_Square(benchmark::State& state)
{
    run_unary_bench<Order>(state, [](const auto& a) { return DA1<Order>(square(a)); });
}

template <int Order>
static void BM_Cube(benchmark::State& state)
{
    run_unary_bench<Order>(state, [](const auto& a) { return DA1<Order>(cube(a)); });
}

template <int Order>
static void BM_Sqrt(benchmark::State& state)
{
    run_unary_bench<Order>(state, [](const auto& a) { return DA1<Order>(sqrt(a)); });
}

#define REGISTER_ORDER_BENCHMARKS(N)        \
    BENCHMARK_TEMPLATE(BM_Add, N);          \
    BENCHMARK_TEMPLATE(BM_Sub, N);          \
    BENCHMARK_TEMPLATE(BM_Mul, N);          \
    BENCHMARK_TEMPLATE(BM_Div, N);          \
    BENCHMARK_TEMPLATE(BM_UnaryNeg, N);     \
    BENCHMARK_TEMPLATE(BM_AddScalarR, N);   \
    BENCHMARK_TEMPLATE(BM_SubScalarR, N);   \
    BENCHMARK_TEMPLATE(BM_MulScalarR, N);   \
    BENCHMARK_TEMPLATE(BM_DivScalarR, N);   \
    BENCHMARK_TEMPLATE(BM_AddScalarL, N);   \
    BENCHMARK_TEMPLATE(BM_SubScalarL, N);   \
    BENCHMARK_TEMPLATE(BM_MulScalarL, N);   \
    BENCHMARK_TEMPLATE(BM_DivScalarL, N);   \
    BENCHMARK_TEMPLATE(BM_Square, N);       \
    BENCHMARK_TEMPLATE(BM_Cube, N);         \
    BENCHMARK_TEMPLATE(BM_Sqrt, N)

REGISTER_ORDER_BENCHMARKS(2);
REGISTER_ORDER_BENCHMARKS(4);
REGISTER_ORDER_BENCHMARKS(8);
REGISTER_ORDER_BENCHMARKS(16);
REGISTER_ORDER_BENCHMARKS(32);

#undef REGISTER_ORDER_BENCHMARKS

} // namespace

BENCHMARK_MAIN();
