include(FetchContent)

find_package(benchmark 1.9 QUIET)

if(NOT benchmark_FOUND AND NOT TARGET benchmark::benchmark)
    message(STATUS "Google Benchmark not found, fetching via FetchContent...")

    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.4
    )

    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
endif()

if(NOT TARGET benchmark::benchmark)
    message(FATAL_ERROR
            "TAX_BUILD_BENCHMARK is ON but no benchmark::benchmark target is available.")
endif()

set(_dace_target "")
if(TAX_USE_DACE)
    if(TAX_DACE_TARGET AND TARGET ${TAX_DACE_TARGET})
        set(_dace_target ${TAX_DACE_TARGET})
    elseif(TARGET dace::dace_s)
        set(_dace_target dace::dace_s)
    elseif(TARGET dace_s)
        set(_dace_target dace_s)
    elseif(TARGET dace::dace)
        set(_dace_target dace::dace)
    elseif(TARGET dace)
        set(_dace_target dace)
    else()
        message(FATAL_ERROR
                "TAX_USE_DACE is ON but no DACE target is available for benchmarks.")
    endif()

    message(STATUS "DACE target for benchmarks: ${_dace_target}")
endif()

add_executable(bench_univariate univariate.cpp)
target_link_libraries(bench_univariate PRIVATE tax benchmark::benchmark)

if(TAX_USE_DACE)
    target_compile_definitions(bench_univariate PRIVATE TAX_BENCH_HAVE_DACE=1)
    target_link_libraries(bench_univariate PRIVATE ${_dace_target})
endif()
