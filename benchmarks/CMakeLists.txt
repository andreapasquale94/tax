# Optional micro-benchmark targets (off by default). Enable with
#   cmake -S . -B build -DTAX_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
# Self-contained: no external benchmark library required.
add_executable(bench_taylor bench_taylor.cpp)
target_link_libraries(bench_taylor PRIVATE tax)
target_compile_features(bench_taylor PRIVATE cxx_std_23)
