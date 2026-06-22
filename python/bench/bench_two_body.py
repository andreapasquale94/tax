"""Manual benchmark: eager vs @tax.jit vs hand-written C++ for the planar two-body RHS.

Run from python/:  .venv/bin/python bench/bench_two_body.py
(Requires a C++23 compiler + Eigen, like the test suite. Not a pytest test —
the deterministic fusion guard lives in tests/test_perf_fusion.py.)
"""
import os, sys, time, pathlib
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
_PYTHON_DIR = str(REPO / "python")
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)
os.environ.setdefault("TAX_INCLUDE", str(REPO / "include"))
os.environ.setdefault("TAX_CACHE_DIR", str(REPO / "python" / ".tax_cache"))

import tax
from tax._codegen import build, load

MU = 398600.4418
N_CALLS = 2000

def two_body(x, mu):
    r3 = (x[0] * x[0] + x[1] * x[1]) ** 1.5
    return tax.concatenate([x[2], x[3], -mu * x[0] / r3, -mu * x[1] / r3])

# Hand-written C++ baseline: the same RHS as one fused extern "C" kernel.
_BASELINE = r'''
#include <tax/tax.hpp>
#include <algorithm>
using namespace tax;
extern "C" int tax_kernel(const double* const* ins, double* const* outs) noexcept {
    using E = TaylorExpansion<double, IsotropicScheme<4, 4>>;
    E::Data d; std::copy_n(ins[0], E::nCoefficients, d.data());
    E rx{d}; std::copy_n(ins[1], E::nCoefficients, d.data());
    E ry{d}; std::copy_n(ins[2], E::nCoefficients, d.data());
    E vx{d}; std::copy_n(ins[3], E::nCoefficients, d.data());
    E vy{d};
    E r3 = pow(rx * rx + ry * ry, 1.5);
    E a0 = vx, a1 = vy, a2 = (-MU_VAL) * rx / r3, a3 = (-MU_VAL) * ry / r3;
    std::copy_n(a0.coefficients().data(), E::nCoefficients, outs[0]);
    std::copy_n(a1.coefficients().data(), E::nCoefficients, outs[1]);
    std::copy_n(a2.coefficients().data(), E::nCoefficients, outs[2]);
    std::copy_n(a3.coefficients().data(), E::nCoefficients, outs[3]);
    return 0;
}
'''.replace("MU_VAL", repr(MU))

def _time(label, fn, n=N_CALLS):
    fn()                                  # warm up (compile / cache)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = (time.perf_counter() - t0) / n * 1e6   # microseconds per call
    print(f"  {label:<28} {dt:8.2f} us/call")
    return dt

def main():
    x = tax.variables([1.0, 0.0, 0.0, 1.0], order=4)
    jitted = tax.jit(lambda t, x, mu: two_body(x, mu))

    # hand C++ baseline
    so = build.compile_kernel(_BASELINE, "bench_two_body_baseline", cxx=build.find_compiler(),
                              includes=build.include_dirs(), opt_flags=["-O3"])
    fn = load.load_kernel(so)
    rows = [x[i].coeffs for i in range(4)]
    n = x.scheme.n_coeff
    def call_cpp():
        load.call_kernel(fn, rows, [n, n, n, n])

    print(f"Planar two-body RHS, order 4, {N_CALLS} warm calls each:")
    _time("eager (per-op FFI)", lambda: two_body(x, MU))
    _time("jit (fused, 1 FFI)", lambda: jitted(0.0, x, MU))
    _time("hand-written C++ baseline", call_cpp)
    print("Expectation: jit ≈ C++ baseline, both well below eager (which pays one FFI per op).")

if __name__ == "__main__":
    main()
