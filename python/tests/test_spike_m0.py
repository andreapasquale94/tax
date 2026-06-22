import numpy as np
from tax._codegen import build, load
from tests._helpers import needs_toolchain   # noqa

SRC = r'''
#include <tax/tax.hpp>
#include <algorithm>
using namespace tax;
extern "C" int tax_kernel(const double* const* ins, double* const* outs) noexcept {
    using E = TaylorExpansion<double, IsotropicScheme<5, 1>>;
    E::Data d; std::copy_n(ins[0], 6, d.data());
    E x{d};
    E r = sin(x) * exp(x);
    std::copy_n(r.coefficients().data(), E::nCoefficients, outs[0]);
    return 0;
}
'''

@needs_toolchain
def test_m0_sin_times_exp(tmp_path, monkeypatch):
    monkeypatch.setenv("TAX_CACHE_DIR", str(tmp_path))
    cxx = build.find_compiler()
    so = build.compile_kernel(SRC, "m0_spike", cxx=cxx,
                              includes=build.include_dirs(), opt_flags=["-O3"])
    fn = load.load_kernel(so)
    # x = 0 + 1*dx, order 5  -> seed [0, 1, 0, 0, 0, 0]
    (out,) = load.call_kernel(fn, [np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])], [6])
    expected = np.array([0.0, 1.0, 1.0, 1.0/3.0, 0.0, -1.0/30.0])
    np.testing.assert_allclose(out, expected, atol=1e-12)
