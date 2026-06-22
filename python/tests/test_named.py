import numpy as np
import pytest
import tax
from tax._frontend import eager
from tax._frontend.scheme import Named, Axis, flat_index, unflat_index
from tax._frontend.types import Expansion
from tax._codegen import build, load
from tests._helpers import needs_toolchain

def test_named_embed_scatters_axis_block():
    # x-only expansion (axis x, dim 2, order 2): put a marker on x's var 1 (flat index 2)
    src = Named.of(2, [Axis("x", 2)])
    coeffs = np.zeros(src.n_coeff)
    coeffs[flat_index((0, 1))] = 7.0          # exponent on x's 2nd coordinate
    e = Expansion(coeffs, src)
    target = src.union(Named.of(2, [Axis("mu", 1)]))   # {mu:1, x:2}: mu var0, x vars1-2
    out = eager._embed(e, target)
    # x's var1 -> union var2; the marker must land at flat_index of e_(union var2)
    dst = [0, 0, 0]
    dst[target.var_offset("x") + 1] = 1
    assert out[flat_index(tuple(dst))] == 7.0
    assert out.shape == (target.n_coeff,)

# --- C++ cross-check: a named product in Python must match the compiled C++ named product ---
_PROBE = r'''
#include <tax/tax.hpp>
#include <algorithm>
#include <array>
extern "C" int tax_kernel(const double* const*, double* const* outs) noexcept {
    using namespace tax;
    std::array<double, 4> x0{1.0, 0.0, 0.0, 1.0};
    auto xs = variables<"x", 4>(x0);            // NE<4, Axis<"x",4>>[4]
    auto mu = variable<"mu", 4>(398600.4418);   // NE<4, Axis<"mu",1>>
    auto f = mu * xs[0];                          // NE<4, Axis<"mu",1>, Axis<"x",4>>
    using F = decltype(f);
    std::copy_n(f.inner().coefficients().data(), F::nCoefficients, outs[0]);
    return 0;
}
'''

@needs_toolchain
def test_named_product_matches_cpp(tmp_path, monkeypatch):
    monkeypatch.setenv("TAX_CACHE_DIR", str(tmp_path))
    so = build.compile_kernel(_PROBE, "named_probe_mu_x", cxx=build.find_compiler(),
                              includes=build.include_dirs(), opt_flags=["-O3"])
    n = Named.of(4, [Axis("mu", 1), Axis("x", 4)]).n_coeff       # 126
    (expected,) = load.call_kernel(load.load_kernel(so), [], [n])
    mu = tax.variable(398600.4418, order=4, name="mu")
    xs = tax.variables([1.0, 0.0, 0.0, 1.0], order=4, name="x")
    f = mu * xs[0]
    assert f.scheme == Named.of(4, [Axis("mu", 1), Axis("x", 4)])
    np.testing.assert_allclose(f.numpy(), expected, rtol=1e-12, atol=1e-9)

@needs_toolchain
def test_named_unary_preserves_axes():
    x = tax.variable(0.0, order=5, name="t")
    f = tax.sin(x)
    assert f.scheme == Named.of(5, [Axis("t", 1)])
    np.testing.assert_allclose(f.numpy()[1], 1.0, atol=1e-12)

@needs_toolchain
def test_named_coeff_keyword_and_gradient_axis():
    # f = a*b for two 1-D axes a, b at (a0,b0) = (2, 3), order 2 -> axes {a, b}
    a = tax.variable(2.0, order=2, name="a")
    b = tax.variable(3.0, order=2, name="b")
    f = a * b
    # union {a, b}: a var0, b var1; f = (2+da)(3+db)
    assert f.coeff(a=0, b=0) == 6.0
    assert f.coeff(a=1, b=0) == 3.0       # ∂/∂a coeff = b0 = 3
    assert f.coeff(a=1, b=1) == 1.0       # mixed da*db coeff
    assert np.allclose(f.gradient(), [3.0, 2.0])      # [b0, a0]
    assert np.allclose(f.gradient("a"), [3.0])
    assert np.allclose(f.gradient("b"), [2.0])

def test_named_coeff_keyword_validation():
    X = tax.variables([1.0, 2.0], order=2, name="x")   # x is dim 2 (not 1-D)
    f = X[0]
    with pytest.raises(ValueError):
        f.coeff(x=1)                       # dim>1 axis via keyword -> error
    with pytest.raises(ValueError):
        f.coeff(0, 0, x=1)                 # positional + keyword mixed

@needs_toolchain
def test_pow_nonpositive_base_raises():
    import pytest, tax
    x = tax.variable(0.0, order=3)
    base = x - 2.0                    # constant term -2 < 0
    with pytest.raises(ValueError):
        base ** 2
    with pytest.raises(ValueError):
        tax.pow(base, 2)

def test_axis_name_must_be_ascii():
    import pytest
    from tax._frontend.scheme import Axis
    with pytest.raises(ValueError):
        Axis("μ", 1)                  # non-ASCII
