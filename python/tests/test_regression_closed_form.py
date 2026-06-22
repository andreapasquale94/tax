import math
import numpy as np
import tax
from tests._helpers import needs_toolchain

@needs_toolchain
def test_exp_series():
    x = tax.variable(0.0, order=6)
    # exp(x) Maclaurin coeffs are 1/k!
    expected = np.array([1.0 / math.factorial(k) for k in range(7)])
    np.testing.assert_allclose(tax.exp(x).numpy(), expected, atol=1e-12)

@needs_toolchain
def test_log1p_series():
    x = tax.variable(0.0, order=5)
    # log(1+x) = x - x^2/2 + x^3/3 - x^4/4 + x^5/5
    expected = np.array([0.0, 1.0, -0.5, 1.0 / 3, -0.25, 0.2])
    np.testing.assert_allclose(tax.log(1.0 + x).numpy(), expected, atol=1e-12)

@needs_toolchain
def test_pythagorean_identity():
    x = tax.variable(0.7, order=6)
    s2c2 = tax.sin(x) * tax.sin(x) + tax.cos(x) * tax.cos(x)
    expected = np.zeros(7); expected[0] = 1.0          # identically 1
    np.testing.assert_allclose(s2c2.numpy(), expected, atol=1e-12)

@needs_toolchain
def test_exp_log_inverse():
    x = tax.variable(0.0, order=5)
    # exp(log(1+x)) == 1 + x exactly (as a truncated series)
    expected = np.zeros(6); expected[0] = 1.0; expected[1] = 1.0
    np.testing.assert_allclose(tax.exp(tax.log(1.0 + x)).numpy(), expected, atol=1e-12)

@needs_toolchain
def test_tanh_equals_sinh_over_cosh():
    x = tax.variable(0.4, order=6)
    np.testing.assert_allclose(tax.tanh(x).numpy(),
                               (tax.sinh(x) / tax.cosh(x)).numpy(), atol=1e-12)

@needs_toolchain
def test_multivariate_product_rule():
    # f = x0 * x1 at (a,b): value a*b; gradient [b, a]; mixed 2nd partial 1
    X = tax.variables([3.0, 5.0], order=2)
    f = X[0] * X[1]
    assert f.value() == 15.0
    np.testing.assert_allclose(f.gradient(), [5.0, 3.0], atol=1e-12)
    assert f.derivative(1, 1) == 1.0          # d²/dx0dx1 (x0 x1) = 1

@needs_toolchain
def test_named_chain_rule_sin_of_product():
    # g = sin(x*p) ; ∂g/∂x = p cos(x p), ∂g/∂p = x cos(x p) at (x,p)
    x = tax.variable(0.5, order=3, name="x")
    p = tax.variable(2.0, order=3, name="p")
    g = tax.sin(x * p)
    c = math.cos(0.5 * 2.0)
    np.testing.assert_allclose(g.gradient("x"), [2.0 * c], atol=1e-12)   # p cos(xp)
    np.testing.assert_allclose(g.gradient("p"), [0.5 * c], atol=1e-12)   # x cos(xp)
