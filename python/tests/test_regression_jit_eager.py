import numpy as np
import pytest
import tax
from tests._helpers import needs_toolchain

# Each case: (builder, make_inputs). builder(*inputs) composes a computation that
# works on either eager handles or jit tracers; we run it eagerly and under @tax.jit
# and assert the fused result equals the op-by-op result.
SCALAR_CASES = {
    "sin_exp":      lambda x: tax.sin(x) * tax.exp(x),
    "deep_mix":     lambda x: tax.sin(tax.exp(x)) * tax.log(2.0 + x) - tax.atan(x),
    "pow_int":      lambda x: (x - 2.0) ** 3 + (x + 1.0) ** 2,
    "pow_real":     lambda x: (x * x + 1.0) ** 1.5,
    "ratio":        lambda x: tax.tanh(x) / (1.0 + tax.cosh(x)),
    "transcend":    lambda x: tax.erf(x) + tax.atanh(x / 4.0) - tax.cbrt(2.0 + x),
}

@needs_toolchain
@pytest.mark.parametrize("name", list(SCALAR_CASES))
def test_jit_equals_eager_scalar(name):
    f = SCALAR_CASES[name]
    x = tax.variable(0.3, order=6)
    eager = f(x)
    jitted = tax.jit(f)(x)
    np.testing.assert_allclose(jitted.numpy(), eager.numpy(), atol=1e-12, rtol=0)

@needs_toolchain
def test_jit_equals_eager_vector_named():
    def f(x, p):
        r = tax.norm(x)
        return tax.concatenate([x[0] * p, x[1] / (1.0 + r), tax.sin(x[0] + x[1])])
    x = tax.variables([1.0, 2.0], order=4, name="x")
    p = tax.variable(0.5, order=4, name="p")
    eager = f(x, p)
    jitted = tax.jit(f)(x, p)
    assert eager.scheme == jitted.scheme
    np.testing.assert_allclose(jitted.numpy(), eager.numpy(), atol=1e-12, rtol=0)
    np.testing.assert_allclose(jitted.jacobian("x"), eager.jacobian("x"), atol=1e-12, rtol=0)
