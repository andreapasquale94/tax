import numpy as np
import tax
from tests._helpers import needs_toolchain

MU = 398600.4418

def _two_body(x, mu):
    r3 = (x[0] * x[0] + x[1] * x[1]) ** 1.5
    return tax.concatenate([x[2], x[3], -mu * x[0] / r3, -mu * x[1] / r3])

@needs_toolchain
def test_jit_unnamed_two_body_matches_eager():
    x = tax.variables([1.0, 0.0, 0.0, 1.0], order=4)     # isotropic, mu baked as float

    @tax.jit
    def rhs(t, x, mu):
        return _two_body(x, mu)

    dx = rhs(0.0, x, MU)
    want = _two_body(x, MU)                               # eager
    np.testing.assert_allclose(dx.value(), np.array([0.0, 1.0, -MU, 0.0]), rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(dx.numpy(), want.numpy(), rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(dx.jacobian(), want.jacobian(), rtol=1e-9, atol=1e-6)

@needs_toolchain
def test_jit_named_two_body_bare_and_signature():
    from tax._frontend.scheme import Named, Axis
    x = tax.variables([1.0, 0.0, 0.0, 1.0], order=4, name="x")
    mu = tax.variable(MU, order=4, name="mu")

    @tax.jit
    def rhs_bare(t, x, mu):
        return _two_body(x, mu)

    dx = rhs_bare(0.0, x, mu)
    want = _two_body(x, mu)                               # eager
    assert dx.scheme == Named.of(4, [Axis("mu", 1), Axis("x", 4)])
    np.testing.assert_allclose(dx.numpy(), want.numpy(), rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(dx.jacobian("x"), want.jacobian("x"), rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(dx.jacobian("mu"), want.jacobian("mu"), rtol=1e-9, atol=1e-9)

    # same RHS, pinned with an explicit numba-style signature (compiles at decoration)
    @tax.jit([tax.f64, tax.ArrayType(order=4, size=4, name="x"),
              tax.ExpansionType(order=4, name="mu")])
    def rhs_sig(t, x, mu):
        return _two_body(x, mu)

    dx2 = rhs_sig(0.0, x, mu)
    np.testing.assert_allclose(dx2.numpy(), want.numpy(), rtol=1e-9, atol=1e-6)
