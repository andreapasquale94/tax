import numpy as np
import tax
from tax._frontend.array import Array
from tax._frontend.scheme import Named, Axis
from tests._helpers import needs_toolchain

MU = 398600.4418


@needs_toolchain
def test_named_two_body_rhs():
    x = tax.variables([1.0, 0.0, 0.0, 1.0], order=4, name="x")    # rx, ry, vx, vy
    mu = tax.variable(MU, order=4, name="mu")

    def rhs(x, mu):
        r3 = (x[0] * x[0] + x[1] * x[1]) ** 1.5
        return tax.concatenate([x[2], x[3], -mu * x[0] / r3, -mu * x[1] / r3])

    dx = rhs(x, mu)
    assert isinstance(dx, Array)
    assert dx.scheme == Named.of(4, [Axis("mu", 1), Axis("x", 4)])   # union, M=5

    # value of the RHS at the state (r = 1): [vx, vy, -mu*rx, -mu*ry] = [0, 1, -mu, 0]
    np.testing.assert_allclose(dx.value(), np.array([0.0, 1.0, -MU, 0.0]), rtol=1e-9, atol=1e-6)

    # ∂(rhs)/∂x : the state-transition block (4x4), x = [rx, ry, vx, vy] at (1,0,0,1)
    jac_x = dx.jacobian("x")
    expected_x = np.array([
        [0.0, 0.0, 1.0, 0.0],     # ∂vx
        [0.0, 0.0, 0.0, 1.0],     # ∂vy
        [2 * MU, 0.0, 0.0, 0.0],  # ∂(-mu rx / r^3) ; at r=1: -mu(1 - 3 rx^2) = 2mu
        [0.0, -MU, 0.0, 0.0],     # ∂(-mu ry / r^3) ; at r=1: -mu
    ])
    np.testing.assert_allclose(jac_x, expected_x, rtol=1e-9, atol=1e-6)

    # ∂(rhs)/∂mu : parameter sensitivity (4x1) = [0, 0, -rx/r^3, -ry/r^3] = [0, 0, -1, 0]
    jac_mu = dx.jacobian("mu")
    np.testing.assert_allclose(jac_mu, np.array([[0.0], [0.0], [-1.0], [0.0]]),
                               rtol=1e-9, atol=1e-9)
