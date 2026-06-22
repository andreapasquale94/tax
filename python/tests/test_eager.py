import numpy as np
import tax
from tests._helpers import needs_toolchain   # noqa

@needs_toolchain
def test_eager_sin_univariate():
    x = tax.variable(0.0, order=5)
    f = tax.sin(x)
    expected = np.array([0, 1, 0, -1/6, 0, 1/120], dtype=float)
    np.testing.assert_allclose(f.numpy(), expected, atol=1e-12)

@needs_toolchain
def test_eager_exp_univariate():
    x = tax.variable(0.0, order=5)
    f = tax.exp(x)
    expected = np.array([1, 1, 1/2, 1/6, 1/24, 1/120], dtype=float)
    np.testing.assert_allclose(f.numpy(), expected, atol=1e-12)

@needs_toolchain
def test_eager_scalar_broadcast_mul():
    x = tax.variable(0.0, order=3)
    f = 2.0 * x            # exercises __rmul__ + _as_expansion
    np.testing.assert_allclose(f.numpy(), np.array([0, 2, 0, 0], dtype=float), atol=1e-12)
