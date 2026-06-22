import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples"))
import two_body                                   # examples/two_body.py
from tests._helpers import needs_toolchain

MU = 398600.4418

@needs_toolchain
def test_unnamed_example_rhs():
    dx = two_body.unnamed_rhs()
    np.testing.assert_allclose(dx.value(), [0.0, 1.0, -MU, 0.0], rtol=1e-9, atol=1e-6)

@needs_toolchain
def test_named_example_rhs():
    dx = two_body.named_rhs()
    np.testing.assert_allclose(dx.value(), [0.0, 1.0, -MU, 0.0], rtol=1e-9, atol=1e-6)
    # ∂(rhs)/∂mu = [0, 0, -rx/r^3, -ry/r^3] = [0,0,-1,0] at the unit-radius state
    np.testing.assert_allclose(dx.jacobian("mu"), [[0.0], [0.0], [-1.0], [0.0]],
                               rtol=1e-9, atol=1e-9)
