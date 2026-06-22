import numpy as np
import tax
from tests._helpers import needs_toolchain

@needs_toolchain
def test_pow_int_negative_base_eager():
    x = tax.variable(0.0, order=3)
    f = (x - 2.0) ** 2                              # (-2 + dx)^2 = 4 - 4 dx + dx^2
    np.testing.assert_allclose(f.numpy(), np.array([4.0, -4.0, 1.0, 0.0]), atol=1e-12)

@needs_toolchain
def test_pow_int_matches_under_jit():
    x = tax.variable(0.0, order=3)
    @tax.jit
    def f(x):
        return (x - 2.0) ** 2
    np.testing.assert_allclose(f(x).numpy(), np.array([4.0, -4.0, 1.0, 0.0]), atol=1e-12)

@needs_toolchain
def test_pow_real_nonpositive_still_guarded():
    import pytest
    x = tax.variable(0.0, order=3)
    with pytest.raises(ValueError):
        (x - 2.0) ** 1.5                            # real exponent, base < 0 -> still raises
