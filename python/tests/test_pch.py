import numpy as np
import tax
from tax._codegen import build
from tests._helpers import needs_toolchain

@needs_toolchain
def test_pch_kernel_numerics_unchanged_and_built_once(tmp_path, monkeypatch):
    monkeypatch.setenv("TAX_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TAX_USE_PCH", raising=False)        # default = enabled
    from tax._frontend import eager
    eager._KERNEL_CACHE.clear()
    x = tax.variable(0.0, order=5)
    f = tax.sin(x)                                           # compiles a kernel (PCH used)
    np.testing.assert_allclose(f.numpy(),
                               [0, 1, 0, -1.0 / 6, 0, 1.0 / 120], atol=1e-12)
    g = tax.exp(x)                                           # second kernel reuses the same PCH
    np.testing.assert_allclose(g.numpy(),
                               [1, 1, 0.5, 1.0 / 6, 1.0 / 24, 1.0 / 120], atol=1e-12)
    pchs = list(tmp_path.glob("*.pch"))
    assert len(pchs) == 1                                    # PCH built exactly once, reused

@needs_toolchain
def test_pch_disabled_still_compiles(tmp_path, monkeypatch):
    monkeypatch.setenv("TAX_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TAX_USE_PCH", "0")                  # disabled
    from tax._frontend import eager
    eager._KERNEL_CACHE.clear()
    x = tax.variable(0.0, order=4)
    f = tax.exp(x)
    np.testing.assert_allclose(f.numpy(),
                               [1, 1, 0.5, 1.0 / 6, 1.0 / 24], atol=1e-12)
    assert list(tmp_path.glob("*.pch")) == []               # no PCH built when disabled

def test_pch_path_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv("TAX_USE_PCH", "0")
    assert build.pch_path("c++", ["/x"], ["-O3"]) is None    # no compiler invoked when disabled
