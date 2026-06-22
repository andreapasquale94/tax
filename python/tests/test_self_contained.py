import numpy as np
import tax
import tax._vendor as vendor
from tax._codegen import build
from tests._helpers import needs_toolchain


@needs_toolchain
def test_compiles_from_vendored_headers_without_tax_include(tmp_path, monkeypatch):
    vendor.sync_from_repo()                               # populate tax/_vendor/include/tax
    monkeypatch.delenv("TAX_INCLUDE", raising=False)      # simulate an installed wheel
    monkeypatch.setenv("TAX_CACHE_DIR", str(tmp_path))
    build.find_tax_include.cache_clear()
    from tax._frontend import eager
    eager._KERNEL_CACHE.clear()
    try:
        x = tax.variable(0.0, order=5)
        f = tax.sin(x) * tax.exp(x)                       # compiles using the vendored headers
        np.testing.assert_allclose(f.numpy(),
                                   [0, 1, 1, 1.0 / 3, 0, -1.0 / 30], atol=1e-12)
    finally:
        build.find_tax_include.cache_clear()              # don't leak the vendored path to other tests
