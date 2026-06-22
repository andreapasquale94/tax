import tax
from tax._frontend import eager
from tax._codegen import build
from tests._helpers import needs_toolchain

@needs_toolchain
def test_jit_fuses_multi_op_into_single_kernel(tmp_path, monkeypatch):
    monkeypatch.setenv("TAX_CACHE_DIR", str(tmp_path))   # fresh on-disk cache
    count = {"n": 0}
    real = build.compile_kernel
    def counting(*a, **k):
        count["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(build, "compile_kernel", counting)

    x = tax.variable(0.0, order=4)

    # Eager: sin, exp, mul -> three distinct (op, scheme) kernels compiled.
    eager._KERNEL_CACHE.clear()
    start = count["n"]
    _ = tax.sin(x) * tax.exp(x)
    eager_compiles = count["n"] - start
    assert eager_compiles >= 2          # op-by-op compiles multiple kernels

    # JIT: the whole function fuses into ONE kernel.
    eager._KERNEL_CACHE.clear()
    @tax.jit
    def f(x):
        return tax.sin(x) * tax.exp(x)
    start = count["n"]
    _ = f(x)
    jit_compiles = count["n"] - start
    assert jit_compiles == 1            # fusion -> a single compiled kernel
    assert jit_compiles < eager_compiles
