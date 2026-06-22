import numpy as np
import tax
from tests._helpers import needs_toolchain

@needs_toolchain
def test_jit_matches_eager_scalar():
    x = tax.variable(0.0, order=5)
    @tax.jit
    def f(x):
        return tax.sin(x) * tax.exp(x)
    got = f(x)
    want = tax.sin(x) * tax.exp(x)                 # eager
    np.testing.assert_allclose(got.numpy(), want.numpy(), atol=1e-12)

@needs_toolchain
def test_jit_matches_eager_vector_map():
    X = tax.variables([1.0, 2.0], order=4)
    @tax.jit
    def g(X):
        return tax.concatenate([X[0] * X[1], X[0] / X[1]])
    got = g(X)
    want = tax.concatenate([X[0] * X[1], X[0] / X[1]])
    assert isinstance(got, tax.Array) and len(got) == 2
    np.testing.assert_allclose(got.numpy(), want.numpy(), atol=1e-12)
    np.testing.assert_allclose(got.jacobian(), want.jacobian(), atol=1e-12)

@needs_toolchain
def test_jit_explicit_signature_compiles_and_matches_lazy():
    sig = [tax.ArrayType(order=4, size=2)]
    @tax.jit(sig)
    def g(X):
        return tax.concatenate([X[0] * X[1], X[0] + X[1]])
    X = tax.variables([1.0, 2.0], order=4)
    got = g(X)
    want = tax.concatenate([X[0] * X[1], X[0] + X[1]])
    np.testing.assert_allclose(got.numpy(), want.numpy(), atol=1e-12)

@needs_toolchain
def test_jit_dump_returns_source():
    @tax.jit(dump=True)
    def f(x):
        return tax.sin(x)
    x = tax.variable(0.0, order=4)
    f(x)
    assert "tax_kernel" in f.dump_source()         # the generated TU is retrievable

def test_jit_signature_arity_mismatch_raises():
    import pytest, tax
    @tax.jit([tax.ArrayType(order=2, size=2)])
    def g(X):
        return X[0] * X[1]
    X = tax.variables([1.0, 2.0], order=2)
    with pytest.raises(TypeError):
        g(X, X)                                            # 2 args vs 1-arg signature

@needs_toolchain
def test_jit_retraces_on_new_signature_but_reuses_match(monkeypatch):
    from tax._frontend import trace as tracemod
    calls = {"n": 0}
    real = tracemod.trace_function
    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(tracemod, "trace_function", counting)
    from tax._frontend import jit as jitmod
    monkeypatch.setattr(jitmod, "trace_function", counting)

    x4 = tax.variable(0.0, order=4)
    @tax.jit
    def f(x):
        return tax.exp(x)
    f(x4); f(x4)                                    # second call -> memo hit, no re-trace
    assert calls["n"] == 1

    x5 = tax.variable(0.0, order=5)                # different scheme -> new signature
    f(x5)
    assert calls["n"] == 2                          # re-traced for the new signature
