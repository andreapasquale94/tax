from __future__ import annotations
import numpy as np
from .ir import single_op_graph
from .scheme import Isotropic, flat_index, unflat_index
from .types import Expansion
from .._codegen.emit_cpp import emit
from .._codegen import build, load

_KERNEL_CACHE: dict[str, tuple] = {}

def _embed(x: Expansion, target) -> np.ndarray:
    """Embed an expansion into `target`. Isotropic: graded-lex prefix. Named: axis-remap scatter."""
    if x.scheme == target:
        return x.coeffs
    from .scheme import Named
    if isinstance(target, Named):
        src = x.scheme  # a Named scheme over a subset of target's axes
        vmap = src.axis_var_map(target)
        out = np.zeros(target.n_coeff, dtype=np.float64)
        for k in range(src.n_coeff):
            c = x.coeffs[k]
            if c == 0.0:
                continue
            a_src = unflat_index(k, src.vars)
            a_dst = [0] * target.vars
            for j in range(src.vars):
                a_dst[vmap[j]] = a_src[j]
            out[flat_index(a_dst)] = c
        return out
    if x.scheme.vars != target.vars or x.scheme.order > target.order:
        raise ValueError(f"cannot embed {x.scheme} into {target}")
    out = np.zeros(target.n_coeff, dtype=np.float64)
    out[: x.scheme.n_coeff] = x.coeffs        # univariate / graded-lex prefix
    return out

def _as_expansion(value, ref_scheme: Isotropic) -> Expansion:
    if isinstance(value, Expansion):
        return value
    coeffs = np.zeros(ref_scheme.n_coeff, dtype=np.float64)
    coeffs[0] = float(value)
    return Expansion(coeffs, ref_scheme)

def run(graph, in_buffers, out_sizes):
    canon = graph.canonical()
    cxx = build.find_compiler()
    cid = build.compiler_id(cxx)
    opt_flags = ["-O3"]
    flags = build.flags_for_key(opt_flags)
    key = build.cache_key(canon, cid=cid, flags=flags)
    cached = _KERNEL_CACHE.get(key)
    if cached is None:
        so = build.compile_kernel(emit(graph), key, cxx=cxx,
                                  includes=build.include_dirs(), opt_flags=opt_flags)
        cached = load.load_kernel(so)
        _KERNEL_CACHE[key] = cached
    return load.call_kernel(cached, in_buffers, out_sizes)

def unary(opcode: str, x) -> Expansion:
    if not isinstance(x, Expansion):
        raise TypeError(f"{opcode}: expected Expansion, got {type(x)!r}")
    result_scheme = x.scheme
    graph = single_op_graph(opcode, [result_scheme], result_scheme)
    (out,) = run(graph, [x.coeffs], [result_scheme.n_coeff])
    return Expansion(out, result_scheme)

def binary(opcode: str, a, b) -> Expansion:
    ref = a.scheme if isinstance(a, Expansion) else b.scheme
    ea, eb = _as_expansion(a, ref), _as_expansion(b, ref)
    if opcode == "pow" and ea.value() <= 0.0:
        raise ValueError(
            f"pow/**: base constant term must be > 0 (got {ea.value()}); "
            "integer powers of a non-positive base are not yet supported"
        )
    result_scheme = ea.scheme.union(eb.scheme)
    graph = single_op_graph(opcode, [result_scheme, result_scheme], result_scheme)
    ba, bb = _embed(ea, result_scheme), _embed(eb, result_scheme)
    (out,) = run(graph, [ba, bb], [result_scheme.n_coeff])
    return Expansion(out, result_scheme)
