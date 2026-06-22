import pytest
import numpy as np
from tax._frontend.scheme import flat_index, unflat_index, num_monomials
from tax._codegen import build, load
from tests._helpers import needs_toolchain   # sets TAX_INCLUDE/TAX_CACHE_DIR

def test_flat_unflat_roundtrip_and_bijection():
    for N in range(0, 6):
        for M in range(1, 5):
            n = num_monomials(N, M)
            seen = set()
            for k in range(n):
                a = unflat_index(k, M)
                assert len(a) == M
                assert sum(a) <= N
                assert flat_index(a) == k          # round-trip
                seen.add(a)
            assert len(seen) == n                  # bijection onto [0, n)

def test_linear_slots_are_i_plus_one():
    # coordinate variable i's linear monomial e_i lands at flat index i+1
    for M in (2, 3, 4):
        for i in range(M):
            e = tuple(1 if j == i else 0 for j in range(M))
            assert flat_index(e) == i + 1

# C++ cross-check: a probe kernel writes encode(unflatIndex<M>(k)) for every k;
# Python must compute the same encoding from its own unflat_index.
_PROBE = r'''
#include <tax/tax.hpp>
extern "C" int tax_kernel(const double* const*, double* const* outs) noexcept {{
    constexpr int N = {N}, M = {M};
    constexpr std::size_t n = tax::numMonomials(N, M);
    for (std::size_t k = 0; k < n; ++k) {{
        auto a = tax::unflatIndex<M>(k);
        double e = 0.0, base = 1.0;
        for (int i = 0; i < M; ++i) {{ e += double(a[std::size_t(i)]) * base; base *= double(N + 1); }}
        outs[0][k] = e;
    }}
    return 0;
}}
'''

def _encode(alpha, N):
    e, base = 0.0, 1.0
    for a in alpha:
        e += a * base
        base *= (N + 1)
    return e

@needs_toolchain
@pytest.mark.parametrize("N,M", [(5, 1), (3, 2), (4, 3)])
def test_layout_matches_cpp(N, M, tmp_path, monkeypatch):
    monkeypatch.setenv("TAX_CACHE_DIR", str(tmp_path))
    src = _PROBE.format(N=N, M=M)
    so = build.compile_kernel(src, f"layout_probe_{N}_{M}", cxx=build.find_compiler(),
                              includes=build.include_dirs(), opt_flags=["-O3"])
    n = num_monomials(N, M)
    (out,) = load.call_kernel(load.load_kernel(so), [], [n])
    expected = np.array([_encode(unflat_index(k, M), N) for k in range(n)])
    np.testing.assert_array_equal(out, expected)
