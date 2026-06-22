from tax._codegen import build
from tests._helpers import needs_toolchain  # importing also sets TAX_INCLUDE/TAX_CACHE_DIR


@needs_toolchain
def test_find_compiler_returns_path():
    cxx = build.find_compiler()
    assert isinstance(cxx, str) and cxx


@needs_toolchain
def test_compiler_id_is_stable_and_nonempty():
    cxx = build.find_compiler()
    cid = build.compiler_id(cxx)
    assert cxx in cid and len(cid) > len(cxx)


@needs_toolchain
def test_include_dirs_exist():
    import os

    for d in build.include_dirs():
        assert os.path.isdir(d), d


def test_cache_key_is_deterministic_and_sensitive():
    k1 = build.cache_key("g", cid="c", flags="-O3")
    k2 = build.cache_key("g", cid="c", flags="-O3")
    k3 = build.cache_key("h", cid="c", flags="-O3")
    assert k1 == k2 and k1 != k3
    assert len(k1) == 64   # sha256 hex digest


@needs_toolchain
def test_compile_kernel_builds_and_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("TAX_CACHE_DIR", str(tmp_path))
    src = (
        'extern "C" int tax_kernel(const double* const* ins, double* const* outs)'
        ' noexcept { outs[0][0] = ins[0][0] + 1.0; return 0; }\n'
    )
    cxx = build.find_compiler()
    so = build.compile_kernel(src, "abc123", cxx=cxx, includes=build.include_dirs(),
                              opt_flags=["-O3"])
    assert so.exists() and so.suffix == ".so"
    # Second call is a cache hit (same path, no recompile needed).
    so2 = build.compile_kernel(src, "abc123", cxx=cxx, includes=build.include_dirs(),
                               opt_flags=["-O3"])
    assert so2 == so
