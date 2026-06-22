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
