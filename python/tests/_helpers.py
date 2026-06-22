import os
import pathlib
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
# Set at IMPORT time (before any skipif marker below is evaluated) so the
# codegen finds the repo headers and writes the JIT cache to a gitignored
# scratch dir. A session fixture would run too late — markers evaluate at
# collection/import.
os.environ.setdefault("TAX_INCLUDE", str(REPO_ROOT / "include"))
os.environ.setdefault("TAX_CACHE_DIR", str(REPO_ROOT / "python" / ".tax_cache"))


def _have_toolchain() -> bool:
    from tax._codegen import build

    try:
        build.find_compiler()
        build.find_eigen_include()
        build.find_tax_include()
        return True
    except Exception:
        return False


needs_toolchain = pytest.mark.skipif(
    not _have_toolchain(), reason="C++ compiler / Eigen / tax headers not available"
)
