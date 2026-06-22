"""Vendored third-party/sibling headers for self-contained JIT compilation.

At wheel-build time, sync_from_repo() copies the repo's include/tax tree here so
the installed package can compile kernels without TAX_INCLUDE pointing at a source
checkout. This directory's contents are git-ignored (synced at build, not committed).
"""
from __future__ import annotations
import pathlib
import shutil

def _repo_include_tax() -> pathlib.Path:
    # _vendor/__init__.py -> _vendor -> tax -> python -> <repo root>
    return pathlib.Path(__file__).resolve().parents[3] / "include" / "tax"

def sync_from_repo() -> pathlib.Path:
    """Copy <repo>/include/tax into tax/_vendor/include/tax; return tax/_vendor/include."""
    src = _repo_include_tax()
    if not src.is_dir():
        raise FileNotFoundError(f"repo headers not found at {src} (not building from the repo?)")
    dst_root = pathlib.Path(__file__).resolve().parent / "include"
    dst = dst_root / "tax"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst_root
