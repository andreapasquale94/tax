import pathlib
import pytest
import tax._vendor as vendor

def test_sync_from_repo_populates_vendor(tmp_path, monkeypatch):
    inc = vendor.sync_from_repo()                       # copy include/tax -> _vendor/include/tax
    inc = pathlib.Path(inc)
    assert (inc / "tax" / "tax.hpp").is_file()          # umbrella header vendored
    assert (inc / "tax" / "core" / "taylor_expansion.hpp").is_file()

def test_find_tax_include_uses_vendor_when_env_unset(monkeypatch):
    vendor.sync_from_repo()
    monkeypatch.delenv("TAX_INCLUDE", raising=False)
    from tax._codegen import build
    build.find_tax_include.cache_clear()               # lru_cache from M1
    resolved = pathlib.Path(build.find_tax_include())
    assert (resolved / "tax" / "tax.hpp").is_file()    # resolves the vendored copy
    build.find_tax_include.cache_clear()
