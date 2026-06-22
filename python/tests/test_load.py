import numpy as np
from tax._codegen import build, load
from tests._helpers import needs_toolchain   # noqa


@needs_toolchain
def test_load_and_call_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("TAX_CACHE_DIR", str(tmp_path))
    src = (
        'extern "C" int tax_kernel(const double* const* ins, double* const* outs)'
        ' noexcept { outs[0][0] = ins[0][0] * 2.0; outs[0][1] = ins[0][1] + 5.0;'
        ' return 0; }\n'
    )
    cxx = build.find_compiler()
    so = build.compile_kernel(src, "load_test", cxx=cxx,
                              includes=build.include_dirs(), opt_flags=["-O3"])
    fn = load.load_kernel(so)
    outs = load.call_kernel(fn, [np.array([3.0, 7.0])], [2])
    assert outs[0][0] == 6.0 and outs[0][1] == 12.0
