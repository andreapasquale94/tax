import pathlib

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"

def test_readme_exists_and_covers_the_api():
    assert README.is_file()
    text = README.read_text()
    for token in ["tax.variable", "tax.variables", "@tax.jit", "tax.concatenate",
                  "jacobian", "C++23", "Eigen", "TAX_CXX"]:
        assert token in text, f"README missing reference to {token!r}"
