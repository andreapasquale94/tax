# ---------------------------------------------------------------------------
# Python bindings for tax (pybind11)
#
# Builds a Python extension module `_tax` that exposes a runtime-dimensioned
# TruncatedExpansion class. Internally, operations dispatch to a precompiled
# grid of (N, M) instantiations bounded by TAX_PY_MAX_N / TAX_PY_MAX_M and
# filtered by TAX_PY_MAX_COEFFS.
# ---------------------------------------------------------------------------

include(FetchContent)

# ---- pybind11 -------------------------------------------------------------
find_package(pybind11 2.11 QUIET CONFIG)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found; fetching via FetchContent")
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# ---- Module sources -------------------------------------------------------
# Operations are split into several translation units so that template
# instantiations for the full (N, M) grid compile in parallel.
set(_tax_py_sources
    src/bindings.cpp
    src/ops_arith.cpp
    src/ops_trig.cpp
    src/ops_hyper.cpp
    src/ops_misc.cpp
    src/ops_pow.cpp
    src/ops_scalar.cpp
    src/ops_access.cpp
)

pybind11_add_module(_tax MODULE ${_tax_py_sources})

target_link_libraries(_tax PRIVATE tax::tax)
target_include_directories(_tax PRIVATE src)
target_compile_features(_tax PRIVATE cxx_std_23)

target_compile_definitions(_tax PRIVATE
    TAX_PY_MAX_N=${TAX_PY_MAX_N}
    TAX_PY_MAX_M=${TAX_PY_MAX_M}
    TAX_PY_MAX_COEFFS=${TAX_PY_MAX_COEFFS}
    TAX_PY_VERSION="${PROJECT_VERSION}"
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(_tax PRIVATE -Wno-psabi)
endif()

# ---- Install / stage outputs ---------------------------------------------
if(SKBUILD)
    install(TARGETS _tax DESTINATION tax)
else()
    set(_tax_py_outdir "${CMAKE_BINARY_DIR}/python/tax")
    set_target_properties(_tax PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${_tax_py_outdir}"
    )
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/tax/ DESTINATION "${_tax_py_outdir}")
endif()
