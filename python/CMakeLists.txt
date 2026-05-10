# Python bindings for tax via nanobind.
#
# Built only when TAX_BUILD_PYTHON is ON.  We use FindPython + nanobind's
# CMake helpers (`nanobind_add_module`) to produce a minimal extension
# module called `_tax`.  The Python-level package wrapper lives in
# `python/tax/__init__.py`.

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

# Locate nanobind via Python: `python -m nanobind --cmake_dir`.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE _nanobind_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _nanobind_rc)

if(NOT _nanobind_rc EQUAL 0)
    message(FATAL_ERROR
        "TAX_BUILD_PYTHON=ON but `python -m nanobind` failed.\n"
        "Install with: pip install nanobind")
endif()

list(APPEND CMAKE_PREFIX_PATH "${_nanobind_dir}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_tax src/tax_module.cpp)
target_link_libraries(_tax PRIVATE tax::tax)
target_compile_features(_tax PRIVATE cxx_std_23)
set_target_properties(_tax PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/tax")

# Wheel install path.  scikit-build-core uses CMake's install(TARGETS …)
# rules to land binaries in the wheel; the `tax` destination puts the
# compiled `_tax.so` next to `python/tax/__init__.py` (which the
# `[tool.scikit-build] wheel.packages` setting collects) so the final
# wheel exposes `tax/_tax.so` and `tax/__init__.py` together.
install(TARGETS _tax LIBRARY DESTINATION tax)

# Stage the Python package alongside the compiled module so users can
# `python -c "import tax"` from the build directory.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tax/__init__.py"
               "${CMAKE_BINARY_DIR}/python/tax/__init__.py" COPYONLY)

if(TAX_BUILD_TEST)
    find_program(PYTEST_EXECUTABLE NAMES pytest)
    if(PYTEST_EXECUTABLE)
        add_test(
            NAME python_bindings
            COMMAND ${PYTEST_EXECUTABLE} -q
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests")
        set_tests_properties(python_bindings PROPERTIES
            ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python")
    else()
        message(STATUS "pytest not found; skipping python_bindings test.")
    endif()
endif()
