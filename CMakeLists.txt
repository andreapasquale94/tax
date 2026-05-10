cmake_minimum_required(VERSION 3.20)
project(tax VERSION 0.2.0 LANGUAGES CXX)

option(TAX_BUILD_TEST "Build the GoogleTest suite" ON)
option(TAX_BUILD_PYTHON "Build the nanobind Python bindings (target tax-python)" OFF)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------- Eigen ----------
# Try the CMake config first, then well-known system include paths,
# then fall back to FetchContent so the build still works inside
# manylinux-style images that may not ship Eigen.
find_package(Eigen3 3.4 QUIET CONFIG)
if(NOT Eigen3_FOUND)
    find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Core
              PATHS /usr/include/eigen3 /usr/local/include/eigen3)
    if(EIGEN3_INCLUDE_DIR)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        target_include_directories(Eigen3::Eigen INTERFACE ${EIGEN3_INCLUDE_DIR})
    else()
        message(STATUS "Eigen not found locally; fetching 3.4.0 via FetchContent.")
        include(FetchContent)
        set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
        set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG        3.4.0
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(eigen)
    endif()
endif()

# ---------- tax (header-only) ----------
add_library(tax INTERFACE)
add_library(tax::tax ALIAS tax)
target_include_directories(tax INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(tax INTERFACE Eigen3::Eigen)
target_compile_features(tax INTERFACE cxx_std_23)

# ---------- tests ----------
if(TAX_BUILD_TEST)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------- python bindings ----------
if(TAX_BUILD_PYTHON)
    if(NOT TAX_BUILD_TEST)
        enable_testing()
    endif()
    add_subdirectory(python)
endif()
