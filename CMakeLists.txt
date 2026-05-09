cmake_minimum_required(VERSION 3.20)
project(tax VERSION 0.2.0 LANGUAGES CXX)

option(TAX_BUILD_TEST "Build the GoogleTest suite" ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------- Eigen ----------
find_package(Eigen3 3.4 QUIET CONFIG)
if(NOT Eigen3_FOUND)
    # Fallback: probe well-known system include paths.
    find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Core
              PATHS /usr/include/eigen3 /usr/local/include/eigen3
              REQUIRED)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen INTERFACE ${EIGEN3_INCLUDE_DIR})
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
