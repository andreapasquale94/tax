cmake_minimum_required(VERSION 4.2)

project(tax VERSION 0.1.0 LANGUAGES CXX)

# -------------------------------
# Options
# -------------------------------
option(TAX_BUILD_TEST "Build and enable unittests" ON)
option(TAX_BUILD_BENCH "Build benchmarks" ON)
option(TAX_ENABLE_EIGEN "Enable Eigen tensor adapters" OFF)

# -------------------------------
# Compiler settings
# -------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wno-psabi)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-fconcepts-diagnostics-depth=2)
    endif()
endif()

# -------------------------------
# Target
# -------------------------------
add_library(tax INTERFACE)
add_library(tax::tax ALIAS tax)

target_compile_features(tax INTERFACE cxx_std_23)
target_include_directories(tax
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(TAX_ENABLE_EIGEN)
    include(FetchContent)
    find_package(Eigen3 5 QUIET)
    if(NOT Eigen3_FOUND)
        message(STATUS "Eigen3 not found, fetching via FetchContent...")
        FetchContent_Declare(
            eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 5.0.0
        )
        FetchContent_MakeAvailable(eigen)
    endif()
    if(NOT TARGET Eigen3::Eigen)
        message(FATAL_ERROR "Eigen3::Eigen target is unavailable after dependency resolution")
    endif()
    target_link_libraries(tax INTERFACE Eigen3::Eigen)
endif()

# -------------------------------
# Benchmarks
# -------------------------------
if(TAX_BUILD_BENCH)
    add_subdirectory(bench)
endif()

# -------------------------------
# Tests
# -------------------------------
if(TAX_BUILD_TEST)
    enable_testing()
    add_subdirectory(tests)
endif()

# -------------------------------
# Installation
# -------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(TAX_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/tax")

install(TARGETS tax
    EXPORT taxTargets
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT taxTargets
    FILE taxTargets.cmake
    NAMESPACE tax::
    DESTINATION ${TAX_INSTALL_CMAKEDIR}
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/taxConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/taxConfig.cmake"
    INSTALL_DESTINATION ${TAX_INSTALL_CMAKEDIR}
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/taxConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/taxConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/taxConfigVersion.cmake"
    DESTINATION ${TAX_INSTALL_CMAKEDIR}
)
