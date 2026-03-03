cmake_minimum_required(VERSION 4.2)

project(tax VERSION 0.1.0 LANGUAGES CXX)

# -------------------------------
# Options
# -------------------------------
option(TAX_BUILD_TEST "Build and enable unittests" ON)
option(TAX_ENABLE_EIGEN "Enable Eigen tensor adapters" OFF)
option(TAX_BUILD_BENCHMARK "Build benchmark suite (Google Benchmark)" OFF)
option(TAX_USE_DACE
       "Enable DACE integration (fetches dacelib/dace@v2.1.0) for tests and benchmarks"
       OFF)

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

# -------------------------------
# Optional Eigen dependency
# -------------------------------
if(TAX_ENABLE_EIGEN)
    find_package(Eigen3 QUIET)
    if(Eigen3_FOUND)
        target_link_libraries(tax INTERFACE Eigen3::Eigen)
        target_compile_definitions(tax INTERFACE TAX_ENABLE_EIGEN=1)
    else()
        message(STATUS "Eigen3 not found, disabling TAX_ENABLE_EIGEN")
        set(TAX_ENABLE_EIGEN OFF)
    endif()
endif()

# -------------------------------
# Optional DACE dependency
# -------------------------------
set(TAX_DACE_TARGET "" CACHE INTERNAL "Resolved DACE target for TAX_USE_DACE" FORCE)
if(TAX_USE_DACE)
    include(FetchContent)

    if(NOT TARGET dace AND NOT TARGET dace_s AND NOT TARGET dace::dace
       AND NOT TARGET dace::dace_s)
        FetchContent_Declare(
            DACE
            GIT_REPOSITORY https://github.com/dacelib/dace.git
            GIT_TAG v2.1.0
        )
        FetchContent_MakeAvailable(DACE)
    endif()

    # Match upstream recommendation from DACE docs:
    #   add_library(dace::dace ALIAS dace)
    #   add_library(dace::dace_s ALIAS dace_s)
    if(TARGET dace AND NOT TARGET dace::dace)
        add_library(dace::dace ALIAS dace)
    endif()
    if(TARGET dace_s AND NOT TARGET dace::dace_s)
        add_library(dace::dace_s ALIAS dace_s)
    endif()

    if(TARGET dace::dace_s)
        set(TAX_DACE_TARGET dace::dace_s)
    elseif(TARGET dace_s)
        set(TAX_DACE_TARGET dace_s)
    elseif(TARGET dace::dace)
        set(TAX_DACE_TARGET dace::dace)
    elseif(TARGET dace)
        set(TAX_DACE_TARGET dace)
    else()
        message(FATAL_ERROR
                "TAX_USE_DACE is ON but no DACE target was found after FetchContent(dacelib/dace@v2.1.0).")
    endif()

    set(TAX_DACE_TARGET "${TAX_DACE_TARGET}" CACHE INTERNAL
        "Resolved DACE target for TAX_USE_DACE" FORCE)
    message(STATUS "DACE enabled via TAX_USE_DACE. Using target: ${TAX_DACE_TARGET}")
endif()

# -------------------------------
# Tests
# -------------------------------
if(TAX_BUILD_TEST)
    enable_testing()
    add_subdirectory(tests)
endif()

if(TAX_BUILD_BENCHMARK)
    add_subdirectory(benchmarks)
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

# Configure whether installed package requires Eigen
if(TAX_ENABLE_EIGEN)
    set(TAX_CONFIG_EIGEN_DEP "find_dependency(Eigen3)")
else()
    set(TAX_CONFIG_EIGEN_DEP "")
endif()

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
