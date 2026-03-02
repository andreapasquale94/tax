cmake_minimum_required(VERSION 4.2)

project(tax VERSION 0.1.0 LANGUAGES CXX)

# -------------------------------
# Options
# -------------------------------
option(TAX_BUILD_TEST "Build and enable unittests" ON)
option(TAX_ENABLE_EIGEN "Enable Eigen tensor adapters" OFF)
option(TAX_BUILD_EXAMPLES "Build example programs" OFF)

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
# Tests
# -------------------------------
if(TAX_BUILD_TEST)
    enable_testing()
    add_subdirectory(tests)
endif()

# -------------------------------
# Examples
# -------------------------------
if(TAX_BUILD_EXAMPLES)
    add_subdirectory(examples)
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
