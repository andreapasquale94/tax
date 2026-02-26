cmake_minimum_required(VERSION 4.2)

project(tax VERSION 0.1.0 LANGUAGES CXX)

# -------------------------------
# Options
# -------------------------------
option(TAX_BUILD_TEST "Build and enable unittests" ON)
option(TAX_BUILD_BENCH "Build benchmarks" ON)

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
target_link_libraries(tax INTERFACE)

target_include_directories(tax
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

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
# TODO