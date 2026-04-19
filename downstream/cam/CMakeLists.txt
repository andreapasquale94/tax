cmake_minimum_required(VERSION 3.28)

project(cam VERSION 0.1.0 LANGUAGES CXX)

# Standalone-or-subdirectory detection ----------------------------------------
if(NOT DEFINED CAM_BUILD_STANDALONE)
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        set(CAM_BUILD_STANDALONE ON)
    else()
        set(CAM_BUILD_STANDALONE OFF)
    endif()
endif()

option(CAM_BUILD_APPS  "Build CAM demo apps"  ON)
option(CAM_BUILD_TESTS "Build CAM tests"      ${CAM_BUILD_STANDALONE})

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Resolve a tax target --------------------------------------------------------
# Three ways to find tax:
#   1) configured as a subdirectory alongside downstream/cam (typical layout)
#   2) a user-provided CMAKE_PREFIX_PATH pointing to an installed tax
#   3) fallback: FetchContent from GitHub

if(NOT TARGET tax::tax)
    # Option 1: this repo's tax/ is our parent's parent
    set(_tax_parent "${CMAKE_CURRENT_SOURCE_DIR}/../..")
    if(EXISTS "${_tax_parent}/CMakeLists.txt" AND EXISTS "${_tax_parent}/include/tax/tax.hpp")
        message(STATUS "[cam] Using sibling tax at ${_tax_parent}")
        # Avoid pulling tax's own tests into our ctest run.
        set(TAX_BUILD_TEST OFF CACHE BOOL "" FORCE)
        set(TAX_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
        add_subdirectory("${_tax_parent}" "${CMAKE_CURRENT_BINARY_DIR}/_tax" EXCLUDE_FROM_ALL)
    else()
        find_package(tax QUIET)
        if(NOT TARGET tax::tax)
            include(FetchContent)
            FetchContent_Declare(tax
                GIT_REPOSITORY https://github.com/andreapasquale94/tax.git
                GIT_TAG        main
            )
            FetchContent_MakeAvailable(tax)
        endif()
    endif()
endif()

# The cam interface library --------------------------------------------------
add_library(cam INTERFACE)
add_library(cam::cam ALIAS cam)

target_include_directories(cam INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(cam INTERFACE cxx_std_23)
target_link_libraries(cam INTERFACE tax::tax)

# Apps -----------------------------------------------------------------------
if(CAM_BUILD_APPS)
    add_executable(cam_state_prop apps/state_prop.cpp)
    target_link_libraries(cam_state_prop PRIVATE cam::cam)
    set_target_properties(cam_state_prop PROPERTIES OUTPUT_NAME state_prop)

    add_executable(cam_state_back_prop apps/state_back_prop.cpp)
    target_link_libraries(cam_state_back_prop PRIVATE cam::cam)
    set_target_properties(cam_state_back_prop PROPERTIES OUTPUT_NAME state_back_prop)

    add_executable(cam_scvx apps/scvx_main.cpp)
    target_link_libraries(cam_scvx PRIVATE cam::cam)
    set_target_properties(cam_scvx PROPERTIES OUTPUT_NAME scvx_main)
endif()

# Tests ----------------------------------------------------------------------
if(CAM_BUILD_TESTS)
    enable_testing()

    add_executable(test_kepler tests/test_kepler.cpp)
    target_link_libraries(test_kepler PRIVATE cam::cam)
    add_test(NAME test_kepler COMMAND test_kepler)

    add_executable(test_tca tests/test_tca.cpp)
    target_link_libraries(test_tca PRIVATE cam::cam)
    add_test(NAME test_tca COMMAND test_tca)

    add_executable(test_scvx tests/test_scvx.cpp)
    target_link_libraries(test_scvx PRIVATE cam::cam)
    add_test(NAME test_scvx COMMAND test_scvx)
endif()

# Install --------------------------------------------------------------------
include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS cam EXPORT camTargets)
