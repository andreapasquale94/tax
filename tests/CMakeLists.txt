include(FetchContent)

find_package(googletest 1.17 QUIET)

if(NOT googletest_FOUND)
    message(STATUS "googletest not found, fetching via FetchContent...")
    include(FetchContent)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
    )

    # Prevent installation rules from polluting the parent project.
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# tests/CMakeLists.txt

# Collect all tests
set(TEST_SOURCES
    testCombinatorics.cpp
    testKernels.cpp
    testDA.cpp
    testExprAdd.cpp
    testExprMul.cpp
    testExprUnary.cpp
    testExprPower.cpp
    testExprTrig.cpp
    testCompose.cpp
)

include(GoogleTest)

foreach(test_file IN LISTS TEST_SOURCES)
    # Extract the file name without extension for the test target
    get_filename_component(test_name ${test_file} NAME_WE)

    # Define the test executable
    add_executable(${test_name} ${test_file})

    # Link against your library + gtest
    target_link_libraries(${test_name}
        PRIVATE
            tax
            GTest::gtest_main
    )

    # Register with CTest
    gtest_discover_tests(${test_name})
endforeach()