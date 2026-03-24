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
    foundation/testCombinatorics.cpp
    foundation/testEnumeration.cpp
    kernels/testKernels.cpp
    core/testTTE.cpp
    core/testNested.cpp
    core/testCompose.cpp
    core/testDerivInteg.cpp
    expr/testExprAdd.cpp
    expr/testExprMul.cpp
    expr/testExprUnary.cpp
    expr/testExprPower.cpp
    expr/testExprTrig.cpp
    expr/testExprLog.cpp
    expr/testExprPow.cpp
    expr/testExprHyp.cpp
    expr/testExprInvHyp.cpp
    expr/testExprInvTrig.cpp
    expr/testExprErf.cpp
    expr/testExprFMA.cpp
    eigen/testVariables.cpp
    eigen/testValue.cpp
    eigen/testEval.cpp
    eigen/testDerivative.cpp
    eigen/testTypes.cpp
    eigen/testInvertMap.cpp
    ads/testAdsTree.cpp
    ads/testAdsGaussian.cpp
    ode/testTaylorIntegrator.cpp
    ode/testIntegrateAds.cpp
    chebyshev/testChebyshevBasic.cpp
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
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    # Register with CTest
    gtest_discover_tests(${test_name})
endforeach()

if(TAX_USE_DACE)
    add_subdirectory(dace)
endif()
