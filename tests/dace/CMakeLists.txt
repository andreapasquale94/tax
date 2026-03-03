set(DACE_TEST_SOURCES
    testUnivariate.cpp
    testMultivariate.cpp
)

set(_dace_target "")
if(TAX_DACE_TARGET AND TARGET ${TAX_DACE_TARGET})
    set(_dace_target ${TAX_DACE_TARGET})
elseif(TARGET dace::dace_s)
    set(_dace_target dace::dace_s)
elseif(TARGET dace_s)
    set(_dace_target dace_s)
elseif(TARGET dace::dace)
    set(_dace_target dace::dace)
elseif(TARGET dace)
    set(_dace_target dace)
else()
    message(FATAL_ERROR
            "tests/dace requires TAX_USE_DACE=ON with an available DaCe target.")
endif()

message(STATUS "DaCe target for tests/dace: ${_dace_target}")

include(GoogleTest)

foreach(test_file IN LISTS DACE_TEST_SOURCES)
    get_filename_component(test_name ${test_file} NAME_WE)

    add_executable(${test_name} ${test_file})
    target_link_libraries(${test_name}
        PRIVATE
            tax
            ${_dace_target}
            GTest::gtest_main
    )
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    gtest_discover_tests(${test_name})
endforeach()
