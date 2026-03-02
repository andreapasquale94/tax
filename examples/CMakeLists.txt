# examples/CMakeLists.txt
# Only built when TAX_ENABLE_EIGEN is ON (ODE header requires Eigen)

if(NOT TAX_ENABLE_EIGEN)
    message(STATUS "TAX_BUILD_EXAMPLES requires TAX_ENABLE_EIGEN=ON — skipping examples")
    return()
endif()

add_executable(two_body two_body.cpp)
target_link_libraries(two_body PRIVATE tax::tax)
