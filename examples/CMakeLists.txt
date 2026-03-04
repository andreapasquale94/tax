add_executable(two_body two_body.cpp)
target_link_libraries(two_body PRIVATE tax::tax)

add_executable(two_body_elliptic two_body_elliptic.cpp)
target_link_libraries(two_body_elliptic PRIVATE tax::tax)
