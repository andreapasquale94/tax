cmake_minimum_required(VERSION 3.28)
project(tax_install_consumer LANGUAGES CXX)

find_package(tax CONFIG REQUIRED)

add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE tax::tax)
