cmake_minimum_required(VERSION 4.2)
project(qlora_cpp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

add_library(qlora_core
        qlora.cc
)
target_include_directories(qlora_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(qlora_cpp main.cc)
target_link_libraries(qlora_cpp PRIVATE qlora_core)

add_executable(qlora_integration_test qlora_integration_test.cc)
target_link_libraries(qlora_integration_test
        PRIVATE
        qlora_core
        gtest_main  # Automatically provides a main() function for tests
)

include(GoogleTest)
gtest_discover_tests(qlora_integration_test)
