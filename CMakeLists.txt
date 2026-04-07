cmake_minimum_required(VERSION 4.0)
project(qlora_cpp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

add_library(qlora_core INTERFACE
        lora_linear_layer.cc lora_linear_layer.h
)
target_include_directories(qlora_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(qlora_core INTERFACE qlora.h)

add_executable(qlora_cpp main.cc)
target_link_libraries(qlora_cpp PRIVATE qlora_core)

add_executable(integration_test integration_test.cc)
target_link_libraries(integration_test
        PRIVATE
        qlora_core
        gtest_main  # Automatically provides a main() function for tests
)

include(GoogleTest)
gtest_discover_tests(integration_test)
