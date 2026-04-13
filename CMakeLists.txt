cmake_minimum_required(VERSION 4.0)
project(qlora_cpp)

# Global C++ Standard Settings
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Setup GoogleTest
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
# For GTest, we typically want to disable its internal warnings to keep our build clean
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Define the Core Library
add_library(qlora_core INTERFACE)
target_include_directories(qlora_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
# Requires C++23 features for this library and its consumers
target_compile_features(qlora_core INTERFACE cxx_std_23)

# Main Executable
add_executable(qlora_cpp main.cc)
target_link_libraries(qlora_cpp PRIVATE qlora_core)

# Testing
add_executable(all_tests
        quantization_test.cc)
target_link_libraries(all_tests
        PRIVATE
        qlora_core
        gtest_main  # Automatically provides a main() function for tests
)

target_compile_options(all_tests PRIVATE -mavx2 -mfma)
target_compile_options(qlora_cpp PRIVATE -mavx2 -mfma)

# Enable automatic test discovery
include(GoogleTest)
gtest_discover_tests(all_tests)
