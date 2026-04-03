cmake_minimum_required(VERSION 4.2)
project(qlora_cpp)

set(CMAKE_CXX_STANDARD 20)

include_directories(.)

add_executable(qlora_cpp
        nf4_constants.h
        nf4_quantization.cc
        numeric_util.h
        quantized_data.h)
