cmake_minimum_required(VERSION 3.10)
project(torch_inference_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "/libtorch")

find_package(Torch REQUIRED)

add_executable(torch_inference libtorch_pred.cpp)
target_link_libraries(torch_inference "${TORCH_LIBRARIES}")

set_property(TARGET torch_inference PROPERTY CXX_STANDARD 17)