cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

set (CMAKE_CXX_STANDARD 11)

project(knn-bench CXX CUDA)

# help debug this CMakeLists.txt
# set(CMAKE_VERBOSE_MAKEFILE ON)

add_subdirectory(kNN-CUDA "${CMAKE_BINARY_DIR}/lib")

# create executable
add_executable(${PROJECT_NAME} main.cpp init.cpp ref.cpp test.cpp)
include_directories(kNN-CUDA)
target_link_libraries(${PROJECT_NAME} PRIVATE kNN-CUDA cublas)
set_target_properties(${PROJECT_NAME} PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON
                      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
