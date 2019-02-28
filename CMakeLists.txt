cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

set (CMAKE_CXX_STANDARD 11)

project(knn-bench CXX CUDA)

# recursively parse and return the entire directory tree.
# the result is placed in output
function(Directories root output)
    set(data "")
    list(APPEND data ${root})
    file(GLOB_RECURSE children LIST_DIRECTORIES true "${root}/*")
    list(SORT children)
    foreach(child ${children})
        if (IS_DIRECTORY ${child})
            list(APPEND data ${child})
        endif()
    endforeach()
    set (${output} ${data} PARENT_SCOPE)
endfunction()

# help debug this CMakeLists.txt
# set(CMAKE_VERBOSE_MAKEFILE ON)

add_subdirectory(kNN-CUDA "${CMAKE_BINARY_DIR}/kNN-CUDA")
set(KNN_CUDA_INCLUDE_DIRS kNN-CUDA)

Directories(flann/src/cpp FLANN_INCLUDE_DIRS)

# collect the source code
file(GLOB_RECURSE ALL_SOURCE_FILES "src/*.cpp")

# message(STATUS "******")
# message(STATUS "ALL_SOURCE_FILES : ${ALL_SOURCE_FILES}")
# message(STATUS "******")

# create executable
add_executable(${PROJECT_NAME}
    ${ALL_SOURCE_FILES})

include_directories(src
    ${KNN_CUDA_INCLUDE_DIRS}
    ${FLANN_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PRIVATE kNN-CUDA cublas lz4)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
