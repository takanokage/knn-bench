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

# include kNN-CUDA
add_subdirectory(kNN-CUDA "${CMAKE_BINARY_DIR}/kNN-CUDA")
set(KNN_CUDA_INCLUDE_DIRS kNN-CUDA)

# include flann
# add_subdirectory(flann "${CMAKE_BINARY_DIR}/flann")
Directories(flann/src/cpp FLANN_INCLUDE_DIRS)

# include PQT
add_subdirectory(pqt "${CMAKE_BINARY_DIR}/pqt")
Directories(pqt PQT_INCLUDE_DIRS)

# include faiss
set(BUILD_TUTORIAL OFF CACHE BOOL "Build tutorials")
set(BUILD_TEST OFF CACHE BOOL "Build tests")
add_subdirectory(faiss "${CMAKE_BINARY_DIR}/faiss")
Directories(faiss FAISS_INCLUDE_DIRS)

# collect the source code
file(GLOB_RECURSE ALL_SOURCE_FILES "src/*.cpp")

# message(STATUS "******")
# message(STATUS "FAISS_INCLUDE_DIRS : ${FAISS_INCLUDE_DIRS}")
# message(STATUS "******")

# create executable
add_executable(${PROJECT_NAME}
    ${ALL_SOURCE_FILES})

include_directories(src
    ${KNN_CUDA_INCLUDE_DIRS}
    ${FLANN_INCLUDE_DIRS}
    ${FAISS_INCLUDE_DIRS}
    ${PQT_INCLUDE_DIRS}
    /usr/local/cuda/include/
    /usr/local/cuda-10.0/samples/common/inc)

target_link_libraries(${PROJECT_NAME} PRIVATE
    kNN-CUDA
    pqt
    gpufaiss
    faiss
    cublas lz4)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
