cmake_minimum_required(VERSION 3.28)

project(APPS VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# TensorRT NvInfer related libs
find_library(NVINFER_LIB nvinfer HINTS /usr/lib/x86_64-linux-gnu/ PATH_SUFFIXES lib lib64 REQUIRED)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS /usr/lib/x86_64-linux-gnu/ PATH_SUFFIXES lib lib64 REQUIRED)
find_library(NVONNXPARSER_LIB nvonnxparser HINTS /usr/lib/x86_64-linux-gnu/ PATH_SUFFIXES lib lib64 REQUIRED)

add_executable(build_engine build_engine.cpp)
target_link_libraries(build_engine PRIVATE ${NVINFER_LIB} ${NVONNXPARSER_LIB} ${NVINFER_PLUGIN_LIB} CUDA::cudart)

add_executable(run_engine run_engine.cpp)
target_link_libraries(run_engine PRIVATE ${NVINFER_LIB} CUDA::cudart)

