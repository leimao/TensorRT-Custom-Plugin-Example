cmake_minimum_required(VERSION 3.28)

project(TENSORRT-PLUGIN-EXAMPLE VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# TensorRT NvInfer related libs
find_library(NVINFER_LIB nvinfer HINTS /usr/lib/x86_64-linux-gnu/ PATH_SUFFIXES lib lib64 REQUIRED)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS /usr/lib/x86_64-linux-gnu/ PATH_SUFFIXES lib lib64 REQUIRED)
find_library(NVONNXPARSER_LIB nvonnxparser HINTS /usr/lib/x86_64-linux-gnu/ PATH_SUFFIXES lib lib64 REQUIRED)

# Add all the source files in the current directory to build the library
add_library(
    identity_conv 
    SHARED 
    ${CMAKE_CURRENT_SOURCE_DIR}/plugins/IdentityConvPlugin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plugins/IdentityConvPluginCreator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plugins/PluginUtils.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/plugins/PluginRegistration.cpp
)

target_include_directories(identity_conv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/plugins)
target_link_libraries(identity_conv ${NVINFER_LIB} ${NVINFER_PLUGIN_LIB} CUDA::cudart)

add_executable(build_engine build_engine.cpp)
target_link_libraries(build_engine ${NVINFER_LIB} ${NVONNXPARSER_LIB} ${NVINFER_PLUGIN_LIB} CUDA::cudart identity_conv)

add_executable(run_engine run_engine.cpp)
target_link_libraries(run_engine ${NVINFER_LIB} CUDA::cudart identity_conv)

# add_subdirectory(ref_plugin)