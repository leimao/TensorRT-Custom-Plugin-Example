cmake_minimum_required(VERSION 3.28)

project(IDENTITY-CONV-IPLUGIN-V2-IOEXT VERSION 0.0.1 LANGUAGES CXX)

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
    identity_conv_iplugin_v2_io_ext
    SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/../PluginUtils/PluginUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/IdentityConvPlugin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/IdentityConvPluginCreator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PluginRegistration.cpp
)

target_include_directories(identity_conv_iplugin_v2_io_ext PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../PluginUtils)
target_link_libraries(identity_conv_iplugin_v2_io_ext PRIVATE ${NVINFER_LIB} ${NVINFER_PLUGIN_LIB} CUDA::cudart)