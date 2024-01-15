cmake_minimum_required(VERSION 3.28)

project(TENSORRT-PLUGIN-EXAMPLE VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# add_subdirectory(plugins_reference)

add_executable(main main.cpp)


# find_path(CUDA_GEMM_INCLUDE_DIRS cuda_gemm_utils.hpp HINTS ${CMAKE_SOURCE_DIR}/include)

# TensorRT NvInfer related libs
find_library(NVINFER_LIB nvinfer HINTS /usr/lib/x86_64-linux-gnu/ PATH_SUFFIXES lib lib64 REQUIRED)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS /usr/lib/x86_64-linux-gnu/ PATH_SUFFIXES lib lib64 REQUIRED)

# Add all the source files in the current directory to build the library
add_library(
    identity_conv 
    SHARED 
    ${CMAKE_CURRENT_SOURCE_DIR}/plugins/IdentityConvPlugin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plugins/IdentityConvPluginCreator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plugins/PluginUtils.cpp
)

# # Add the include directory of the library to the include directories of the project
target_include_directories(identity_conv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/plugins)

# # Set the CUDA architecture to compile the code for
# # https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
# set_target_properties(identity_conv PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(identity_conv PRIVATE ${NVINFER_LIB} ${NVINFER_PLUGIN_LIB} CUDA::cudart)



# install(TARGETS identity_conv DESTINATION lib)
# install(FILES ${CUDA_GEMM_HEADERS} DESTINATION include)

# add_executable(profile_cuda_gemm_fp32 profile_cuda_gemm_fp32.cu)
# target_link_libraries(profile_cuda_gemm_fp32 cuda_gemm CUDA::cublas)
# set_target_properties(profile_cuda_gemm_fp32 PROPERTIES CUDA_ARCHITECTURES native)

# add_executable(profile_cuda_gemm_fp16 profile_cuda_gemm_fp16.cu)
# target_link_libraries(profile_cuda_gemm_fp16 cuda_gemm CUDA::cublas)
# set_target_properties(profile_cuda_gemm_fp16 PROPERTIES CUDA_ARCHITECTURES native)