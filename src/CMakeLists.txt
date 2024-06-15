cmake_minimum_required(VERSION 3.28)

project(TENSORRT-CUSTOM-PLUGIN-IDENTITY-CONV-EXAMPLE VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# add_executable(build_engine build_engine.cpp)
# target_link_libraries(build_engine PRIVATE ${NVINFER_LIB} ${NVONNXPARSER_LIB} ${NVINFER_PLUGIN_LIB} CUDA::cudart)

# add_executable(run_engine run_engine.cpp)
# target_link_libraries(run_engine PRIVATE ${NVINFER_LIB} CUDA::cudart)

add_subdirectory(plugins)
add_subdirectory(apps)
