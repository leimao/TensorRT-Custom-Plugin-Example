cmake_minimum_required(VERSION 3.5.0)

project(Toy VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(IdentityConvIPluginV2IOExt)
# IPluginV3 is only supported in TensorRT 10.0.0 and later.
add_subdirectory(IdentityConvIPluginV3)
