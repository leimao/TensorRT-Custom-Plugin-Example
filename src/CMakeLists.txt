cmake_minimum_required(VERSION 3.28)

project(TENSORRT-CUSTOM-PLUGIN-IDENTITY-CONV-EXAMPLE VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(plugins)
add_subdirectory(apps)
