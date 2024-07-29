cmake_minimum_required(VERSION 3.15)

project(VideoPipeline)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-std=c++17 -lpthread)

OPTION(ENABLE_CUDA "使用CUDA,TensorRT" ON)
OPTION(ENABLE_TENSORRT "使用TensorRT" OFF)
option(ENABLE_TEST "编译测试代码" ON)
option(ENABLE_TRITON "使用Triton" ON)



# 基础代码依赖配置
include(${CMAKE_SOURCE_DIR}/cmake/base.cmake)
if (ENABLE_CUDA)
    # CUDA依赖配置
    include(${CMAKE_SOURCE_DIR}/cmake/cuda.cmake)
endif ()
if (ENABLE_TENSORRT)
    # TensorRT依赖配置
    include(${CMAKE_SOURCE_DIR}/cmake/tensorrt.cmake)
endif ()
if(ENABLE_TEST)
    # 测试代码依赖配置
    include(${CMAKE_SOURCE_DIR}/cmake/test.cmake)
endif()
if(ENABLE_TRITON)
    # Triton依赖配置
    include(${CMAKE_SOURCE_DIR}/cmake/triton.cmake)
endif()