cmake_minimum_required(VERSION 3.15)
project(VideoPipeline)

set(CMAKE_CXX_STANDARD 17)

OPTION(ENABLE_CUDA "使用CUDA,TensorRT" ON)
OPTION(ENABLE_TENSORRT "使用TensorRT" ON)
OPTION(ENABLE_HWCODEC "使用硬件编解码能力" OFF)
option(ENABLE_TEST "编译测试代码" ON)




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